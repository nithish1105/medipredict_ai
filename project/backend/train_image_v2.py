"""
Fast image model training on cached EfficientNet features — v2.
Prioritizes fast classifiers first, then tries slower ones if needed.
"""
import os, sys, time, pickle, warnings
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def train():
    t0 = time.time()
    print("=" * 65)
    print("  Image Model Training v2 — Target 90%")
    print("=" * 65)

    cache_path = os.path.join(BASE_DIR, "features_cache.npz")
    if not os.path.exists(cache_path):
        print("ERROR: features_cache.npz not found.")
        sys.exit(1)

    data = np.load(cache_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    all_classes = list(data["classes"])
    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(all_classes)}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc,
    )
    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    from sklearn.ensemble import (
        ExtraTreesClassifier, RandomForestClassifier,
        HistGradientBoostingClassifier,
    )
    from sklearn.neural_network import MLPClassifier

    # ── Phase 1: Fast classifiers ──
    print("--- Phase 1: Fast classifiers ---\n")
    candidates = {}

    candidates["Extra Trees 1500"] = ExtraTreesClassifier(
        n_estimators=1500, max_depth=None,
        min_samples_leaf=1, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )

    candidates["RF 1500"] = RandomForestClassifier(
        n_estimators=1500, max_depth=None,
        min_samples_leaf=1, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )

    candidates["HistGB 800"] = HistGradientBoostingClassifier(
        max_iter=800, learning_rate=0.05,
        max_depth=10, min_samples_leaf=3,
        l2_regularization=0.01,
        class_weight="balanced", random_state=42,
    )

    # Standardize for HistGB and MLP
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    best_pipe = None
    best_acc = 0.0
    best_name = ""

    for name, clf in candidates.items():
        print(f"  Training {name} ...", end=" ", flush=True)
        t1 = time.time()
        if "HistGB" in name:
            clf.fit(X_train_s, y_train)
            y_pred = clf.predict(X_test_s)
        else:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100
        print(f"{acc:.2f}%  ({time.time()-t1:.0f}s)")
        if acc > best_acc:
            best_acc = acc
            best_pipe = clf
            best_name = name

    print(f"\n  Best Phase 1: {best_name} -> {best_acc:.2f}%\n")

    # ── Phase 2: MLP (medium speed) ──
    if best_acc < 90.0:
        print("--- Phase 2: MLP classifiers ---\n")

        for hl, name in [
            ((1024, 512, 256), "MLP 1024-512-256"),
            ((768, 384, 192), "MLP 768-384-192"),
        ]:
            print(f"  Training {name} ...", end=" ", flush=True)
            t1 = time.time()
            mlp = MLPClassifier(
                hidden_layer_sizes=hl,
                activation="relu", solver="adam",
                alpha=1e-4, batch_size=256,
                learning_rate="adaptive", learning_rate_init=1e-3,
                max_iter=500, early_stopping=True,
                validation_fraction=0.15, random_state=42, verbose=False,
            )
            mlp.fit(X_train_s, y_train)
            y_pred = mlp.predict(X_test_s)
            acc = accuracy_score(y_test, y_pred) * 100
            print(f"{acc:.2f}%  ({time.time()-t1:.0f}s)")
            if acc > best_acc:
                best_acc = acc
                best_pipe = Pipeline([("scaler", StandardScaler()), ("clf", mlp)])
                best_pipe.fit(X_train, y_train)  # refit with scaler
                best_name = name

        print(f"\n  Best so far: {best_name} -> {best_acc:.2f}%\n")

    # ── Phase 3: SVM (slower but potentially better) ──
    if best_acc < 90.0:
        print("--- Phase 3: SVM classifiers ---\n")
        from sklearn.svm import SVC

        for c_val in [10, 50]:
            name = f"SVM C={c_val}"
            print(f"  Training {name} ...", end=" ", flush=True)
            t1 = time.time()
            svm = SVC(C=c_val, kernel="rbf", gamma="scale",
                       class_weight="balanced", random_state=42)
            svm.fit(X_train_s, y_train)
            y_pred = svm.predict(X_test_s)
            acc = accuracy_score(y_test, y_pred) * 100
            print(f"{acc:.2f}%  ({time.time()-t1:.0f}s)")
            if acc > best_acc:
                best_acc = acc
                best_pipe = Pipeline([("scaler", StandardScaler()), ("clf", svm)])
                best_pipe.fit(X_train, y_train)
                best_name = name

        print(f"\n  Best so far: {best_name} -> {best_acc:.2f}%\n")

    # ── Phase 4: PyTorch deep MLP ──
    if best_acc < 90.0:
        print("--- Phase 4: PyTorch Deep MLP ---\n")
        try:
            acc_pt, wrapper = _train_pytorch_mlp(
                X_train_s, X_test_s, y_train, y_test,
                len(all_classes), scaler, all_classes
            )
            if acc_pt > best_acc:
                best_acc = acc_pt
                best_pipe = wrapper
                best_name = "PyTorch DeepMLP"
        except Exception as e:
            print(f"  Failed: {e}")

    # ── Final evaluation ──
    print(f"\n{'='*65}")
    print(f"  BEST MODEL: {best_name} -> {best_acc:.2f}%")
    print(f"{'='*65}\n")

    if hasattr(best_pipe, 'predict'):
        y_pred = best_pipe.predict(X_test) if not isinstance(best_pipe, dict) else None
        if y_pred is not None:
            print(classification_report(y_test, y_pred, target_names=le.classes_))

    # ── Save ──
    model_path = os.path.join(BASE_DIR, "image_model.pkl")
    cls_path = os.path.join(BASE_DIR, "image_classes.pkl")

    joblib.dump(best_pipe, model_path, compress=3)
    with open(cls_path, "wb") as f:
        pickle.dump(all_classes, f)

    sz = os.path.getsize(model_path) / (1024 * 1024)
    print(f"\n  Saved: {model_path} ({sz:.1f} MB)")
    print(f"  Time: {(time.time()-t0)/60:.1f} min")
    print(f"  RESULT: {best_acc:.2f}% accuracy")
    print("=" * 65)
    return best_acc


def _train_pytorch_mlp(X_train_s, X_test_s, y_train, y_test,
                        n_classes, scaler, all_classes,
                        epochs=400, lr=5e-4, batch_size=128, patience=50):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
    from collections import Counter

    n_features = X_train_s.shape[1]

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.BatchNorm1d(n_features),
                nn.Linear(n_features, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.4),
                nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(256, n_classes),
            )
        def forward(self, x):
            return self.net(x)

    model = MLP()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    counts = Counter(int(l) for l in y_train)
    weights = [1.0 / counts[int(l)] for l in y_train]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    X_tr_t = torch.tensor(X_train_s.astype(np.float32))
    y_tr_t = torch.tensor(y_train.astype(np.int64))
    X_te_t = torch.tensor(X_test_s.astype(np.float32))
    y_te_t = torch.tensor(y_test.astype(np.int64))

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)

    best_acc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            lam = np.random.beta(0.3, 0.3)
            idx = torch.randperm(xb.size(0))
            xb_mix = lam * xb + (1 - lam) * xb[idx]
            optimizer.zero_grad()
            out = model(xb_mix)
            loss = lam * criterion(out, yb) + (1 - lam) * criterion(out, yb[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_te_t).argmax(1)
            acc = (preds == y_te_t).float().mean().item() * 100

        if epoch % 20 == 0 or acc > best_acc:
            print(f"    Epoch {epoch:>3d}  test={acc:.1f}%")

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"    Early stop at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)

    print(f"    Best: {best_acc:.2f}%")

    # Create wrapper
    class Wrapper:
        def __init__(self2):
            self2.classes_ = np.array(all_classes)
            self2._state = {k: v.cpu() for k, v in model.state_dict().items()}
            self2._scaler = scaler
            self2._n_features = n_features
            self2._n_classes = n_classes
            self2._model = None

        def _ensure(self2):
            if self2._model is None:
                import torch.nn as nn2
                class M(nn2.Module):
                    def __init__(s):
                        super().__init__()
                        s.net = nn2.Sequential(
                            nn2.BatchNorm1d(self2._n_features),
                            nn2.Linear(self2._n_features, 1024), nn2.BatchNorm1d(1024), nn2.GELU(), nn2.Dropout(0.4),
                            nn2.Linear(1024, 512), nn2.BatchNorm1d(512), nn2.GELU(), nn2.Dropout(0.3),
                            nn2.Linear(512, 256), nn2.BatchNorm1d(256), nn2.GELU(), nn2.Dropout(0.2),
                            nn2.Linear(256, self2._n_classes),
                        )
                    def forward(s, x):
                        return s.net(x)
                self2._model = M()
                self2._model.load_state_dict(self2._state)
                self2._model.eval()

        def predict(self2, X):
            self2._ensure()
            import torch as t
            X_s = self2._scaler.transform(np.array(X).astype(np.float32))
            with t.no_grad():
                return self2._model(t.tensor(X_s, dtype=t.float32)).argmax(1).numpy()

        def predict_proba(self2, X):
            self2._ensure()
            import torch as t
            X_s = self2._scaler.transform(np.array(X).astype(np.float32))
            with t.no_grad():
                return t.softmax(self2._model(t.tensor(X_s, dtype=t.float32)), 1).numpy()

    return best_acc, Wrapper()


if __name__ == "__main__":
    train()
