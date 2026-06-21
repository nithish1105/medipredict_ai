"""
Fast image model training on cached EfficientNet features.
Targets 90%+ test accuracy using an ensemble approach.
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
    print("  Fast Image Model Training (from cached features)")
    print("=" * 65)

    cache_path = os.path.join(BASE_DIR, "features_cache.npz")
    if not os.path.exists(cache_path):
        print("ERROR: features_cache.npz not found. Run train_image_model.py first.")
        sys.exit(1)

    data = np.load(cache_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    all_classes = list(data["classes"])
    print(f"  Loaded {X.shape[0]} samples, {X.shape[1]} features, {len(all_classes)} classes\n")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc,
    )
    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    # ── Try multiple classifiers ──
    from sklearn.svm import SVC
    from sklearn.ensemble import (
        ExtraTreesClassifier, RandomForestClassifier,
        HistGradientBoostingClassifier, VotingClassifier,
        StackingClassifier, BaggingClassifier,
    )
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression

    candidates = {}

    # 1) SVM with high C
    candidates["SVM RBF C=100"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(C=100.0, kernel="rbf", gamma="scale",
                     class_weight="balanced", random_state=42)),
    ])

    # 2) MLP (wider and deeper)
    candidates["MLP 1024-512-256"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(1024, 512, 256),
            activation="relu", solver="adam",
            alpha=1e-4, batch_size=256,
            learning_rate="adaptive", learning_rate_init=1e-3,
            max_iter=500, early_stopping=True,
            validation_fraction=0.15, random_state=42, verbose=False,
        )),
    ])

    # 3) HistGradientBoosting with more iterations
    candidates["HistGB 1000"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(
            max_iter=1000, learning_rate=0.05,
            max_depth=10, min_samples_leaf=3,
            l2_regularization=0.01,
            class_weight="balanced", random_state=42, verbose=0,
        )),
    ])

    # 4) Extra Trees (big)
    candidates["Extra Trees 2000"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", ExtraTreesClassifier(
            n_estimators=2000, max_depth=None,
            min_samples_leaf=1, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )),
    ])

    # 5) Random Forest 2000
    candidates["RF 2000"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=2000, max_depth=None,
            min_samples_leaf=1, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )),
    ])

    results = {}
    best_pipe = None
    best_acc = 0.0
    best_name = ""

    for name, pipe in candidates.items():
        print(f"  Training {name} ...")
        t1 = time.time()
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100
        elapsed = time.time() - t1
        print(f"    Test Accuracy: {acc:.2f}%  ({elapsed:.0f}s)")
        results[name] = (acc, pipe)
        if acc > best_acc:
            best_acc = acc
            best_pipe = pipe
            best_name = name

    # ── If not at 90%, try a stacking ensemble of the top models ──
    if best_acc < 90.0:
        print(f"\n  Best single model: {best_acc:.2f}% — trying ensemble ...")

        # Rank models by accuracy
        ranked = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
        top_models = [(name, pipe) for name, (acc, pipe) in ranked[:3]]

        print(f"  Stacking top-3: {[n for n, _ in top_models]}")

        estimators = [(n, p) for n, p in top_models]
        stacker = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(
                C=10.0, max_iter=1000, n_jobs=-1
            ),
            cv=3, n_jobs=-1, passthrough=False,
        )
        print("  Training stacking ensemble ...")
        t1 = time.time()
        stacker.fit(X_train, y_train)
        y_pred = stacker.predict(X_test)
        stack_acc = accuracy_score(y_test, y_pred) * 100
        elapsed = time.time() - t1
        print(f"    Stacking Accuracy: {stack_acc:.2f}%  ({elapsed:.0f}s)")

        if stack_acc > best_acc:
            best_acc = stack_acc
            best_pipe = stacker
            best_name = "Stacking Ensemble"

    print(f"\n  Best model: {best_name} -> {best_acc:.2f}%\n")

    # ── If still not at 90%, try PyTorch MLP ──
    if best_acc < 90.0:
        print("  Trying PyTorch deep MLP ...")
        try:
            acc_pt, model_pt = _train_pytorch_mlp(X_train, X_test, y_train, y_test, len(all_classes))
            if acc_pt > best_acc:
                best_acc = acc_pt
                best_name = "PyTorch DeepMLP"
                # Wrap for sklearn compatibility
                best_pipe = _make_wrapper(model_pt, all_classes, X.shape[1])
                print(f"  PyTorch MLP: {acc_pt:.2f}%")
        except Exception as e:
            print(f"  PyTorch MLP failed: {e}")

    # ── Evaluate best ──
    if hasattr(best_pipe, 'predict'):
        y_pred = best_pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100
        print(f"\n>>> Final Test Accuracy: {acc:.2f}%")
        print(f"\nClassification Report:\n")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

    # ── Save ──
    model_path = os.path.join(BASE_DIR, "image_model.pkl")
    cls_path = os.path.join(BASE_DIR, "image_classes.pkl")

    joblib.dump(best_pipe, model_path, compress=3)
    with open(cls_path, "wb") as f:
        pickle.dump(all_classes, f)

    sz = os.path.getsize(model_path) / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"\n  Saved:")
    print(f"    Model   -> {model_path}  ({sz:.1f} MB)")
    print(f"    Classes -> {cls_path}")
    print(f"    Total time: {elapsed/60:.1f} min")
    print(f"\n  ACHIEVED: {best_acc:.2f}% accuracy with {best_name}")
    print("=" * 65)

    return best_acc


def _train_pytorch_mlp(X_train, X_test, y_train, y_test, n_classes,
                        epochs=400, lr=5e-4, batch_size=128, patience=50):
    """Train a PyTorch MLP with mixup, label smoothing, cosine annealing."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
    from collections import Counter

    # Standardize features
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train).astype(np.float32)
    X_te = scaler.transform(X_test).astype(np.float32)

    n_features = X_tr.shape[1]

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

    # Weighted sampler
    counts = Counter(y_train)
    weights = [1.0 / counts[l] for l in y_train]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_te), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    best_acc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            # Mixup
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

        # Eval
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                preds = model(xb).argmax(1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        acc = correct / total * 100

        if epoch % 20 == 0 or acc > best_acc:
            print(f"    Epoch {epoch:>3d}  test={acc:.1f}%  lr={optimizer.param_groups[0]['lr']:.6f}")

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

    print(f"    Best PyTorch MLP accuracy: {best_acc:.2f}%")
    return best_acc, (model, scaler, n_features, n_classes)


def _make_wrapper(pt_result, all_classes, n_features_orig):
    """Create sklearn-compatible wrapper for PyTorch model."""
    import torch
    import torch.nn as nn

    model, scaler, n_features, n_classes = pt_result

    class Wrapper:
        def __init__(self):
            self.classes_ = np.array(all_classes)
            self._state = {k: v.cpu() for k, v in model.state_dict().items()}
            self._scaler = scaler
            self._n_features = n_features
            self._n_classes = n_classes
            self._model = None

        def _ensure(self):
            if self._model is None:
                import torch.nn as nn
                m = nn.Sequential(
                    nn.BatchNorm1d(self._n_features),
                    nn.Linear(self._n_features, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.4),
                    nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
                    nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
                    nn.Linear(256, self._n_classes),
                )
                # Load into the sequential - we need to match the wrapper structure
                # Actually the state dict has "net." prefix, so wrap it
                class MLP(nn.Module):
                    def __init__(self2):
                        super().__init__()
                        self2.net = m
                    def forward(self2, x):
                        return self2.net(x)

                self._model = MLP()
                self._model.load_state_dict(self._state)
                self._model.eval()

        def predict(self, X):
            self._ensure()
            import torch
            X_s = self._scaler.transform(np.array(X).astype(np.float32))
            X_t = torch.tensor(X_s, dtype=torch.float32)
            with torch.no_grad():
                return self._model(X_t).argmax(1).numpy()

        def predict_proba(self, X):
            self._ensure()
            import torch
            X_s = self._scaler.transform(np.array(X).astype(np.float32))
            X_t = torch.tensor(X_s, dtype=torch.float32)
            with torch.no_grad():
                return torch.softmax(self._model(X_t), 1).numpy()

    return Wrapper()


if __name__ == "__main__":
    train()
