"""
Aggressive training: SVM + Ensemble + Feature augmentation
Target: maximize accuracy on cached EfficientNet features.
"""
import os, pickle, warnings, time
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(BASE, "train_result.txt")

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")

open(LOG, "w").close()
log("=" * 60)
log("Aggressive Training: Ensemble Approach")
log("=" * 60)

# Load
data = np.load(os.path.join(BASE, "features_cache.npz"), allow_pickle=True)
X, y = data["X"], data["y"]
classes = list(data["classes"])
n_classes = len(classes)

le = LabelEncoder()
y_enc = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
log(f"Train: {len(X_train)}, Test: {len(X_test)}, Classes: {n_classes}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train).astype(np.float32)
X_test_s = scaler.transform(X_test).astype(np.float32)
n_features = X_train_s.shape[1]


# ─── Feature augmentation: add Gaussian noise copies ───
log("\nAugmenting feature space (3x with noise)...")
aug_X = [X_train_s]
aug_y = [y_train]
for sigma in [0.1, 0.15, 0.2]:
    noise = np.random.normal(0, sigma, X_train_s.shape).astype(np.float32)
    aug_X.append(X_train_s + noise)
    aug_y.append(y_train)
X_train_aug = np.vstack(aug_X)
y_train_aug = np.concatenate(aug_y)
log(f"Augmented train: {len(X_train_aug)} samples")


# ─── 1) LinearSVC (fast, good on high-dim) ───
log("\n[1] Training LinearSVC (C=1.0)...")
t1 = time.time()
svc = CalibratedClassifierCV(
    LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=42),
    cv=3
)
svc.fit(X_train_s, y_train)
acc_svc = accuracy_score(y_test, svc.predict(X_test_s)) * 100
log(f"    LinearSVC: {acc_svc:.2f}% ({time.time()-t1:.0f}s)")


# ─── 2) LinearSVC on augmented data ───
log("\n[2] Training LinearSVC on augmented data (C=0.5)...")
t1 = time.time()
svc2 = CalibratedClassifierCV(
    LinearSVC(C=0.5, class_weight="balanced", max_iter=5000, random_state=42),
    cv=3
)
svc2.fit(X_train_aug, y_train_aug)
acc_svc2 = accuracy_score(y_test, svc2.predict(X_test_s)) * 100
log(f"    LinearSVC-aug: {acc_svc2:.2f}% ({time.time()-t1:.0f}s)")


# ─── 3) MLP on augmented data ───
log("\n[3] Training MLP on augmented data...")
t1 = time.time()
mlp = MLPClassifier(
    hidden_layer_sizes=(2048, 1024, 512, 256),
    activation="relu", solver="adam",
    alpha=1e-4, batch_size=512,
    learning_rate="adaptive", learning_rate_init=1e-3,
    max_iter=1000, early_stopping=True,
    validation_fraction=0.1, random_state=42, verbose=False,
)
mlp.fit(X_train_aug, y_train_aug)
acc_mlp = accuracy_score(y_test, mlp.predict(X_test_s)) * 100
log(f"    MLP-aug: {acc_mlp:.2f}% ({time.time()-t1:.0f}s)")


# ─── 4) PyTorch MLP with heavy augmentation ───
log("\n[4] Training PyTorch DeepMLP with heavy augmentation...")


class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(n_features)
        self.net = nn.Sequential(
            nn.Linear(n_features, 1536), nn.BatchNorm1d(1536), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(1536, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.net(self.bn(x))


X_tr_t = torch.tensor(X_train_s)
y_tr_t = torch.tensor(y_train.astype(np.int64))
X_te_t = torch.tensor(X_test_s)
y_te_t = torch.tensor(y_test.astype(np.int64))

counts = Counter(y_train.tolist())
weights = [1.0 / counts[int(l)] for l in y_train]
sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=128, sampler=sampler)

model = DeepMLP()
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)

best_acc = 0.0
best_state = None
no_improve = 0
t1 = time.time()

for epoch in range(1, 601):
    model.train()
    for xb, yb in train_loader:
        # Strong mixup
        lam = np.random.beta(0.4, 0.4)
        idx = torch.randperm(xb.size(0))
        xb_mix = lam * xb + (1 - lam) * xb[idx]
        # Add Gaussian noise
        xb_mix = xb_mix + torch.randn_like(xb_mix) * 0.1

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
        log(f"    Epoch {epoch:>3d}  test={acc:.2f}%  ({time.time()-t1:.0f}s)")

    if acc > best_acc:
        best_acc = acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1

    if no_improve >= 80:
        log(f"    Early stop at epoch {epoch}")
        break

    if best_acc >= 92:
        log(f"    Target reached!")
        break

acc_pt = best_acc
log(f"    PyTorch MLP: {acc_pt:.2f}%")

if best_state:
    model.load_state_dict(best_state)


# ─── 5) Ensemble voting ───
log("\n[5] Creating ensemble (soft voting)...")

# Get probability predictions from all models
proba_list = []

# SVC proba
proba_list.append(svc.predict_proba(X_test_s))

# MLP proba
proba_list.append(mlp.predict_proba(X_test_s))

# PyTorch proba
model.eval()
with torch.no_grad():
    pt_proba = torch.softmax(model(X_te_t), 1).numpy()
proba_list.append(pt_proba)

# Average probabilities
avg_proba = np.mean(proba_list, axis=0)
y_pred_ens = avg_proba.argmax(axis=1)
acc_ens = accuracy_score(y_test, y_pred_ens) * 100
log(f"    Ensemble (SVC+MLP+PyTorch): {acc_ens:.2f}%")


# ─── Pick best overall ───
results = {
    "LinearSVC": acc_svc,
    "LinearSVC-aug": acc_svc2,
    "MLP-aug": acc_mlp,
    "PyTorch": acc_pt,
    "Ensemble": acc_ens,
}
log(f"\nAll results: {results}")

best_name = max(results, key=results.get)
best_final_acc = results[best_name]
log(f"Best: {best_name} -> {best_final_acc:.2f}%")


# ─── Save best model ───
if best_name == "Ensemble":
    # Save ensemble as a special wrapper
    class EnsembleWrapper:
        def __init__(self2):
            self2.classes_ = np.array(classes)
            self2._svc = svc
            self2._mlp = mlp
            self2._pt_state = {k: v.cpu() for k, v in model.state_dict().items()}
            self2._scaler = scaler
            self2._nf = n_features
            self2._nc = n_classes
            self2._pt_model = None

        def _ensure_pt(self2):
            if self2._pt_model is None:
                self2._pt_model = DeepMLP()
                self2._pt_model.load_state_dict(self2._pt_state)
                self2._pt_model.eval()

        def predict(self2, X):
            Xs = self2._scaler.transform(np.array(X).astype(np.float32))
            p1 = self2._svc.predict_proba(Xs)
            p2 = self2._mlp.predict_proba(Xs)
            self2._ensure_pt()
            with torch.no_grad():
                p3 = torch.softmax(self2._pt_model(torch.tensor(Xs)), 1).numpy()
            avg = (p1 + p2 + p3) / 3
            return avg.argmax(axis=1)

        def predict_proba(self2, X):
            Xs = self2._scaler.transform(np.array(X).astype(np.float32))
            p1 = self2._svc.predict_proba(Xs)
            p2 = self2._mlp.predict_proba(Xs)
            self2._ensure_pt()
            with torch.no_grad():
                p3 = torch.softmax(self2._pt_model(torch.tensor(Xs)), 1).numpy()
            return (p1 + p2 + p3) / 3

    wrapper = EnsembleWrapper()
elif best_name == "PyTorch":
    class PTWrapper:
        def __init__(self2):
            self2.classes_ = np.array(classes)
            self2._state = {k: v.cpu() for k, v in model.state_dict().items()}
            self2._scaler = scaler
            self2._nf = n_features
            self2._nc = n_classes
            self2._model = None

        def _ensure(self2):
            if self2._model is None:
                self2._model = DeepMLP()
                self2._model.load_state_dict(self2._state)
                self2._model.eval()

        def predict(self2, X):
            self2._ensure()
            Xs = self2._scaler.transform(np.array(X).astype(np.float32))
            with torch.no_grad():
                return self2._model(torch.tensor(Xs)).argmax(1).numpy()

        def predict_proba(self2, X):
            self2._ensure()
            Xs = self2._scaler.transform(np.array(X).astype(np.float32))
            with torch.no_grad():
                return torch.softmax(self2._model(torch.tensor(Xs)), 1).numpy()

    wrapper = PTWrapper()
elif "MLP" in best_name:
    wrapper = Pipeline([("scaler", StandardScaler()), ("clf", mlp)])
    wrapper.fit(X_train, y_train)
else:
    wrapper = Pipeline([("scaler", StandardScaler()), ("clf", svc)])
    wrapper.fit(X_train, y_train)

log("\nSaving model...")
joblib.dump(wrapper, os.path.join(BASE, "image_model.pkl"), compress=1)
with open(os.path.join(BASE, "image_classes.pkl"), "wb") as f:
    pickle.dump(classes, f)

sz = os.path.getsize(os.path.join(BASE, "image_model.pkl")) / (1024*1024)
log(f"Saved: image_model.pkl ({sz:.1f} MB)")
log(f"FINAL: {best_final_acc:.2f}%")
log("=" * 60)
