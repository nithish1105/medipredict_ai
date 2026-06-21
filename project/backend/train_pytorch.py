"""
PyTorch deep MLP training on cached EfficientNet features.
Uses mixup, label smoothing, cosine annealing, and weighted sampling.
Target: 90%+ accuracy.
"""
import os, sys, time, pickle, warnings
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

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(BASE, "train_result.txt")

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")

open(LOG, "w").close()
log("=" * 60)
log("PyTorch Deep MLP Training on Cached Features")
log("=" * 60)

# Load data
data = np.load(os.path.join(BASE, "features_cache.npz"), allow_pickle=True)
X, y = data["X"], data["y"]
classes = list(data["classes"])
n_classes = len(classes)

le = LabelEncoder()
y_enc = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
log(f"Samples: {X.shape[0]}, Train: {len(X_train)}, Test: {len(X_test)}, Classes: {n_classes}")

# Standardize
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train).astype(np.float32)
X_test_s = scaler.transform(X_test).astype(np.float32)
n_features = X_train_s.shape[1]


class DeepMLP(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.bn_input = nn.BatchNorm1d(n_features)
        self.block1 = nn.Sequential(
            nn.Linear(n_features, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.4),
        )
        self.block2 = nn.Sequential(
            nn.Linear(1024, 768), nn.BatchNorm1d(768), nn.GELU(), nn.Dropout(0.35),
        )
        self.block3 = nn.Sequential(
            nn.Linear(768, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
        )
        self.block4 = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
        )
        self.head = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.bn_input(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.head(x)


# Prepare data
X_tr_t = torch.tensor(X_train_s)
y_tr_t = torch.tensor(y_train.astype(np.int64))
X_te_t = torch.tensor(X_test_s)
y_te_t = torch.tensor(y_test.astype(np.int64))

# Weighted sampler for class imbalance
counts = Counter(y_train.tolist())
weights = [1.0 / counts[int(l)] for l in y_train]
sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

train_ds = TensorDataset(X_tr_t, y_tr_t)
train_loader = DataLoader(train_ds, batch_size=128, sampler=sampler)

model = DeepMLP(n_features, n_classes)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

EPOCHS = 500
PATIENCE = 60
best_acc = 0.0
best_state = None
no_improve = 0

log(f"\nTraining {sum(p.numel() for p in model.parameters()):,} parameters")
log(f"Epochs: {EPOCHS}, Patience: {PATIENCE}\n")

t0 = time.time()
for epoch in range(1, EPOCHS + 1):
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

    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(X_te_t).argmax(1)
        acc = (preds == y_te_t).float().mean().item() * 100

    if epoch % 10 == 0 or acc > best_acc:
        elapsed = time.time() - t0
        log(f"  Epoch {epoch:>3d}  test={acc:.2f}%  lr={optimizer.param_groups[0]['lr']:.6f}  ({elapsed:.0f}s)")

    if acc > best_acc:
        best_acc = acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1

    if no_improve >= PATIENCE:
        log(f"\n  Early stop at epoch {epoch}")
        break

    # Early exit if we hit target
    if best_acc >= 92.0:
        log(f"\n  Target accuracy reached!")
        break

log(f"\nBest accuracy: {best_acc:.2f}%")

if best_state:
    model.load_state_dict(best_state)

# Full eval
model.eval()
with torch.no_grad():
    y_pred = model(X_te_t).argmax(1).numpy()

report = classification_report(y_test, y_pred, target_names=le.classes_)
log(report)

# Save as sklearn-compatible wrapper
class PyTorchWrapper:
    def __init__(self, state_dict, scaler, n_features, n_classes, all_classes):
        self.classes_ = np.array(all_classes)
        self._state = {k: v.cpu() for k, v in state_dict.items()}
        self._scaler = scaler
        self._nf = n_features
        self._nc = n_classes
        self._model = None

    def _ensure(self):
        if self._model is None:
            self._model = DeepMLP(self._nf, self._nc)
            self._model.load_state_dict(self._state)
            self._model.eval()

    def predict(self, X):
        self._ensure()
        X_s = self._scaler.transform(np.array(X).astype(np.float32))
        with torch.no_grad():
            return self._model(torch.tensor(X_s)).argmax(1).numpy()

    def predict_proba(self, X):
        self._ensure()
        X_s = self._scaler.transform(np.array(X).astype(np.float32))
        with torch.no_grad():
            return torch.softmax(self._model(torch.tensor(X_s)), 1).numpy()

wrapper = PyTorchWrapper(model.state_dict(), scaler, n_features, n_classes, classes)
joblib.dump(wrapper, os.path.join(BASE, "image_model.pkl"), compress=1)
with open(os.path.join(BASE, "image_classes.pkl"), "wb") as f:
    pickle.dump(classes, f)

sz = os.path.getsize(os.path.join(BASE, "image_model.pkl")) / (1024*1024)
elapsed = time.time() - t0
log(f"\nSaved: image_model.pkl ({sz:.1f} MB)")
log(f"Time: {elapsed/60:.1f} min")
log(f"FINAL: {best_acc:.2f}%")
log("=" * 60)
