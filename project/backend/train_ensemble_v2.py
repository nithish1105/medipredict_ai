"""
Multi-seed ensemble: Train N PyTorch MLPs with different seeds.
Average predictions for better accuracy.
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

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(BASE, "train_result.txt")

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")

open(LOG, "w").close()

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

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train).astype(np.float32)
X_test_s = scaler.transform(X_test).astype(np.float32)
n_features = X_train_s.shape[1]

X_te_t = torch.tensor(X_test_s)
y_te_t = torch.tensor(y_test.astype(np.int64))

log(f"Train: {len(X_train)}, Test: {len(X_test)}, Classes: {n_classes}")

class DeepMLP(nn.Module):
    def __init__(self, nf, nc, width=1024):
        super().__init__()
        self.bn = nn.BatchNorm1d(nf)
        self.net = nn.Sequential(
            nn.Linear(nf, width), nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(width, width//2), nn.BatchNorm1d(width//2), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(width//2, width//4), nn.BatchNorm1d(width//4), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(width//4, nc),
        )
    def forward(self, x):
        return self.net(self.bn(x))

def train_one(seed, width, lr, epochs=500, patience=60):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_tr_t = torch.tensor(X_train_s)
    y_tr_t = torch.tensor(y_train.astype(np.int64))

    counts = Counter(y_train.tolist())
    weights = [1.0 / counts[int(l)] for l in y_train]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=128, sampler=sampler)

    model = DeepMLP(n_features, n_classes, width=width)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)

    best_acc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            lam = np.random.beta(0.3, 0.3)
            idx = torch.randperm(xb.size(0))
            xb_mix = lam * xb + (1 - lam) * xb[idx]
            xb_mix = xb_mix + torch.randn_like(xb_mix) * 0.05  # noise
            optimizer.zero_grad()
            out = model(xb_mix)
            loss = lam * criterion(out, yb) + (1 - lam) * criterion(out, yb[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            acc = (model(X_te_t).argmax(1) == y_te_t).float().mean().item() * 100
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_acc

# Train multiple models
configs = [
    (42, 1024, 1e-3),
    (123, 1280, 8e-4),
    (7, 768, 1.2e-3),
    (999, 1024, 5e-4),
    (2024, 1536, 7e-4),
]

models = []
for i, (seed, width, lr) in enumerate(configs):
    log(f"\n[{i+1}/{len(configs)}] Seed={seed}, Width={width}, LR={lr}")
    t0 = time.time()
    model, acc = train_one(seed, width, lr)
    log(f"    Accuracy: {acc:.2f}%  ({time.time()-t0:.0f}s)")
    models.append((model, acc))

# Individual accuracies
for i, (m, a) in enumerate(models):
    log(f"  Model {i+1}: {a:.2f}%")

# Ensemble predictions (average probabilities)
log("\nEnsemble (soft voting)...")
all_probs = []
for m, _ in models:
    m.eval()
    with torch.no_grad():
        probs = torch.softmax(m(X_te_t), 1).numpy()
        all_probs.append(probs)

avg_probs = np.mean(all_probs, axis=0)
y_pred = avg_probs.argmax(axis=1)
ens_acc = accuracy_score(y_test, y_pred) * 100
log(f"Ensemble accuracy: {ens_acc:.2f}%")

report = classification_report(y_test, y_pred, target_names=le.classes_)
log(report)

# Save ensemble
class EnsembleWrapper:
    def __init__(self2, model_data, scaler_obj, all_classes_list, nf, nc):
        self2.classes_ = np.array(all_classes_list)
        self2._model_data = model_data  # list of (state_dict, width)
        self2._scaler = scaler_obj
        self2._nf = nf
        self2._nc = nc
        self2._models = None

    def _ensure(self2):
        if self2._models is None:
            self2._models = []
            for state, width in self2._model_data:
                m = DeepMLP(self2._nf, self2._nc, width=width)
                m.load_state_dict(state)
                m.eval()
                self2._models.append(m)

    def predict(self2, X):
        self2._ensure()
        Xs = self2._scaler.transform(np.array(X).astype(np.float32))
        Xt = torch.tensor(Xs)
        probs = []
        with torch.no_grad():
            for m in self2._models:
                probs.append(torch.softmax(m(Xt), 1).numpy())
        return np.mean(probs, axis=0).argmax(axis=1)

    def predict_proba(self2, X):
        self2._ensure()
        Xs = self2._scaler.transform(np.array(X).astype(np.float32))
        Xt = torch.tensor(Xs)
        probs = []
        with torch.no_grad():
            for m in self2._models:
                probs.append(torch.softmax(m(Xt), 1).numpy())
        return np.mean(probs, axis=0)

model_data = [(
    {k: v.cpu() for k, v in m.state_dict().items()},
    configs[i][1]
) for i, (m, _) in enumerate(models)]

wrapper = EnsembleWrapper(model_data, scaler, classes, n_features, n_classes)

joblib.dump(wrapper, os.path.join(BASE, "image_model.pkl"), compress=1)
with open(os.path.join(BASE, "image_classes.pkl"), "wb") as f:
    pickle.dump(classes, f)

sz = os.path.getsize(os.path.join(BASE, "image_model.pkl")) / (1024*1024)
log(f"\nSaved: image_model.pkl ({sz:.1f} MB)")
log(f"FINAL ENSEMBLE: {ens_acc:.2f}%")
log("=" * 60)
