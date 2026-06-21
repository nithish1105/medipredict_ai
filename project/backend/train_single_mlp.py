"""
train_single_mlp.py - Train ONE MLP with original 75% approach
Usage: python train_single_mlp.py <config_index>  (0-4)
"""
import os, sys, time, gc, warnings, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))

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
    def forward(self, x): return self.net(self.bn(x))

CONFIGS = [
    (42, 1024, 1e-3),
    (123, 1280, 8e-4),
    (7, 768, 1.2e-3),
    (999, 1024, 5e-4),
    (2024, 1536, 7e-4),
]

def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    outfile = os.path.join(BASE, f"ckpt_{idx}.pt")
    lockfile = os.path.join(BASE, f"ckpt_{idx}.lock")
    
    if os.path.exists(outfile):
        print(f"Model {idx} already exists -> {outfile}", flush=True)
        return

    if os.path.exists(lockfile):
        try:
            with open(lockfile, "r", encoding="utf-8") as f:
                lock_txt = f.read().strip()
        except Exception:
            lock_txt = "<unreadable>"
        print(f"Lock exists for model {idx} ({lock_txt}); refusing to start a duplicate run.", flush=True)
        return

    with open(lockfile, "w", encoding="utf-8") as f:
        f.write(f"pid={os.getpid()} time={time.time()}\n")
    
    try:
        seed, width, lr = CONFIGS[idx]
        print(f"Training model {idx}: seed={seed} width={width} lr={lr}", flush=True)

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

        print(f"  Train: {len(X_train)}, Test: {len(X_test)}", flush=True)

        # Train
        torch.manual_seed(seed)
        np.random.seed(seed)
        X_tr_t = torch.tensor(X_train_s)
        y_tr_t = torch.tensor(y_train.astype(np.int64))

        counts = Counter(y_train.tolist())
        weights = [1.0 / counts[int(l)] for l in y_train]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=128, sampler=sampler)

        model = DeepMLP(n_features, n_classes, width=width)
        nparams = sum(p.numel() for p in model.parameters())
        print(f"  Params: {nparams:,}", flush=True)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)

        best_acc = 0.0
        best_state = None
        no_improve = 0
        t0 = time.time()

        for epoch in range(1, 501):
            model.train()
            for xb, yb in loader:
                lam = np.random.beta(0.3, 0.3)
                idx2 = torch.randperm(xb.size(0))
                xb_mix = lam * xb + (1 - lam) * xb[idx2]
                xb_mix = xb_mix + torch.randn_like(xb_mix) * 0.05
                optimizer.zero_grad()
                out = model(xb_mix)
                loss = lam * criterion(out, yb) + (1 - lam) * criterion(out, yb[idx2])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                acc = (model(X_te_t).argmax(1) == y_te_t).float().mean().item() * 100

            if epoch % 50 == 0:
                print(f"    ep {epoch}: {acc:.1f}% best={best_acc:.1f}%", flush=True)

            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= 60:
                print(f"    Early stop at epoch {epoch}", flush=True)
                break

        elapsed = time.time() - t0
        print(f"  Best accuracy: {best_acc:.2f}%  Time: {elapsed:.0f}s", flush=True)

        # Save checkpoint
        torch.save({
            "state_dict": best_state,
            "width": width,
            "seed": seed,
            "accuracy": best_acc,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "classes": classes,
            "n_features": n_features,
            "n_classes": n_classes,
        }, outfile)
        print(f"  Saved -> {outfile}", flush=True)
        print("DONE!", flush=True)
    finally:
        try:
            if os.path.exists(lockfile):
                os.remove(lockfile)
        except Exception:
            pass

if __name__ == "__main__":
    main()
