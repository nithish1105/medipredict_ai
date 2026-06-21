"""
Fast Accurate Image Detection Training
======================================
Optimized training for quick results with high accuracy.
"""

import os
import sys
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AccurateClassifier(nn.Module):
    """Deep MLP for accurate classification."""
    def __init__(self, n_features=1280, n_classes=28):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(n_features),
            
            nn.Linear(n_features, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.4),
            
            nn.Linear(1024, 768),
            nn.BatchNorm1d(768),
            nn.GELU(),
            nn.Dropout(0.35),
            
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, n_classes),
        )
        
    def forward(self, x):
        return self.net(x)


def mixup_data(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0))
    return lam * x + (1 - lam) * x[index], y, y[index], lam


def train():
    print("=" * 60)
    print("  FAST ACCURATE IMAGE DETECTION TRAINING")
    print("=" * 60)
    
    # Load data
    cache_path = os.path.join(BASE_DIR, "features_cache.npz")
    if not os.path.exists(cache_path):
        print("ERROR: features_cache.npz not found!")
        sys.exit(1)
    
    data = np.load(cache_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    all_classes = list(data["classes"])
    n_classes = len(all_classes)
    
    print(f"\nDataset: {X.shape[0]} samples, {n_classes} classes")
    
    # Encode labels
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    y_enc = np.array([class_to_idx[label] for label in y])
    
    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")
    
    # Datasets
    train_ds = FeatureDataset(X_train, y_train)
    test_ds = FeatureDataset(X_test, y_test)
    
    # Weighted sampler
    class_counts = Counter(y_train.tolist())
    weights = [1.0 / class_counts[int(l)] for l in y_train]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=128, sampler=sampler)
    test_loader = DataLoader(test_ds, batch_size=512)
    
    # Model
    model = AccurateClassifier(n_features=X.shape[1], n_classes=n_classes)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    
    print(f"Training {sum(p.numel() for p in model.parameters()):,} parameters...")
    print("-" * 60)
    
    best_acc = 0.0
    best_state = None
    patience = 0
    max_patience = 30
    
    for epoch in range(1, 151):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for xb, yb in train_loader:
            xb_mix, ya, yb_m, lam = mixup_data(xb, yb)
            optimizer.zero_grad()
            out = model(xb_mix)
            loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb_m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * xb.size(0)
            _, pred = out.max(1)
            correct += (pred == ya).sum().item()
            total += xb.size(0)
        
        scheduler.step()
        
        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                out = model(xb)
                _, pred = out.max(1)
                test_correct += (pred == yb).sum().item()
                test_total += yb.size(0)
        
        test_acc = test_correct / test_total * 100
        
        if epoch % 10 == 0 or test_acc > best_acc:
            print(f"Epoch {epoch:3d}: loss={total_loss/total:.4f} "
                  f"train={correct/total*100:.1f}% test={test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        
        if patience >= max_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    model.load_state_dict(best_state)
    print(f"\n{'=' * 60}")
    print(f"  BEST TEST ACCURACY: {best_acc:.2f}%")
    print(f"{'=' * 60}")
    
    # Save model
    print("\nSaving model...")
    
    # PyTorch format
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': all_classes,
        'n_features': X.shape[1],
        'n_classes': n_classes,
        'accuracy': best_acc,
    }, os.path.join(BASE_DIR, "image_classifier.pt"))
    
    # Sklearn-compatible wrapper
    wrapper = ModelWrapper(model, all_classes, X.shape[1])
    joblib.dump(wrapper, os.path.join(BASE_DIR, "image_model.pkl"), compress=3)
    
    # Classes
    with open(os.path.join(BASE_DIR, "image_classes.pkl"), "wb") as f:
        pickle.dump(all_classes, f)
    
    print(f"Saved: image_classifier.pt, image_model.pkl, image_classes.pkl")
    
    return model, all_classes, best_acc


class ModelWrapper:
    """Sklearn-compatible wrapper."""
    def __init__(self, model, classes, n_features):
        self.classes_ = np.array(classes)
        self._n_features = n_features
        self._n_classes = len(classes)
        self._state = {k: v.cpu() for k, v in model.state_dict().items()}
        self._model = None
    
    def _ensure_model(self):
        if self._model is None:
            self._model = AccurateClassifier(self._n_features, self._n_classes)
            self._model.load_state_dict(self._state)
            self._model.eval()
    
    def predict(self, X):
        self._ensure_model()
        X_t = torch.tensor(np.array(X), dtype=torch.float32)
        with torch.no_grad():
            return self._model(X_t).argmax(dim=1).numpy()
    
    def predict_proba(self, X):
        self._ensure_model()
        X_t = torch.tensor(np.array(X), dtype=torch.float32)
        with torch.no_grad():
            return torch.softmax(self._model(X_t), dim=1).numpy()


if __name__ == "__main__":
    t0 = time.time()
    model, classes, acc = train()
    print(f"\nTotal time: {time.time()-t0:.0f}s")
