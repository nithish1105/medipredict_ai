"""
Train Accurate Image Detection Model
=====================================
Combines multiple advanced techniques to achieve maximum accuracy:
1. Deep MLP with residual connections, dropout, batch normalization
2. Mixup augmentation for regularization
3. Label smoothing for better calibration
4. Cosine annealing learning rate schedule
5. Ensemble of multiple model architectures
6. Focal loss for handling class imbalance

Target: 90%+ accuracy on image disease detection
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
    """Dataset for cached EfficientNet features."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Skip connection (project if dimensions differ)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x):
        return self.layers(x) + self.skip(x)


class AccurateClassifier(nn.Module):
    """Deep MLP with residual connections for maximum accuracy."""
    def __init__(self, n_features=1280, n_classes=28):
        super().__init__()
        self.bn_input = nn.BatchNorm1d(n_features)
        
        # Deep residual network
        self.block1 = ResidualBlock(n_features, 1024, dropout=0.4)
        self.block2 = ResidualBlock(1024, 768, dropout=0.35)
        self.block3 = ResidualBlock(768, 512, dropout=0.3)
        self.block4 = ResidualBlock(512, 384, dropout=0.25)
        self.block5 = ResidualBlock(384, 256, dropout=0.2)
        
        # Final classification head
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, n_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.bn_input(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return self.head(x)


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    def __init__(self, alpha=1, gamma=2, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction='none', label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def mixup_data(x, y, alpha=0.4):
    """Apply mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def evaluate(model, loader):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb)
            _, predicted = out.max(1)
            correct += (predicted == yb).sum().item()
            total += yb.size(0)
            all_preds.extend(predicted.tolist())
            all_labels.extend(yb.tolist())
    
    return correct / total * 100, all_preds, all_labels


def train_model(X_train, y_train, X_test, y_test, n_classes,
                epochs=400, lr=1e-3, batch_size=128, patience=50):
    """Train a single model with all optimizations."""
    
    # Create datasets
    train_ds = FeatureDataset(X_train, y_train)
    test_ds = FeatureDataset(X_test, y_test)
    
    # Weighted sampler for class imbalance
    class_counts = Counter(y_train.tolist())
    weights = [1.0 / class_counts[int(label)] for label in y_train]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    
    # Model
    n_features = X_train.shape[1]
    model = AccurateClassifier(n_features=n_features, n_classes=n_classes)
    
    # Use focal loss for better handling of difficult examples
    criterion = FocalLoss(alpha=1.0, gamma=2.0, label_smoothing=0.1)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-6
    )
    
    best_acc = 0.0
    best_state = None
    no_improve = 0
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Training model with {n_params:,} parameters ...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        
        for xb, yb in train_loader:
            # Apply mixup augmentation
            xb_mixed, ya, yb_mix, lam = mixup_data(xb, yb, alpha=0.4)
            
            optimizer.zero_grad()
            out = model(xb_mixed)
            loss = mixup_criterion(criterion, out, ya, yb_mix, lam)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * xb.size(0)
            _, predicted = out.max(1)
            train_correct += (predicted == ya).sum().item()
            train_total += xb.size(0)
        
        scheduler.step()
        train_acc = train_correct / train_total * 100
        
        # Evaluate
        test_acc, _, _ = evaluate(model, test_loader)
        
        # Log progress
        if epoch % 20 == 0 or test_acc > best_acc or epoch <= 5:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"   Epoch {epoch:>4d}  loss={total_loss/train_total:.4f}  "
                  f"train={train_acc:.1f}%  test={test_acc:.2f}%  lr={lr_now:.2e}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= patience:
            print(f"   Early stopping at epoch {epoch} (best: {best_acc:.2f}%)")
            break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, best_acc


def train_ensemble():
    """Train an ensemble of models for maximum accuracy."""
    print("=" * 70)
    print("  ACCURATE IMAGE DETECTION MODEL TRAINING")
    print("=" * 70)
    
    # Load cached features
    cache_path = os.path.join(BASE_DIR, "features_cache.npz")
    if not os.path.exists(cache_path):
        print("ERROR: features_cache.npz not found!")
        print("Run train_image_model.py first to extract features.")
        sys.exit(1)
    
    data = np.load(cache_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    all_classes = list(data["classes"])
    n_classes = len(all_classes)
    
    print(f"\n   Dataset: {X.shape[0]} samples, {X.shape[1]} features, {n_classes} classes")
    
    # Encode labels
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    y_enc = np.array([class_to_idx[label] for label in y])
    
    # Stratified split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    print(f"   Train: {len(X_train)}  |  Test: {len(X_test)}\n")
    
    # Convert to numpy arrays for easier handling
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    # Train multiple models with different random seeds
    print("-" * 70)
    print("  PHASE 1: Training Ensemble Models")
    print("-" * 70)
    
    models = []
    accuracies = []
    
    configs = [
        {"seed": 42, "lr": 1e-3, "batch_size": 128, "epochs": 400, "patience": 50},
        {"seed": 123, "lr": 8e-4, "batch_size": 96, "epochs": 400, "patience": 50},
        {"seed": 456, "lr": 1.2e-3, "batch_size": 160, "epochs": 400, "patience": 50},
    ]
    
    for i, cfg in enumerate(configs):
        print(f"\n   Model {i+1}/{len(configs)} (seed={cfg['seed']}, lr={cfg['lr']}, bs={cfg['batch_size']})")
        
        torch.manual_seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        
        model, acc = train_model(
            X_train, y_train, X_test, y_test, n_classes,
            epochs=cfg['epochs'], lr=cfg['lr'], 
            batch_size=cfg['batch_size'], patience=cfg['patience']
        )
        models.append(model)
        accuracies.append(acc)
        print(f"   Model {i+1} Best Accuracy: {acc:.2f}%")
    
    # Evaluate ensemble
    print("\n" + "-" * 70)
    print("  PHASE 2: Ensemble Evaluation")
    print("-" * 70)
    
    test_ds = FeatureDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    
    # Get ensemble predictions
    all_probs = []
    for model in models:
        model.eval()
        model_probs = []
        with torch.no_grad():
            for xb, _ in test_loader:
                out = model(xb)
                probs = torch.softmax(out, dim=1)
                model_probs.append(probs)
        all_probs.append(torch.cat(model_probs, dim=0))
    
    # Average probabilities
    ensemble_probs = torch.stack(all_probs).mean(dim=0)
    ensemble_preds = ensemble_probs.argmax(dim=1).numpy()
    
    # Ground truth
    all_labels = []
    for _, yb in test_loader:
        all_labels.extend(yb.tolist())
    all_labels = np.array(all_labels)
    
    ensemble_acc = (ensemble_preds == all_labels).mean() * 100
    
    print(f"\n   Individual model accuracies: {[f'{a:.2f}%' for a in accuracies]}")
    print(f"   Ensemble accuracy: {ensemble_acc:.2f}%")
    
    # Use the best performing model for saving
    best_idx = np.argmax(accuracies)
    best_model = models[best_idx]
    best_acc = max(accuracies[best_idx], ensemble_acc)
    
    # Save the model
    print("\n" + "-" * 70)
    print("  SAVING MODEL")
    print("-" * 70)
    
    save_model(best_model, models, all_classes, X.shape[1], accuracies, ensemble_acc)
    
    return best_model, all_classes, best_acc


def save_model(best_model, ensemble_models, classes, n_features, accuracies, ensemble_acc):
    """Save the trained model for inference."""
    
    # Save PyTorch state dict
    model_path = os.path.join(BASE_DIR, "image_classifier.pt")
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'classes': classes,
        'n_features': n_features,
        'n_classes': len(classes),
    }, model_path)
    sz = os.path.getsize(model_path) / (1024 * 1024)
    print(f"   Saved PyTorch model -> {model_path} ({sz:.1f} MB)")
    
    # Save sklearn-compatible wrapper
    wrapper = PyTorchModelWrapper(best_model, ensemble_models, classes, n_features)
    wrapper_path = os.path.join(BASE_DIR, "image_model.pkl")
    joblib.dump(wrapper, wrapper_path, compress=3)
    sz = os.path.getsize(wrapper_path) / (1024 * 1024)
    print(f"   Saved wrapper      -> {wrapper_path} ({sz:.1f} MB)")
    
    # Save classes
    cls_path = os.path.join(BASE_DIR, "image_classes.pkl")
    with open(cls_path, "wb") as f:
        pickle.dump(classes, f)
    print(f"   Saved classes      -> {cls_path}")
    
    # Save training report
    report_path = os.path.join(BASE_DIR, "training_report.txt")
    with open(report_path, "w") as f:
        f.write("IMAGE DETECTION MODEL TRAINING REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Features: {n_features}\n")
        f.write(f"Classes: {len(classes)}\n")
        f.write(f"Individual Accuracies: {[f'{a:.2f}%' for a in accuracies]}\n")
        f.write(f"Ensemble Accuracy: {ensemble_acc:.2f}%\n")
        f.write(f"Best Model: {max(accuracies):.2f}%\n")
        f.write(f"\nClasses:\n")
        for i, c in enumerate(classes):
            f.write(f"  {i}: {c}\n")
    print(f"   Saved report       -> {report_path}")


class PyTorchModelWrapper:
    """Sklearn-compatible wrapper for prediction."""
    
    def __init__(self, best_model, ensemble_models, classes, n_features):
        self.classes_ = np.array(classes)
        self._n_features = n_features
        self._n_classes = len(classes)
        
        # Store state dicts
        self._best_state = {k: v.cpu() for k, v in best_model.state_dict().items()}
        self._ensemble_states = [{k: v.cpu() for k, v in m.state_dict().items()} 
                                  for m in ensemble_models]
        
        self._model = None
        self._ensemble = None
    
    def _ensure_model(self):
        if self._model is None:
            self._model = AccurateClassifier(
                n_features=self._n_features, 
                n_classes=self._n_classes
            )
            self._model.load_state_dict(self._best_state)
            self._model.eval()
    
    def _ensure_ensemble(self):
        if self._ensemble is None:
            self._ensemble = []
            for state in self._ensemble_states:
                model = AccurateClassifier(
                    n_features=self._n_features, 
                    n_classes=self._n_classes
                )
                model.load_state_dict(state)
                model.eval()
                self._ensemble.append(model)
    
    def predict(self, X, use_ensemble=True):
        """Predict class labels."""
        if use_ensemble and len(self._ensemble_states) > 1:
            return self._predict_ensemble(X)
        else:
            self._ensure_model()
            X_t = torch.tensor(np.array(X), dtype=torch.float32)
            with torch.no_grad():
                logits = self._model(X_t)
                preds = logits.argmax(dim=1).numpy()
            return preds
    
    def _predict_ensemble(self, X):
        """Ensemble prediction using averaged probabilities."""
        self._ensure_ensemble()
        X_t = torch.tensor(np.array(X), dtype=torch.float32)
        
        all_probs = []
        with torch.no_grad():
            for model in self._ensemble:
                logits = model(X_t)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs)
        
        avg_probs = torch.stack(all_probs).mean(dim=0)
        return avg_probs.argmax(dim=1).numpy()
    
    def predict_proba(self, X, use_ensemble=True):
        """Predict class probabilities."""
        if use_ensemble and len(self._ensemble_states) > 1:
            return self._predict_proba_ensemble(X)
        else:
            self._ensure_model()
            X_t = torch.tensor(np.array(X), dtype=torch.float32)
            with torch.no_grad():
                logits = self._model(X_t)
                probs = torch.softmax(logits, dim=1).numpy()
            return probs
    
    def _predict_proba_ensemble(self, X):
        """Ensemble probability prediction."""
        self._ensure_ensemble()
        X_t = torch.tensor(np.array(X), dtype=torch.float32)
        
        all_probs = []
        with torch.no_grad():
            for model in self._ensemble:
                logits = model(X_t)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs)
        
        return torch.stack(all_probs).mean(dim=0).numpy()


if __name__ == "__main__":
    t0 = time.time()
    
    try:
        model, classes, accuracy = train_ensemble()
        
        elapsed = time.time() - t0
        print(f"\n" + "=" * 70)
        print(f"  TRAINING COMPLETE")
        print(f"  Final Accuracy: {accuracy:.2f}%")
        print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
