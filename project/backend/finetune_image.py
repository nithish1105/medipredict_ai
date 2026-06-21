"""
Fine-tune image classifier - Two approaches
=============================================
Stage 1: Train a deep MLP on cached EfficientNet features (fast, minutes)
Stage 2: Fine-tune EfficientNet end-to-end from ZIP archives (slow, hours)

Usage:
    python finetune_image.py                      # Stage 1 only (use cached features)
    python finetune_image.py --full-finetune      # Stage 2: end-to-end fine-tuning
"""

import os
import sys
import time
import pickle
import argparse
import zipfile
import io
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(os.path.dirname(BASE_DIR), "datasets")

# ── Archive config (same as train_image_model.py) ────────────────────
CLASS_MAP_1 = {}  # archive 1: use folder names as-is
CLASS_MAP_3 = {   # HAM10000
    "akiec": "Actinic Keratosis Basal Cell Carcinoma and other Malign",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma Skin Cancer Nevi and Moles",
    "bkl":   "Seborrheic Keratoses and other Benign Tumors",
    "vasc":  "Vascular Tumors",
}
CLASS_MAP_4 = {   # IMG_CLASSES
    "Actinic Keratosis":            "Actinic Keratosis Basal Cell Carcinoma and other Malign",
    "Atopic Dermatitis":            "Atopic Dermatitis Photos",
    "Eczema":                       "Eczema Photos",
    "Melanoma":                     "Melanoma Skin Cancer Nevi and Moles",
    "Psoriasis":                    "Psoriasis pictures Lichen Planus and related diseases",
    "Seborrheic Keratoses":         "Seborrheic Keratoses and other Benign Tumors",
    "Tinea Ringworm Candidiasis":   "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Warts Molluscum":              "Warts Molluscum and other Viral Infections",
}
ARCHIVE_CFGS = [
    ("archive (1).zip", "train", CLASS_MAP_1, "Skin"),
    ("archive (2).zip", "",      {},           "Eye"),
    ("archive (3).zip", "",      CLASS_MAP_3,  "HAM10000"),
    ("archive (4).zip", "train", CLASS_MAP_4,  "IMG_CLASSES"),
]


# ══════════════════════════════════════════════════════════════════════
#  Stage 1: Deep MLP on cached features
# ══════════════════════════════════════════════════════════════════════

class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DeepClassifier(nn.Module):
    """MLP with BatchNorm, Dropout, and residual-like skip connections."""
    def __init__(self, n_features=1280, n_classes=28):
        super().__init__()
        self.bn_input = nn.BatchNorm1d(n_features)
        
        self.block1 = nn.Sequential(
            nn.Linear(n_features, 768),
            nn.BatchNorm1d(768),
            nn.GELU(),
            nn.Dropout(0.4),
        )
        self.block2 = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.block3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.head = nn.Linear(256, n_classes)
        
    def forward(self, x):
        x = self.bn_input(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x)


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation for tabular data."""
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
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_stage1(epochs=200, lr=1e-3, batch_size=128, patience=30):
    """Train classifier MLP on cached EfficientNet features."""
    print("=" * 65)
    print("  Stage 1: Deep MLP on Cached Features")
    print("=" * 65)
    
    cache_path = os.path.join(BASE_DIR, "features_cache.npz")
    if not os.path.exists(cache_path):
        print(f"ERROR: {cache_path} not found. Run train_image_model.py first.")
        return None
    
    data = np.load(cache_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    all_classes = list(data["classes"])
    n_classes = len(all_classes)
    print(f"   Loaded {X.shape[0]} samples, {X.shape[1]} features, {n_classes} classes\n")
    
    # Encode labels
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    y_enc = np.array([class_to_idx[label] for label in y])
    
    # Stratified split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    print(f"   Train: {len(X_train)}  |  Test: {len(X_test)}")
    
    # Datasets & weighted sampler for class imbalance
    train_ds = FeatureDataset(X_train, y_train)
    test_ds = FeatureDataset(X_test, y_test)
    
    class_counts = Counter(y_train)
    weights = [1.0 / class_counts[label] for label in y_train]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                             num_workers=0, pin_memory=False)
    
    # Model
    model = DeepClassifier(n_features=X.shape[1], n_classes=n_classes)
    
    # Label smoothing loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    best_acc = 0.0
    best_state = None
    no_improve = 0
    
    print(f"\n   Training {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"   Epochs: {epochs}, LR: {lr}, Batch: {batch_size}\n")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for xb, yb in train_loader:
            # Mixup augmentation
            xb_mixed, ya, yb_mix, lam = mixup_data(xb, yb, alpha=0.3)
            
            optimizer.zero_grad()
            out = model(xb_mixed)
            loss = mixup_criterion(criterion, out, ya, yb_mix, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * xb.size(0)
            _, predicted = out.max(1)
            correct += (predicted == ya).sum().item()
            total += xb.size(0)
        
        scheduler.step()
        train_acc = correct / total * 100
        
        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                out = model(xb)
                _, predicted = out.max(1)
                test_correct += (predicted == yb).sum().item()
                test_total += xb.size(0)
        test_acc = test_correct / test_total * 100
        
        if epoch % 10 == 0 or test_acc > best_acc:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"   Epoch {epoch:>3d}  loss={total_loss/total:.4f}  "
                  f"train={train_acc:.1f}%  test={test_acc:.1f}%  lr={lr_now:.6f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= patience:
            print(f"\n   Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break
    
    print(f"\n   Best Test Accuracy: {best_acc:.2f}%")
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, all_classes, best_acc, X.shape[1]


# ══════════════════════════════════════════════════════════════════════
#  Stage 2: End-to-end fine-tuning
# ══════════════════════════════════════════════════════════════════════

class ZipImageDataset(Dataset):
    """PyTorch Dataset that reads images from ZIP archives."""
    
    def __init__(self, samples, transform=None):
        """samples: list of (zip_path, img_path_in_zip, label_idx)"""
        self.samples = samples
        self.transform = transform
        self._zip_handles = {}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        zip_path, img_path, label = self.samples[idx]
        
        try:
            if zip_path not in self._zip_handles:
                self._zip_handles[zip_path] = zipfile.ZipFile(zip_path, 'r')
            zf = self._zip_handles[zip_path]
            
            img_data = zf.read(img_path)
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            
            if self.transform:
                img = self.transform(img)
            
            return img, label
        except Exception:
            # Return a blank image on error
            img = Image.new("RGB", (224, 224), (128, 128, 128))
            if self.transform:
                img = self.transform(img)
            return img, label


def collect_zip_samples(max_per_class=300):
    """Collect image paths from ZIP archives, returning (zip_path, img_path, class_name)."""
    samples = []  # (zip_path, img_path, class_name)
    
    for zip_name, train_folder, class_map, label in ARCHIVE_CFGS:
        zip_path = os.path.join(DATASET_DIR, zip_name)
        if not os.path.exists(zip_path):
            print(f"   SKIP {zip_name} (not found)")
            continue
        
        print(f"\n   Scanning {label} ({zip_name}) ...")
        zf = zipfile.ZipFile(zip_path, 'r')
        names = zf.namelist()
        
        # Group by class folder
        class_images = {}
        for name in names:
            if name.endswith('/') or not name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            parts = name.replace('\\', '/').split('/')
            if len(parts) < 2:
                continue
            
            if train_folder:
                # e.g., "train/ClassName/img.jpg"
                try:
                    train_idx = parts.index(train_folder)
                    if train_idx + 1 < len(parts) - 1:
                        raw_class = parts[train_idx + 1]
                    else:
                        continue
                except ValueError:
                    continue
            else:
                raw_class = parts[-2]
            
            mapped = class_map.get(raw_class, raw_class) if class_map else raw_class
            if mapped not in class_images:
                class_images[mapped] = []
            class_images[mapped].append((zip_path, name))
        
        zf.close()
        
        for cls_name, imgs in sorted(class_images.items()):
            selected = random.sample(imgs, min(len(imgs), max_per_class))
            for zp, ip in selected:
                samples.append((zp, ip, cls_name))
            print(f"      {cls_name[:50]:<50} {len(selected):>4}")
    
    return samples


class FineTuneModel(nn.Module):
    """EfficientNet-B0 with custom classifier head for fine-tuning."""
    
    def __init__(self, n_classes, freeze_backbone=True):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze last 2 blocks of features
            for param in self.backbone.features[-2:].parameters():
                param.requires_grad = True
        
        # Replace classifier
        in_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, n_classes),
        )
    
    def forward(self, x):
        return self.backbone(x)


def train_stage2(epochs=15, lr=1e-4, batch_size=32, max_per_class=300):
    """Full end-to-end fine-tuning of EfficientNet-B0."""
    print("=" * 65)
    print("  Stage 2: End-to-end EfficientNet-B0 Fine-tuning")
    print("=" * 65)
    
    # Collect samples
    raw_samples = collect_zip_samples(max_per_class=max_per_class)
    all_classes = sorted(set(s[2] for s in raw_samples))
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    n_classes = len(all_classes)
    
    print(f"\n   Total samples: {len(raw_samples)}, Classes: {n_classes}")
    
    # Convert to (zip_path, img_path, label_idx)
    indexed_samples = [(zp, ip, class_to_idx[cn]) for zp, ip, cn in raw_samples]
    
    # Split
    random.shuffle(indexed_samples)
    split = int(len(indexed_samples) * 0.8)
    train_samples = indexed_samples[:split]
    test_samples = indexed_samples[split:]
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    train_ds = ZipImageDataset(train_samples, transform=train_transform)
    test_ds = ZipImageDataset(test_samples, transform=test_transform)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model
    model = FineTuneModel(n_classes=n_classes, freeze_backbone=True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n   Parameters: {trainable:,} trainable / {total:,} total")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    best_state = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total_n = 0
        t0 = time.time()
        
        for batch_idx, (xb, yb) in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * xb.size(0)
            _, predicted = out.max(1)
            correct += (predicted == yb).sum().item()
            total_n += xb.size(0)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"      batch {batch_idx+1}/{len(train_loader)}  "
                      f"loss={total_loss/total_n:.4f}  acc={correct/total_n*100:.1f}%")
        
        scheduler.step()
        train_acc = correct / total_n * 100
        
        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                out = model(xb)
                _, predicted = out.max(1)
                test_correct += (predicted == yb).sum().item()
                test_total += xb.size(0)
        test_acc = test_correct / test_total * 100
        
        elapsed = time.time() - t0
        print(f"   Epoch {epoch:>2d}/{epochs}  loss={total_loss/total_n:.4f}  "
              f"train={train_acc:.1f}%  test={test_acc:.1f}%  ({elapsed:.0f}s)")
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    print(f"\n   Best Test Accuracy: {best_acc:.2f}%")
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, all_classes, best_acc, n_classes


# ══════════════════════════════════════════════════════════════════════
#  Save & export
# ══════════════════════════════════════════════════════════════════════

def save_mlp_for_inference(model, all_classes, n_features):
    """Save the MLP classifier in a format compatible with the inference pipeline."""
    
    # Save PyTorch model
    model_info = {
        "state_dict": model.state_dict(),
        "n_features": n_features,
        "n_classes": len(all_classes),
        "classes": all_classes,
        "model_type": "mlp_classifier",
    }
    
    pt_path = os.path.join(BASE_DIR, "image_classifier.pt")
    torch.save(model_info, pt_path)
    print(f"   Saved classifier -> {pt_path}")
    
    # Also save classes for compatibility
    cls_path = os.path.join(BASE_DIR, "image_classes.pkl")
    with open(cls_path, "wb") as f:
        pickle.dump(all_classes, f)
    print(f"   Saved classes    -> {cls_path}")
    
    # Create a sklearn-compatible wrapper that app.py can load  
    wrapper = PyTorchModelWrapper(model, all_classes)
    import joblib
    wrapper_path = os.path.join(BASE_DIR, "image_model.pkl")
    joblib.dump(wrapper, wrapper_path, compress=3)
    sz = os.path.getsize(wrapper_path) / (1024 * 1024)
    print(f"   Saved wrapper    -> {wrapper_path}  ({sz:.1f} MB)")


class PyTorchModelWrapper:
    """Sklearn-compatible wrapper so app.py can use model.predict() / predict_proba()."""
    
    def __init__(self, model, classes):
        self.classes_ = np.array(classes)
        self._state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        self._n_features = None
        self._n_classes = len(classes)
        
        # Get architecture info
        for k, v in self._state_dict.items():
            if 'bn_input' in k and 'weight' in k:
                self._n_features = v.shape[0]
                break
        if self._n_features is None:
            self._n_features = 1280
        
        self._model = None
    
    def _ensure_model(self):
        if self._model is None:
            self._model = DeepClassifier(
                n_features=self._n_features, 
                n_classes=self._n_classes
            )
            self._model.load_state_dict(self._state_dict)
            self._model.eval()
    
    def predict(self, X):
        self._ensure_model()
        X_t = torch.tensor(np.array(X), dtype=torch.float32)
        with torch.no_grad():
            logits = self._model(X_t)
            preds = logits.argmax(dim=1).numpy()
        return preds
    
    def predict_proba(self, X):
        self._ensure_model()
        X_t = torch.tensor(np.array(X), dtype=torch.float32)
        with torch.no_grad():
            logits = self._model(X_t)
            probs = torch.softmax(logits, dim=1).numpy()
        return probs


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-finetune", action="store_true",
                        help="Run Stage 2: end-to-end fine-tuning")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=40)
    args = parser.parse_args()
    
    t0 = time.time()
    
    if args.full_finetune:
        model, classes, acc, n_cls = train_stage2(
            epochs=args.epochs, lr=1e-4, batch_size=32
        )
        # TODO: save fine-tuned model
        print(f"\n   Stage 2 accuracy: {acc:.2f}%")
    else:
        result = train_stage1(
            epochs=args.epochs, lr=args.lr, 
            batch_size=args.batch_size, patience=args.patience
        )
        if result is None:
            sys.exit(1)
        model, classes, acc, n_features = result
        
        print(f"\n   Saving model ...")
        save_mlp_for_inference(model, classes, n_features)
    
    elapsed = time.time() - t0
    print(f"\n   Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("=" * 65)
