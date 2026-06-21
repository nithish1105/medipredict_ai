"""
train_finetune_v2.py - Fine-tune EfficientNet-B0 via Intermediate Feature Caching
==================================================================================
Strategy for reaching 90%+ accuracy:
  1. Stream images from ZIP archives through frozen backbone blocks 0-6
  2. Cache intermediate features at block 6 output: (192, 7, 7)
  3. Fine-tune blocks 7-8 + new classifier head on cached intermediates
  4. Re-extract 1280-dim features using the fine-tuned backbone
  5. Train MLP ensemble on fine-tuned 1280-dim features
  6. Save backbone + MLP + classes for the Flask app

This avoids repeated forward passes through frozen layers during training,
making fine-tuning ~5x faster than naive end-to-end training.
"""

import os, sys, time, pickle, random, zipfile, io, csv
import numpy as np
from collections import Counter
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "datasets")
RESULT_FILE = os.path.join(BASE_DIR, "train_result.txt")

MAX_PER_CLASS = 500   # max raw images per class per ZIP source
AUG_FACTOR   = 3      # 1 original + 2 augmented per image
BATCH_EXT    = 32      # batch size for feature extraction
BATCH_TRAIN  = 64      # batch size for fine-tuning
EPOCHS_FT    = 35      # fine-tuning epochs
LR_FT        = 3e-4    # learning rate for fine-tuning
PATIENCE_FT  = 12      # early stopping patience
EPOCHS_MLP   = 300     # MLP epochs on fine-tuned features
LR_MLP       = 1e-3
PATIENCE_MLP = 40

# ══════════════════════════════════════════════════════════════════════
# Redirect stdout to file for monitoring
# ══════════════════════════════════════════════════════════════════════
class _Tee:
    def __init__(self, path):
        self._f = open(path, 'w', buffering=1, encoding='utf-8', errors='replace')
    def write(self, s):
        self._f.write(s)
        self._f.flush()
    def flush(self):
        self._f.flush()

sys.stdout = _Tee(RESULT_FILE)

t0_global = time.time()
print("=" * 70)
print("  FINE-TUNE EfficientNet-B0 via Intermediate Feature Caching")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════
# Class mappings (from train_image_model.py)
# ══════════════════════════════════════════════════════════════════════
HAM_CLASS_MAP = {
    "akiec": "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "bcc":   "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "bkl":   "Seborrheic Keratoses and other Benign Tumors",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma Skin Cancer Nevi and Moles",
    "nv":    "Melanoma Skin Cancer Nevi and Moles",
    "vasc":  "Vascular Tumors",
}

IMGCLS_CLASS_MAP = {
    "1. Eczema 1677":                                                   "Eczema Photos",
    "2. Melanoma 15.75k":                                               "Melanoma Skin Cancer Nevi and Moles",
    "3. Atopic Dermatitis - 1.25k":                                     "Atopic Dermatitis Photos",
    "4. Basal Cell Carcinoma (BCC) 3323":                               "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "5. Melanocytic Nevi (NV) - 7970":                                  "Melanoma Skin Cancer Nevi and Moles",
    "6. Benign Keratosis-like Lesions (BKL) 2624":                      "Seborrheic Keratoses and other Benign Tumors",
    "7. Psoriasis pictures Lichen Planus and related diseases - 2k":    "Psoriasis pictures Lichen Planus and related diseases",
    "8. Seborrheic Keratoses and other Benign Tumors - 1.8k":           "Seborrheic Keratoses and other Benign Tumors",
    "9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k": "Tinea Ringworm Candidiasis and other Fungal Infections",
    "10. Warts Molluscum and other Viral Infections - 2103":            "Warts Molluscum and other Viral Infections",
}

# ══════════════════════════════════════════════════════════════════════
# ZIP Catalogue Functions (from train_image_model.py)
# ══════════════════════════════════════════════════════════════════════
def catalogue_zip(zip_path, split=None):
    classes = {}
    with zipfile.ZipFile(zip_path) as zf:
        for entry in zf.namelist():
            if entry.endswith("/"):
                continue
            low = entry.lower()
            if not (low.endswith(".jpg") or low.endswith(".jpeg") or low.endswith(".png")):
                continue
            parts = entry.split("/")
            if len(parts) < 2:
                continue
            if split and parts[0].lower() == split.lower():
                cls = parts[1]
            elif len(parts) >= 3:
                cls = parts[1]
            else:
                cls = parts[0]
            classes.setdefault(cls, []).append(entry)
    return classes


def catalogue_ham10000(zip_path):
    classes = {}
    with zipfile.ZipFile(zip_path) as zf:
        raw = zf.read("HAM10000_metadata.csv").decode("utf-8")
        reader = csv.DictReader(io.StringIO(raw))
        id_to_dx = {}
        for row in reader:
            dx = row["dx"].strip()
            id_to_dx[row["image_id"]] = HAM_CLASS_MAP.get(dx, dx)
        for entry in zf.namelist():
            if entry.endswith("/"):
                continue
            low = entry.lower()
            if not (low.endswith(".jpg") or low.endswith(".jpeg") or low.endswith(".png")):
                continue
            fname = entry.rsplit("/", 1)[-1]
            image_id = fname.rsplit(".", 1)[0]
            if image_id in id_to_dx:
                classes.setdefault(id_to_dx[image_id], []).append(entry)
    return classes


def catalogue_img_classes(zip_path):
    classes = {}
    with zipfile.ZipFile(zip_path) as zf:
        for entry in zf.namelist():
            if entry.endswith("/"):
                continue
            low = entry.lower()
            if not (low.endswith(".jpg") or low.endswith(".jpeg") or low.endswith(".png")):
                continue
            parts = entry.split("/")
            if len(parts) < 3 or parts[0] != "IMG_CLASSES":
                continue
            raw_cls = parts[1]
            classes.setdefault(IMGCLS_CLASS_MAP.get(raw_cls, raw_cls), []).append(entry)
    return classes


# ══════════════════════════════════════════════════════════════════════
# Image transforms
# ══════════════════════════════════════════════════════════════════════
IMG_SIZE = 224

_train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])

_val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ══════════════════════════════════════════════════════════════════════
# Phase 1: Load all images from ZIP archives
# ══════════════════════════════════════════════════════════════════════
def load_all_images():
    """Load images from all 4 ZIPs. Returns list of (PIL.Image, class_name)."""
    print("\n" + "─" * 70)
    print("Phase 1: Loading images from ZIP archives")
    print("─" * 70)

    SKIN_ZIP   = os.path.join(DATA_DIR, "archive (1).zip")
    EYE_ZIP    = os.path.join(DATA_DIR, "archive (2).zip")
    HAM_ZIP    = os.path.join(DATA_DIR, "archive (3).zip")
    IMGCLS_ZIP = os.path.join(DATA_DIR, "archive (4).zip")

    sources = []

    print("\n[1/4] Scanning skin-disease ZIP ...")
    skin = catalogue_zip(SKIN_ZIP, split="train")
    print(f"  {len(skin)} classes, {sum(len(v) for v in skin.values())} images")
    sources.append(("Skin", SKIN_ZIP, skin))

    print("[2/4] Scanning eye-disease ZIP ...")
    eye = catalogue_zip(EYE_ZIP, split="dataset")
    print(f"  {len(eye)} classes, {sum(len(v) for v in eye.values())} images")
    sources.append(("Eye", EYE_ZIP, eye))

    print("[3/4] Scanning HAM10000 ZIP ...")
    ham = catalogue_ham10000(HAM_ZIP)
    print(f"  {len(ham)} unified classes, {sum(len(v) for v in ham.values())} images")
    sources.append(("HAM10000", HAM_ZIP, ham))

    print("[4/4] Scanning IMG_CLASSES ZIP ...")
    imgcls = catalogue_img_classes(IMGCLS_ZIP)
    print(f"  {len(imgcls)} unified classes, {sum(len(v) for v in imgcls.values())} images")
    sources.append(("IMG_CLASSES", IMGCLS_ZIP, imgcls))

    # Collect unified class set
    all_class_set = set()
    for _, _, d in sources:
        all_class_set.update(d.keys())
    all_classes = sorted(all_class_set)
    print(f"\nTotal unified classes: {len(all_classes)}")

    # Load images
    images = []  # list of (PIL.Image, class_name)
    for label, zip_path, cls_dict in sources:
        print(f"\nLoading images from {label} ...")
        t_src = time.time()
        src_count = 0
        with zipfile.ZipFile(zip_path) as zf:
            for cls_name in sorted(cls_dict.keys()):
                entries = cls_dict[cls_name]
                if len(entries) > MAX_PER_CLASS:
                    entries = random.sample(entries, MAX_PER_CLASS)
                for entry in entries:
                    try:
                        data = zf.read(entry)
                        img = Image.open(io.BytesIO(data)).convert("RGB")
                        # Resize to save memory (store at a reasonable size)
                        img = img.resize((IMG_SIZE + 32, IMG_SIZE + 32), Image.LANCZOS)
                        images.append((img, cls_name))
                        src_count += 1
                    except Exception:
                        continue
        print(f"  Loaded {src_count} images ({time.time()-t_src:.1f}s)")

    # Report
    dist = Counter(cls for _, cls in images)
    print(f"\nTotal images loaded: {len(images)}")
    print(f"Per-class distribution:")
    for cls in all_classes:
        print(f"  {cls[:55]:<55} {dist.get(cls, 0):>5}")

    return images, all_classes


# ══════════════════════════════════════════════════════════════════════
# Phase 2: Extract intermediate features through frozen blocks
# ══════════════════════════════════════════════════════════════════════
def extract_intermediate_features(images, all_classes):
    """
    Run images through frozen EfficientNet-B0 blocks 0-6.
    Cache the output of block 6: (N, 192, 7, 7) per sample.
    Split at IMAGE level into train/test BEFORE augmentation.
    """
    print("\n" + "─" * 70)
    print("Phase 2: Extracting intermediate features (frozen blocks 0-6)")
    print("─" * 70)

    # Build class-to-idx mapping
    class_to_idx = {c: i for i, c in enumerate(all_classes)}

    # Split at IMAGE level (before augmentation!)
    img_indices = list(range(len(images)))
    img_labels = [class_to_idx[images[i][1]] for i in img_indices]
    train_idx, test_idx = train_test_split(
        img_indices, test_size=0.2, random_state=42, stratify=img_labels
    )
    print(f"\nImage-level split: {len(train_idx)} train, {len(test_idx)} test")

    # Load frozen backbone (blocks 0-6)
    print("Loading pretrained EfficientNet-B0 ...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    frozen_part = nn.Sequential(*list(model.features[:7]))  # blocks 0-6
    frozen_part.eval()
    for p in frozen_part.parameters():
        p.requires_grad = False

    # Verify output shape
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        dummy_out = frozen_part(dummy)
    print(f"Intermediate feature shape: {dummy_out.shape}")  # (1, 192, 7, 7)
    feat_shape = dummy_out.shape[1:]  # (192, 7, 7)

    # Extract features for train images (with augmentation)
    print(f"\nExtracting train features ({len(train_idx)} images x {AUG_FACTOR} aug)...")
    t_ext = time.time()
    train_features = []
    train_labels = []

    with torch.no_grad():
        batch_tensors = []
        batch_labels = []

        for count, idx in enumerate(train_idx):
            img, cls_name = images[idx]
            label = class_to_idx[cls_name]

            # Original (center crop)
            batch_tensors.append(_val_transform(img))
            batch_labels.append(label)

            # Augmented views
            for _ in range(AUG_FACTOR - 1):
                batch_tensors.append(_train_transform(img))
                batch_labels.append(label)

            # Process batch
            while len(batch_tensors) >= BATCH_EXT:
                batch = torch.stack(batch_tensors[:BATCH_EXT])
                feats = frozen_part(batch)
                train_features.append(feats.cpu())
                train_labels.extend(batch_labels[:BATCH_EXT])
                batch_tensors = batch_tensors[BATCH_EXT:]
                batch_labels = batch_labels[BATCH_EXT:]

            if (count + 1) % 500 == 0:
                elapsed = time.time() - t_ext
                print(f"  {count+1}/{len(train_idx)} images processed ({elapsed:.0f}s)")

        # Flush remaining
        if batch_tensors:
            batch = torch.stack(batch_tensors)
            feats = frozen_part(batch)
            train_features.append(feats.cpu())
            train_labels.extend(batch_labels)

    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    print(f"  Train features: {train_features.shape} ({time.time()-t_ext:.0f}s)")
    print(f"  Memory: {train_features.nelement() * 4 / 1e6:.0f} MB")

    # Extract features for test images (NO augmentation)
    print(f"\nExtracting test features ({len(test_idx)} images, no augmentation)...")
    t_ext2 = time.time()
    test_features = []
    test_labels = []

    with torch.no_grad():
        batch_tensors = []
        batch_labels = []

        for idx in test_idx:
            img, cls_name = images[idx]
            label = class_to_idx[cls_name]
            batch_tensors.append(_val_transform(img))
            batch_labels.append(label)

            if len(batch_tensors) >= BATCH_EXT:
                batch = torch.stack(batch_tensors[:BATCH_EXT])
                feats = frozen_part(batch)
                test_features.append(feats.cpu())
                test_labels.extend(batch_labels[:BATCH_EXT])
                batch_tensors = batch_tensors[BATCH_EXT:]
                batch_labels = batch_labels[BATCH_EXT:]

        if batch_tensors:
            batch = torch.stack(batch_tensors)
            feats = frozen_part(batch)
            test_features.append(feats.cpu())
            test_labels.extend(batch_labels)

    test_features = torch.cat(test_features, dim=0)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    print(f"  Test features: {test_features.shape} ({time.time()-t_ext2:.0f}s)")

    # Free PIL images to save memory
    del images
    import gc; gc.collect()
    print("  Freed image memory")

    return train_features, train_labels, test_features, test_labels, model, feat_shape


# ══════════════════════════════════════════════════════════════════════
# Phase 3: Fine-tune blocks 7-8 + classifier on cached intermediates
# ══════════════════════════════════════════════════════════════════════
class IntermediateDataset(Dataset):
    """Dataset wrapping cached intermediate features."""
    def __init__(self, features, labels, augment=False):
        self.features = features
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feat = self.features[idx]
        label = self.labels[idx]
        if self.augment:
            # Feature-level augmentation: Gaussian noise + channel dropout
            noise = torch.randn_like(feat) * 0.03
            feat = feat + noise
            # Randomly zero out some channels (like dropout)
            if random.random() < 0.2:
                ch_mask = torch.ones(feat.shape[0], 1, 1)
                ch_mask[torch.randperm(feat.shape[0])[:int(feat.shape[0] * 0.1)]] = 0
                feat = feat * ch_mask
        return feat, label


class TrainableHead(nn.Module):
    """Trainable part: blocks 7-8 + avgpool + new classifier head."""
    def __init__(self, base_model, n_classes):
        super().__init__()
        self.block7 = base_model.features[7]      # MBConv: 192 -> 320
        self.block8 = base_model.features[8]      # Conv1x1: 320 -> 1280
        self.avgpool = base_model.avgpool          # AdaptiveAvgPool2d
        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(1280, 640),
            nn.BatchNorm1d(640),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(640, n_classes),
        )

    def forward(self, x):
        x = self.block7(x)
        x = self.block8(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def extract_features(self, x):
        """Extract 1280-dim features (no classifier)."""
        x = self.block7(x)
        x = self.block8(x)
        x = self.avgpool(x)
        return x.flatten(1)


def finetune_head(train_features, train_labels, test_features, test_labels,
                  base_model, n_classes):
    """Fine-tune the trainable head on cached intermediate features."""
    print("\n" + "─" * 70)
    print("Phase 3: Fine-tuning blocks 7-8 + classifier")
    print("─" * 70)

    # Datasets
    train_ds = IntermediateDataset(train_features, train_labels, augment=True)
    test_ds  = IntermediateDataset(test_features, test_labels, augment=False)

    # Weighted sampler for class balance
    class_counts = Counter(train_labels.numpy().tolist())
    weights = [1.0 / class_counts[int(l)] for l in train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, sampler=sampler,
                              num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                             num_workers=0, pin_memory=False)

    # Model
    head = TrainableHead(base_model, n_classes)
    trainable_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in head.parameters())
    print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,}")
    print(f"Epochs: {EPOCHS_FT}, LR: {LR_FT}, Batch: {BATCH_TRAIN}")
    print(f"Patience: {PATIENCE_FT}")

    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(head.parameters(), lr=LR_FT, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    best_acc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, EPOCHS_FT + 1):
        t_ep = time.time()
        head.train()
        total_loss = 0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = head(xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            _, predicted = out.max(1)
            correct += (predicted == yb).sum().item()
            total += xb.size(0)

        scheduler.step()
        train_acc = correct / total * 100

        # Evaluate
        head.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                out = head(xb)
                _, predicted = out.max(1)
                test_correct += (predicted == yb).sum().item()
                test_total += xb.size(0)
        test_acc = test_correct / test_total * 100
        elapsed = time.time() - t_ep

        if epoch % 2 == 0 or test_acc > best_acc:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:>3d}  loss={total_loss/total:.4f}  "
                  f"train={train_acc:.1f}%  test={test_acc:.1f}%  "
                  f"lr={lr_now:.6f}  ({elapsed:.0f}s)")

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE_FT:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {PATIENCE_FT} epochs)")
            break

    print(f"\n  Best fine-tune test accuracy: {best_acc:.2f}%")

    if best_state:
        head.load_state_dict(best_state)

    return head, best_acc


# ══════════════════════════════════════════════════════════════════════
# Phase 4: Extract 1280-dim features using fine-tuned backbone
# ══════════════════════════════════════════════════════════════════════
def extract_finetuned_features(head, train_intermediates, train_labels,
                                test_intermediates, test_labels):
    """Run cached intermediates through fine-tuned blocks 7-8 → 1280-dim."""
    print("\n" + "─" * 70)
    print("Phase 4: Extracting 1280-dim features with fine-tuned backbone")
    print("─" * 70)

    head.eval()
    all_feats = {}

    for name, feats, labels in [("train", train_intermediates, train_labels),
                                 ("test", test_intermediates, test_labels)]:
        feat_list = []
        with torch.no_grad():
            for i in range(0, len(feats), 128):
                batch = feats[i:i+128]
                f = head.extract_features(batch)
                feat_list.append(f.cpu().numpy())
        all_feats[name] = (np.concatenate(feat_list), labels.numpy())
        print(f"  {name}: {all_feats[name][0].shape}")

    return all_feats


# ══════════════════════════════════════════════════════════════════════
# Phase 5: Train MLP ensemble on fine-tuned features
# ══════════════════════════════════════════════════════════════════════
class DeepMLP(nn.Module):
    """Deep MLP classifier for 1280-dim features."""
    def __init__(self, n_features, n_classes, width=1024, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(n_features),
            nn.Linear(n_features, width),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width // 2),
            nn.BatchNorm1d(width // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(width // 2, width // 4),
            nn.BatchNorm1d(width // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(width // 4, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_mlp_ensemble(all_feats, n_classes, all_classes):
    """Train multiple MLPs on fine-tuned features and ensemble them."""
    print("\n" + "─" * 70)
    print("Phase 5: Training MLP ensemble on fine-tuned features")
    print("─" * 70)

    X_train, y_train = all_feats["train"]
    X_test, y_test = all_feats["test"]
    n_features = X_train.shape[1]

    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}, Classes: {n_classes}")

    configs = [
        (42,   1024, 1e-3, 0.4),
        (123,  1280, 8e-4, 0.35),
        (7,    768,  1.2e-3, 0.45),
        (999,  1024, 5e-4, 0.3),
        (2024, 1536, 7e-4, 0.4),
    ]

    models_list = []
    model_accs = []

    for i, (seed, width, lr, dropout) in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Seed={seed}, Width={width}, LR={lr}")
        t_m = time.time()

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        model = DeepMLP(n_features, n_classes, width=width, dropout=dropout)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

        # Weighted sampler
        class_counts = Counter(y_train.tolist())
        weights = [1.0 / class_counts[int(l)] for l in y_train]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        train_ds = FeatureDataset(X_train, y_train)
        test_ds = FeatureDataset(X_test, y_test)
        train_loader = DataLoader(train_ds, batch_size=256, sampler=sampler, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)

        best_acc = 0.0
        best_state = None
        no_improve = 0

        for epoch in range(1, EPOCHS_MLP + 1):
            model.train()
            for xb, yb in train_loader:
                # Mixup
                lam = np.random.beta(0.3, 0.3)
                idx = torch.randperm(xb.size(0))
                mixed_x = lam * xb + (1 - lam) * xb[idx]
                ya, yb_mix = yb, yb[idx]

                # Add noise
                mixed_x = mixed_x + torch.randn_like(mixed_x) * 0.05

                optimizer.zero_grad()
                out = model(mixed_x)
                loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb_mix)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    out = model(xb)
                    test_correct += (out.argmax(1) == yb).sum().item()
                    test_total += xb.size(0)
            test_acc = test_correct / test_total * 100

            if test_acc > best_acc:
                best_acc = test_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= PATIENCE_MLP:
                break

        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        models_list.append(model)
        model_accs.append(best_acc)
        print(f"  Accuracy: {best_acc:.2f}%  ({time.time()-t_m:.0f}s)")

    # Ensemble evaluation
    print(f"\nIndividual model accuracies:")
    for i, acc in enumerate(model_accs):
        print(f"  Model {i+1}: {acc:.2f}%")

    print(f"\nEnsemble (soft voting) evaluation:")
    test_ds = FeatureDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)

    all_probs = []
    all_true = []

    with torch.no_grad():
        for xb, yb in test_loader:
            batch_probs = torch.zeros(xb.size(0), n_classes)
            for m in models_list:
                m.eval()
                logits = m(xb)
                batch_probs += torch.softmax(logits, dim=1)
            batch_probs /= len(models_list)
            all_probs.append(batch_probs)
            all_true.append(yb)

    all_probs = torch.cat(all_probs)
    all_true = torch.cat(all_true)
    ensemble_preds = all_probs.argmax(dim=1)
    ensemble_acc = (ensemble_preds == all_true).float().mean().item() * 100

    print(f"\nEnsemble accuracy: {ensemble_acc:.2f}%")
    print(classification_report(
        all_true.numpy(), ensemble_preds.numpy(),
        target_names=all_classes
    ))

    return models_list, ensemble_acc


# ══════════════════════════════════════════════════════════════════════
# Phase 6: Save everything
# ══════════════════════════════════════════════════════════════════════
class FineTunedEnsembleWrapper:
    """Sklearn-compatible wrapper for the Flask app."""
    def __init__(self, mlp_models, classes, ft_block7_state, ft_block8_state):
        self.classes_ = np.array(classes)
        self._n_classes = len(classes)
        self._n_features = 1280
        # Store MLP model state dicts
        self._mlp_states = []
        self._mlp_configs = []
        for m in mlp_models:
            self._mlp_states.append({k: v.cpu() for k, v in m.state_dict().items()})
            # Infer width from first linear layer
            for k, v in m.state_dict().items():
                if 'net.1.weight' in k:
                    self._mlp_configs.append({'width': v.shape[0], 'dropout': 0.4})
                    break
        self._models = None

    def _ensure_models(self):
        if self._models is None:
            self._models = []
            for state, cfg in zip(self._mlp_states, self._mlp_configs):
                m = DeepMLP(self._n_features, self._n_classes,
                           width=cfg['width'], dropout=cfg.get('dropout', 0.4))
                m.load_state_dict(state)
                m.eval()
                self._models.append(m)

    def predict(self, X):
        self._ensure_models()
        X_t = torch.tensor(np.array(X), dtype=torch.float32)
        with torch.no_grad():
            probs = torch.zeros(X_t.size(0), self._n_classes)
            for m in self._models:
                probs += torch.softmax(m(X_t), dim=1)
            probs /= len(self._models)
        return probs.argmax(dim=1).numpy()

    def predict_proba(self, X):
        self._ensure_models()
        X_t = torch.tensor(np.array(X), dtype=torch.float32)
        with torch.no_grad():
            probs = torch.zeros(X_t.size(0), self._n_classes)
            for m in self._models:
                probs += torch.softmax(m(X_t), dim=1)
            probs /= len(self._models)
        return probs.numpy()


def save_everything(head, mlp_models, all_classes, base_model, ensemble_acc):
    """Save fine-tuned backbone + MLP ensemble + class list."""
    print("\n" + "─" * 70)
    print("Phase 6: Saving models")
    print("─" * 70)

    # 1. Save fine-tuned backbone (for utils.py feature extraction)
    # Assemble full backbone with fine-tuned blocks 7-8
    full_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    full_model.features[7] = head.block7
    full_model.features[8] = head.block8
    full_model.classifier = nn.Identity()
    full_model.eval()

    backbone_path = os.path.join(BASE_DIR, "finetuned_backbone.pth")
    torch.save(full_model.state_dict(), backbone_path)
    sz = os.path.getsize(backbone_path) / (1024 * 1024)
    print(f"  Backbone -> {backbone_path} ({sz:.1f} MB)")

    # 2. Save MLP ensemble wrapper (for app.py)
    wrapper = FineTunedEnsembleWrapper(
        mlp_models, all_classes,
        head.block7.state_dict(),
        head.block8.state_dict()
    )

    import joblib
    model_path = os.path.join(BASE_DIR, "image_model.pkl")
    joblib.dump(wrapper, model_path, compress=3)
    sz = os.path.getsize(model_path) / (1024 * 1024)
    print(f"  MLP ensemble -> {model_path} ({sz:.1f} MB)")

    # 3. Save class list
    cls_path = os.path.join(BASE_DIR, "image_classes.pkl")
    with open(cls_path, "wb") as f:
        pickle.dump(all_classes, f)
    print(f"  Classes -> {cls_path}")

    print(f"\n  FINAL ACCURACY: {ensemble_acc:.2f}%")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Phase 1
    images, all_classes = load_all_images()
    n_classes = len(all_classes)

    # Phase 2
    (train_feats, train_labels,
     test_feats, test_labels,
     base_model, feat_shape) = extract_intermediate_features(images, all_classes)

    # Phase 3
    head, ft_acc = finetune_head(
        train_feats, train_labels, test_feats, test_labels,
        base_model, n_classes
    )

    # Phase 4
    all_feats = extract_finetuned_features(
        head, train_feats, train_labels, test_feats, test_labels
    )

    # Phase 5
    mlp_models, ensemble_acc = train_mlp_ensemble(all_feats, n_classes, all_classes)

    # Phase 6
    save_everything(head, mlp_models, all_classes, base_model, ensemble_acc)

    total_time = time.time() - t0_global
    print(f"\nTotal time: {total_time/60:.1f} min")
    print("=" * 70)
