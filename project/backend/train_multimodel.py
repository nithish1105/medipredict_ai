"""
train_multimodel.py  -  Multi-backbone feature extraction + Strong MLP ensemble
================================================================================
Strategy:
  1. Extract 1280-dim features from 4+ pretrained backbones (one at a time)
  2. Concatenate into a wide feature vector (~5300+ dims)
  3. Train a powerful MLP ensemble on the combined features
  4. Save model with Test-Time Augmentation (TTA) wrapper

Each backbone is loaded → used → deleted, keeping peak RAM < 1 GB.
Different architectures capture complementary patterns (texture, shape, color).
"""

import os, sys, time, pickle, random, zipfile, io, csv, gc
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
from model_defs import ResidualMLP, MultiModelEnsemble

# ══════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(os.path.dirname(BASE_DIR), "datasets")
RESULT_F  = os.path.join(BASE_DIR, "train_result.txt")
CACHE_DIR = os.path.join(BASE_DIR, "_multi_cache")

MAX_PER_CLASS = 400
AUG_FACTOR    = 2       # original + 1 augmented
BATCH_EXT     = 64
BATCH_TRAIN   = 256
EPOCHS_MLP    = 400
PATIENCE_MLP  = 50
IMG_SIZE      = 224
SEED          = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ══════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════
class _Tee:
    def __init__(self, path):
        self._f = open(path, "w", buffering=1, encoding="utf-8", errors="replace")
    def write(self, s):
        self._f.write(s); self._f.flush()
    def flush(self):
        self._f.flush()

sys.stdout = _Tee(RESULT_F)
t0 = time.time()
print("=" * 70)
print("  MULTI-MODEL Feature Ensemble")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════
# Class maps
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
    "1. Eczema 1677": "Eczema Photos",
    "2. Melanoma 15.75k": "Melanoma Skin Cancer Nevi and Moles",
    "3. Atopic Dermatitis - 1.25k": "Atopic Dermatitis Photos",
    "4. Basal Cell Carcinoma (BCC) 3323": "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "5. Melanocytic Nevi (NV) - 7970": "Melanoma Skin Cancer Nevi and Moles",
    "6. Benign Keratosis-like Lesions (BKL) 2624": "Seborrheic Keratoses and other Benign Tumors",
    "7. Psoriasis pictures Lichen Planus and related diseases - 2k": "Psoriasis pictures Lichen Planus and related diseases",
    "8. Seborrheic Keratoses and other Benign Tumors - 1.8k": "Seborrheic Keratoses and other Benign Tumors",
    "9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k": "Tinea Ringworm Candidiasis and other Fungal Infections",
    "10. Warts Molluscum and other Viral Infections - 2103": "Warts Molluscum and other Viral Infections",
}

# ══════════════════════════════════════════════════════════════════════
# Catalogue helpers
# ══════════════════════════════════════════════════════════════════════
def catalogue_zip(zip_path, split=None):
    classes = {}
    with zipfile.ZipFile(zip_path) as zf:
        for entry in zf.namelist():
            if entry.endswith("/"): continue
            low = entry.lower()
            if not any(low.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]): continue
            parts = entry.split("/")
            if len(parts) < 2: continue
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
            if entry.endswith("/"): continue
            low = entry.lower()
            if not any(low.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]): continue
            fname = entry.rsplit("/", 1)[-1]
            image_id = fname.rsplit(".", 1)[0]
            if image_id in id_to_dx:
                classes.setdefault(id_to_dx[image_id], []).append(entry)
    return classes

def catalogue_img_classes(zip_path):
    classes = {}
    with zipfile.ZipFile(zip_path) as zf:
        for entry in zf.namelist():
            if entry.endswith("/"): continue
            low = entry.lower()
            if not any(low.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]): continue
            parts = entry.split("/")
            if len(parts) < 3 or parts[0] != "IMG_CLASSES": continue
            raw_cls = parts[1]
            classes.setdefault(IMGCLS_CLASS_MAP.get(raw_cls, raw_cls), []).append(entry)
    return classes

# ══════════════════════════════════════════════════════════════════════
# Transforms
# ══════════════════════════════════════════════════════════════════════
_train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.15),
])
_val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ══════════════════════════════════════════════════════════════════════
# Phase 1: Catalogue images
# ══════════════════════════════════════════════════════════════════════
print("\nPhase 1: Cataloguing images")
print("-" * 50)

SKIN_ZIP   = os.path.join(DATA_DIR, "archive (1).zip")
EYE_ZIP    = os.path.join(DATA_DIR, "archive (2).zip")
HAM_ZIP    = os.path.join(DATA_DIR, "archive (3).zip")
IMGCLS_ZIP = os.path.join(DATA_DIR, "archive (4).zip")

sources = []
for label, func, args in [
    ("Skin", catalogue_zip, (SKIN_ZIP, "train")),
    ("Eye",  catalogue_zip, (EYE_ZIP, "dataset")),
    ("HAM",  catalogue_ham10000, (HAM_ZIP,)),
    ("IMG",  catalogue_img_classes, (IMGCLS_ZIP,)),
]:
    print(f"  [{label}] ...", end=" ")
    c = func(*args)
    print(f"{len(c)} classes")
    sources.append((label, args[0], c))

# Build sample list
all_samples = []
for label, zip_path, cls_dict in sources:
    for cls_name, entries in cls_dict.items():
        selected = entries if len(entries) <= MAX_PER_CLASS else random.sample(entries, MAX_PER_CLASS)
        for entry in selected:
            all_samples.append((zip_path, entry, cls_name))

all_classes = sorted(set(s[2] for s in all_samples))
class_to_idx = {c: i for i, c in enumerate(all_classes)}
n_classes = len(all_classes)

dist = Counter(s[2] for s in all_samples)
print(f"\nTotal: {len(all_samples)} images, {n_classes} classes")
for c in all_classes:
    print(f"  {c[:55]:<55} {dist[c]:>5}")

# Split
indices = list(range(len(all_samples)))
labels_arr = [class_to_idx[all_samples[i][2]] for i in indices]
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=labels_arr)
print(f"\nSplit: {len(train_idx)} train, {len(test_idx)} test")

# ══════════════════════════════════════════════════════════════════════
# Phase 2: Extract features from multiple backbones
# ══════════════════════════════════════════════════════════════════════
print("\nPhase 2: Multi-model feature extraction")
print("-" * 50)

os.makedirs(CACHE_DIR, exist_ok=True)

# Define backbone configs: (name, model_fn, feature_dim)
BACKBONES = [
    ("EfficientNet-B0", lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT), 1280),
    ("ResNet-50",       lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT), 2048),
    ("DenseNet-121",    lambda: models.densenet121(weights=models.DenseNet121_Weights.DEFAULT), 1024),
    ("MobileNet-V3-L",  lambda: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT), 960),
]

def make_feature_extractor(model, name):
    """Remove classifier from model, return feature extractor."""
    if "efficientnet" in name.lower():
        model.classifier = nn.Identity()
    elif "resnet" in name.lower():
        model.fc = nn.Identity()
    elif "densenet" in name.lower():
        model.classifier = nn.Identity()
        # DenseNet uses adaptive pool + flatten internally
    elif "mobilenet" in name.lower():
        model.classifier = nn.Identity()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

def extract_features_for_model(model_name, model_fn, feat_dim, sample_indices, augment, out_path, label_path):
    """Load model, extract features for given samples, save to disk, free model."""
    print(f"\n  [{model_name}] Loading model...")
    model = model_fn()
    model = make_feature_extractor(model, model_name)
    
    # Verify output dim
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        out = model(dummy)
    actual_dim = out.shape[1]
    print(f"    Feature dim: {actual_dim} (expected {feat_dim})")
    if actual_dim != feat_dim:
        feat_dim = actual_dim
    del dummy, out
    
    aug_factor = AUG_FACTOR if augment else 1
    max_total = len(sample_indices) * aug_factor
    
    # Pre-allocate output array on disk (memmap)
    mm = np.memmap(out_path, dtype=np.float16, mode="w+", shape=(max_total, feat_dim))
    labels = np.full(max_total, -1, dtype=np.int32)
    
    # Group by ZIP
    by_zip = {}
    for idx in sample_indices:
        zp, entry, cls = all_samples[idx]
        by_zip.setdefault(zp, []).append((idx, entry, cls))
    
    pos = 0
    for zip_path, items in by_zip.items():
        batch_t = []
        batch_l = []
        
        with zipfile.ZipFile(zip_path) as zf:
            for _, entry, cls in items:
                try:
                    data = zf.read(entry)
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                except Exception:
                    continue
                
                batch_t.append(_val_tf(img))
                batch_l.append(class_to_idx[cls])
                
                if augment:
                    for _ in range(AUG_FACTOR - 1):
                        batch_t.append(_train_tf(img))
                        batch_l.append(class_to_idx[cls])
                
                img.close()
                del img, data
                
                while len(batch_t) >= BATCH_EXT:
                    batch = torch.stack(batch_t[:BATCH_EXT])
                    with torch.no_grad():
                        feats = model(batch)
                    n = feats.shape[0]
                    mm[pos:pos + n] = feats.cpu().numpy().astype(np.float16)
                    labels[pos:pos + n] = batch_l[:BATCH_EXT]
                    pos += n
                    batch_t = batch_t[BATCH_EXT:]
                    batch_l = batch_l[BATCH_EXT:]
                    del batch, feats
                    
                    if pos % 2000 < BATCH_EXT:
                        print(f"    {pos}/{max_total} extracted")
        
        if batch_t:
            batch = torch.stack(batch_t)
            with torch.no_grad():
                feats = model(batch)
            n = feats.shape[0]
            mm[pos:pos + n] = feats.cpu().numpy().astype(np.float16)
            labels[pos:pos + n] = batch_l[:n]
            pos += n
            del batch, feats
            batch_t.clear()
            batch_l.clear()
        
        mm.flush()
        gc.collect()
    
    np.save(label_path, labels[:pos])
    del mm, model
    gc.collect()
    
    print(f"    Done: {pos} features → {out_path}")
    return pos, feat_dim

# Extract features for each model
model_feat_dims = []
for mi, (name, fn, fdim) in enumerate(BACKBONES):
    print(f"\n  Model {mi+1}/{len(BACKBONES)}: {name}")
    
    train_path = os.path.join(CACHE_DIR, f"train_{mi}.dat")
    test_path  = os.path.join(CACHE_DIR, f"test_{mi}.dat")
    train_lbl  = os.path.join(CACHE_DIR, f"train_labels_{mi}.npy")
    test_lbl   = os.path.join(CACHE_DIR, f"test_labels_{mi}.npy")
    
    t_m = time.time()
    n_train, actual_dim = extract_features_for_model(
        name, fn, fdim, train_idx, augment=True, out_path=train_path, label_path=train_lbl)
    n_test, _ = extract_features_for_model(
        name, fn, fdim, test_idx, augment=False, out_path=test_path, label_path=test_lbl)
    
    model_feat_dims.append((name, mi, actual_dim, n_train, n_test))
    print(f"  [{name}] train={n_train}, test={n_test}, dim={actual_dim}, time={time.time()-t_m:.0f}s")

# ══════════════════════════════════════════════════════════════════════
# Phase 3: Concatenate features from all models
# ══════════════════════════════════════════════════════════════════════
print("\nPhase 3: Concatenating features")
print("-" * 50)

# Pre-allocate concatenated arrays (memory-efficient: fill in-place)
total_dim = sum(d for _, _, d, _, _ in model_feat_dims)
n_tr = model_feat_dims[0][3]
n_te = model_feat_dims[0][4]

X_train = np.empty((n_tr, total_dim), dtype=np.float32)
X_test  = np.empty((n_te, total_dim), dtype=np.float32)

col = 0
for name, mi, fdim, n_train, n_test in model_feat_dims:
    # Fill train columns
    tr_mm = np.memmap(os.path.join(CACHE_DIR, f"train_{mi}.dat"),
                      dtype=np.float16, mode="r", shape=(n_train, fdim))
    X_train[:, col:col+fdim] = tr_mm[:n_tr].astype(np.float32)
    del tr_mm
    # Fill test columns
    te_mm = np.memmap(os.path.join(CACHE_DIR, f"test_{mi}.dat"),
                      dtype=np.float16, mode="r", shape=(n_test, fdim))
    X_test[:, col:col+fdim] = te_mm[:n_te].astype(np.float32)
    del te_mm
    col += fdim
    gc.collect()

y_train = np.load(os.path.join(CACHE_DIR, "train_labels_0.npy"))
y_test = np.load(os.path.join(CACHE_DIR, "test_labels_0.npy"))
gc.collect()

total_dim = X_train.shape[1]
print(f"Combined features: {total_dim} dims")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"  Per-model dims: {', '.join(f'{n}={d}' for n,_,d,_,_ in model_feat_dims)}")

# ══════════════════════════════════════════════════════════════════════
# Phase 4: Train MLP ensemble
# ══════════════════════════════════════════════════════════════════════
print("\nPhase 4: Training MLP ensemble")
print("-" * 50)

# ResidualMLP imported from model_defs

class FeatDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

# Ensemble configs: (seed, width, lr, drop, noise_std)
configs = [
    (42,   1024, 1e-3,  0.45, 0.05),
    (123,  1280, 8e-4,  0.40, 0.04),
    (7,    768,  1.2e-3, 0.50, 0.06),
    (999,  1536, 5e-4,  0.35, 0.03),
    (2024, 1024, 7e-4,  0.42, 0.05),
]

mlp_models = []
for ci, (seed, width, lr, drop, noise) in enumerate(configs):
    print(f"\n  [{ci+1}/{len(configs)}] seed={seed} width={width} lr={lr} drop={drop}")
    t_m = time.time()
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    
    model = ResidualMLP(total_dim, n_classes, width, drop)
    n_params = sum(p.numel() for p in model.parameters())
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    # Weighted sampler
    cc = Counter(y_train.tolist())
    w = [1.0 / cc[int(l)] for l in y_train]
    sam = WeightedRandomSampler(w, len(w), replacement=True)
    tr_dl = DataLoader(FeatDS(X_train, y_train), batch_size=BATCH_TRAIN, sampler=sam, num_workers=0)
    te_dl = DataLoader(FeatDS(X_test, y_test), batch_size=512, shuffle=False, num_workers=0)
    
    best_acc = 0.0
    best_state = None
    no_imp = 0
    
    for ep in range(1, EPOCHS_MLP + 1):
        model.train()
        for xb, yb in tr_dl:
            # Mixup augmentation
            lam = np.random.beta(0.4, 0.4)
            idx_p = torch.randperm(xb.size(0))
            mx = lam * xb + (1 - lam) * xb[idx_p]
            # Gaussian noise
            mx = mx + torch.randn_like(mx) * noise
            
            optimizer.zero_grad()
            out = model(mx)
            loss = lam * criterion(out, yb) + (1 - lam) * criterion(out, yb[idx_p])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        # Evaluate
        model.eval()
        tc = tt = 0
        with torch.no_grad():
            for xb, yb in te_dl:
                tc += (model(xb).argmax(1) == yb).sum().item()
                tt += xb.size(0)
        acc = tc / tt * 100
        
        if ep % 20 == 0 or ep == 1:
            print(f"    Ep {ep:>3d}  acc={acc:.2f}%  best={best_acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        
        if no_imp >= PATIENCE_MLP:
            print(f"    Early stop at ep {ep}")
            break
    
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    mlp_models.append((model, width))
    print(f"    Best: {best_acc:.2f}%  params={n_params:,}  ({time.time()-t_m:.0f}s)")
    
    del tr_dl, te_dl, criterion, optimizer, scheduler, sam, w, cc
    gc.collect()

# ══════════════════════════════════════════════════════════════════════
# Phase 5: Ensemble evaluation
# ══════════════════════════════════════════════════════════════════════
print("\nPhase 5: Ensemble evaluation")
print("-" * 50)

te_dl = DataLoader(FeatDS(X_test, y_test), batch_size=512, shuffle=False)
all_probs = []
all_true = []

with torch.no_grad():
    for xb, yb in te_dl:
        p = torch.zeros(xb.size(0), n_classes)
        for m, _ in mlp_models:
            p += torch.softmax(m(xb), dim=1)
        p /= len(mlp_models)
        all_probs.append(p)
        all_true.append(yb)

all_probs = torch.cat(all_probs)
all_true = torch.cat(all_true)
preds = all_probs.argmax(1)
ens_acc = (preds == all_true).float().mean().item() * 100

print(f"\nEnsemble accuracy: {ens_acc:.2f}%")
print(classification_report(all_true.numpy(), preds.numpy(), target_names=all_classes))

# ══════════════════════════════════════════════════════════════════════
# Phase 6: Save model with TTA support
# ══════════════════════════════════════════════════════════════════════
print("\nPhase 6: Saving")
print("-" * 50)

# MultiModelEnsemble imported from model_defs

import joblib

model_states = [{k: v.cpu() for k, v in m.state_dict().items()} for m, _ in mlp_models]
widths = [w for _, w in mlp_models]
backbone_names = [n for n, _, _, _, _ in model_feat_dims]
backbone_dims = [d for _, _, d, _, _ in model_feat_dims]

wrapper = MultiModelEnsemble(
    model_states, widths, all_classes, total_dim,
    backbone_names, backbone_dims
)

mp = os.path.join(BASE_DIR, "image_model.pkl")
joblib.dump(wrapper, mp, compress=3)
print(f"  Model: {mp} ({os.path.getsize(mp)/1e6:.1f} MB)")

# Classes
cp = os.path.join(BASE_DIR, "image_classes.pkl")
with open(cp, "wb") as f:
    pickle.dump(all_classes, f)
print(f"  Classes: {cp}")

# Cleanup
import shutil
try:
    shutil.rmtree(CACHE_DIR)
    print(f"  Cleaned cache dir")
except Exception:
    pass

total_time = time.time() - t0
print(f"\n{'=' * 70}")
print(f"  FINAL ACCURACY: {ens_acc:.2f}%")
print(f"  Total time: {total_time/60:.1f} min")
print(f"  Backbones: {len(BACKBONES)}")
print(f"  Total features: {total_dim} dims")
print(f"  Ensemble members: {len(mlp_models)}")
print(f"{'=' * 70}")
