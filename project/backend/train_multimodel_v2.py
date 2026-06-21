"""
train_multimodel_v2.py  -  Multi-backbone ensemble (simple, no memmap)
======================================================================
Extracts features from 4 pretrained CNN backbones, one at a time.
Full-model features are small (~100-200 MB each), so plain numpy arrays 
work fine. Each model is loaded → extracted → deleted before the next.

Peak RAM: ~800 MB (model + feature array + images).
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
from sklearn.metrics import classification_report

# Import model definitions for pickle compatibility
from model_defs import ResidualMLP, MultiModelEnsemble

# ══════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(os.path.dirname(BASE_DIR), "datasets")
RESULT_F  = os.path.join(BASE_DIR, "train_result.txt")
SEED      = 42

MAX_PER_CLASS = 400
AUG_FACTOR    = 2
BATCH_EXT     = 32
BATCH_TRAIN   = 256
EPOCHS_MLP    = 400
PATIENCE_MLP  = 50
IMG_SIZE      = 224

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ══════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════
class _Tee:
    def __init__(self, p):
        self._f = open(p, "w", buffering=1, encoding="utf-8", errors="replace")
    def write(self, s): self._f.write(s); self._f.flush()
    def flush(self): self._f.flush()

sys.stdout = _Tee(RESULT_F)
t0 = time.time()
print("=" * 70)
print("  MULTI-MODEL v2  (simple numpy, no memmap)")
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
            id_to_dx[row["image_id"]] = HAM_CLASS_MAP.get(row["dx"].strip(), row["dx"].strip())
        for entry in zf.namelist():
            if entry.endswith("/"): continue
            low = entry.lower()
            if not any(low.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]): continue
            image_id = entry.rsplit("/", 1)[-1].rsplit(".", 1)[0]
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
            cls = IMGCLS_CLASS_MAP.get(parts[1], parts[1])
            classes.setdefault(cls, []).append(entry)
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
# Phase 1: Catalogue
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

all_samples = []
for _, zip_path, cls_dict in sources:
    for cls_name, entries in cls_dict.items():
        sel = entries if len(entries) <= MAX_PER_CLASS else random.sample(entries, MAX_PER_CLASS)
        for entry in sel:
            all_samples.append((zip_path, entry, cls_name))

all_classes = sorted(set(s[2] for s in all_samples))
class_to_idx = {c: i for i, c in enumerate(all_classes)}
n_classes = len(all_classes)
dist = Counter(s[2] for s in all_samples)

print(f"\nTotal: {len(all_samples)} images, {n_classes} classes")
for c in all_classes:
    print(f"  {c[:55]:<55} {dist[c]:>5}")

indices = list(range(len(all_samples)))
labels_arr = [class_to_idx[all_samples[i][2]] for i in indices]
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=labels_arr)
print(f"\nSplit: {len(train_idx)} train, {len(test_idx)} test")

# ══════════════════════════════════════════════════════════════════════
# Phase 2: Extract features from each backbone
# ══════════════════════════════════════════════════════════════════════
print("\nPhase 2: Feature extraction (one model at a time)")
print("-" * 50)

BACKBONES = [
    ("EfficientNet-B0", lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT), 1280),
    ("ResNet-50",       lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT), 2048),
    ("DenseNet-121",    lambda: models.densenet121(weights=models.DenseNet121_Weights.DEFAULT), 1024),
    ("MobileNet-V3-L",  lambda: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT), 960),
]

def make_extractor(model, name):
    """Remove classifier head, return feature extractor."""
    n = name.lower()
    if "efficientnet" in n: model.classifier = nn.Identity()
    elif "resnet" in n: model.fc = nn.Identity()
    elif "densenet" in n: model.classifier = nn.Identity()
    elif "mobilenet" in n: model.classifier = nn.Identity()
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    return model

def extract_one_model(model, sample_indices, augment):
    """
    Extract features for given samples.
    Returns: (features_np, labels_np) as plain numpy arrays.
    """
    aug_factor = AUG_FACTOR if augment else 1
    max_n = len(sample_indices) * aug_factor
    
    # Pre-allocate ONE numpy array (not a list of tensors)
    feat_dim = None
    all_feats = None
    all_labels = np.empty(max_n, dtype=np.int32)
    pos = 0
    
    # Group by ZIP
    by_zip = {}
    for idx in sample_indices:
        zp, entry, cls = all_samples[idx]
        by_zip.setdefault(zp, []).append((entry, cls))
    
    for zip_path, items in by_zip.items():
        batch_tensors = []
        batch_labels = []
        
        with zipfile.ZipFile(zip_path) as zf:
            for entry, cls in items:
                try:
                    raw = zf.read(entry)
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                except Exception:
                    continue
                
                # Original
                batch_tensors.append(_val_tf(img))
                batch_labels.append(class_to_idx[cls])
                
                # Augmented
                if augment:
                    for _ in range(AUG_FACTOR - 1):
                        batch_tensors.append(_train_tf(img))
                        batch_labels.append(class_to_idx[cls])
                
                img.close()
                del img, raw
                
                # Process full batch
                while len(batch_tensors) >= BATCH_EXT:
                    batch = torch.stack(batch_tensors[:BATCH_EXT])
                    with torch.no_grad():
                        out = model(batch).cpu().numpy()
                    
                    n = out.shape[0]
                    if all_feats is None:
                        feat_dim = out.shape[1]
                        all_feats = np.empty((max_n, feat_dim), dtype=np.float32)
                    
                    all_feats[pos:pos+n] = out
                    all_labels[pos:pos+n] = batch_labels[:BATCH_EXT]
                    pos += n
                    
                    batch_tensors = batch_tensors[BATCH_EXT:]
                    batch_labels = batch_labels[BATCH_EXT:]
                    del batch, out
                    
                    if pos % 2000 < BATCH_EXT:
                        print(f"    {pos}/{max_n}")
        
        # Flush remaining for this ZIP
        if batch_tensors:
            batch = torch.stack(batch_tensors)
            with torch.no_grad():
                out = model(batch).cpu().numpy()
            n = out.shape[0]
            if all_feats is None:
                feat_dim = out.shape[1]
                all_feats = np.empty((max_n, feat_dim), dtype=np.float32)
            all_feats[pos:pos+n] = out
            all_labels[pos:pos+n] = batch_labels[:n]
            pos += n
            del batch, out
            batch_tensors.clear()
            batch_labels.clear()
        
        # GC after each ZIP
        gc.collect()
    
    return all_feats[:pos], all_labels[:pos], feat_dim

# Extract from each model, save to .npy, free model
CACHE_DIR = os.path.join(BASE_DIR, "_mm_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

model_info = []  # (name, feat_dim)

for mi, (name, model_fn, expected_dim) in enumerate(BACKBONES):
    print(f"\n  [{mi+1}/{len(BACKBONES)}] {name}")
    t_m = time.time()
    
    model = model_fn()
    model = make_extractor(model, name)
    
    # Verify
    with torch.no_grad():
        d = model(torch.randn(1, 3, IMG_SIZE, IMG_SIZE))
    actual_dim = d.shape[1]
    print(f"    dim={actual_dim}")
    del d
    
    # Train
    print(f"    Extracting train ({len(train_idx)} × {AUG_FACTOR})...")
    tr_feats, tr_labels, fdim = extract_one_model(model, train_idx, augment=True)
    np.save(os.path.join(CACHE_DIR, f"tr_feat_{mi}.npy"), tr_feats)
    np.save(os.path.join(CACHE_DIR, f"tr_lab_{mi}.npy"), tr_labels)
    n_train = tr_feats.shape[0]
    del tr_feats, tr_labels
    gc.collect()
    
    # Test
    print(f"    Extracting test ({len(test_idx)})...")
    te_feats, te_labels, _ = extract_one_model(model, test_idx, augment=False)
    np.save(os.path.join(CACHE_DIR, f"te_feat_{mi}.npy"), te_feats)
    np.save(os.path.join(CACHE_DIR, f"te_lab_{mi}.npy"), te_labels)
    n_test = te_feats.shape[0]
    del te_feats, te_labels
    gc.collect()
    
    # Free model completely
    del model
    gc.collect()
    
    elapsed = time.time() - t_m
    model_info.append((name, fdim))
    print(f"    Done: train={n_train}, test={n_test}, dim={fdim} ({elapsed:.0f}s)")

# ══════════════════════════════════════════════════════════════════════
# Phase 3: Concatenate features
# ══════════════════════════════════════════════════════════════════════
print("\nPhase 3: Concatenating features")
print("-" * 50)

total_dim = sum(d for _, d in model_info)
print(f"  Total feature dim: {total_dim}")
print(f"  Per model: {', '.join(f'{n}={d}' for n, d in model_info)}")

# Load one at a time and fill concatenated array
tr_lab = np.load(os.path.join(CACHE_DIR, "tr_lab_0.npy"))
te_lab = np.load(os.path.join(CACHE_DIR, "te_lab_0.npy"))
n_tr, n_te = len(tr_lab), len(te_lab)

X_train = np.empty((n_tr, total_dim), dtype=np.float32)
X_test  = np.empty((n_te, total_dim), dtype=np.float32)

col = 0
for mi, (name, fdim) in enumerate(model_info):
    tr = np.load(os.path.join(CACHE_DIR, f"tr_feat_{mi}.npy"))
    X_train[:, col:col+fdim] = tr[:n_tr]
    del tr
    
    te = np.load(os.path.join(CACHE_DIR, f"te_feat_{mi}.npy"))
    X_test[:, col:col+fdim] = te[:n_te]
    del te
    
    col += fdim
    gc.collect()

y_train = tr_lab
y_test  = te_lab

print(f"  Train: {X_train.shape} ({X_train.nbytes/1e6:.0f} MB)")
print(f"  Test:  {X_test.shape} ({X_test.nbytes/1e6:.0f} MB)")

# ══════════════════════════════════════════════════════════════════════
# Phase 4: Train MLP ensemble
# ══════════════════════════════════════════════════════════════════════
print("\nPhase 4: Training MLP ensemble")
print("-" * 50)

class FeatDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

configs = [
    (42,   1024, 1e-3,  0.45, 0.05),
    (123,  1280, 8e-4,  0.40, 0.04),
    (7,    768,  1.2e-3, 0.50, 0.06),
    (999,  1536, 5e-4,  0.35, 0.03),
    (2024, 1024, 7e-4,  0.42, 0.05),
]

mlp_models = []
for ci, (seed, width, lr, drop, noise) in enumerate(configs):
    print(f"\n  [{ci+1}/{len(configs)}] seed={seed} w={width} lr={lr} drop={drop}")
    t_m = time.time()
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    
    model = ResidualMLP(total_dim, n_classes, width, drop)
    
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
    
    cc = Counter(y_train.tolist())
    w = [1.0 / cc[int(l)] for l in y_train]
    sam = WeightedRandomSampler(w, len(w), replacement=True)
    tr_dl = DataLoader(FeatDS(X_train, y_train), batch_size=BATCH_TRAIN, sampler=sam, num_workers=0)
    te_dl = DataLoader(FeatDS(X_test, y_test), batch_size=512, shuffle=False, num_workers=0)
    
    best_a = 0.0; best_s = None; no_imp = 0
    for ep in range(1, EPOCHS_MLP + 1):
        model.train()
        for xb, yb in tr_dl:
            lam = np.random.beta(0.4, 0.4)
            perm = torch.randperm(xb.size(0))
            mx = lam * xb + (1 - lam) * xb[perm]
            mx = mx + torch.randn_like(mx) * noise
            opt.zero_grad()
            out = model(mx)
            loss = lam * crit(out, yb) + (1 - lam) * crit(out, yb[perm])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        
        model.eval()
        tc = tt = 0
        with torch.no_grad():
            for xb, yb in te_dl:
                tc += (model(xb).argmax(1) == yb).sum().item()
                tt += xb.size(0)
        acc = tc / tt * 100
        
        if ep % 20 == 0 or ep <= 3:
            print(f"      ep {ep:>3d}  acc={acc:.2f}%  best={best_a:.2f}%")
        
        if acc > best_a:
            best_a = acc; best_s = {k: v.clone() for k, v in model.state_dict().items()}; no_imp = 0
        else:
            no_imp += 1
        if no_imp >= PATIENCE_MLP:
            print(f"      early stop ep {ep}")
            break
    
    if best_s: model.load_state_dict(best_s)
    model.eval()
    mlp_models.append((model, width))
    print(f"    Best: {best_a:.2f}% ({time.time()-t_m:.0f}s)")
    del tr_dl, te_dl, crit, opt, sched, sam, w
    gc.collect()

# ══════════════════════════════════════════════════════════════════════
# Phase 5: Ensemble evaluation
# ══════════════════════════════════════════════════════════════════════
print("\nPhase 5: Ensemble evaluation")
print("-" * 50)

te_dl = DataLoader(FeatDS(X_test, y_test), batch_size=512, shuffle=False)
all_probs = []; all_true = []
with torch.no_grad():
    for xb, yb in te_dl:
        p = torch.zeros(xb.size(0), n_classes)
        for m, _ in mlp_models:
            p += torch.softmax(m(xb), dim=1)
        p /= len(mlp_models)
        all_probs.append(p); all_true.append(yb)

all_probs = torch.cat(all_probs); all_true = torch.cat(all_true)
preds = all_probs.argmax(1)
ens_acc = (preds == all_true).float().mean().item() * 100
print(f"\nEnsemble accuracy: {ens_acc:.2f}%")
print(classification_report(all_true.numpy(), preds.numpy(), target_names=all_classes))

# ══════════════════════════════════════════════════════════════════════
# Phase 6: Save
# ══════════════════════════════════════════════════════════════════════
print("\nPhase 6: Saving")
print("-" * 50)

import joblib

states = [{k: v.cpu() for k, v in m.state_dict().items()} for m, _ in mlp_models]
widths = [w for _, w in mlp_models]
bnames = [n for n, _ in model_info]
bdims  = [d for _, d in model_info]

wrapper = MultiModelEnsemble(states, widths, all_classes, total_dim, bnames, bdims)
mp = os.path.join(BASE_DIR, "image_model.pkl")
joblib.dump(wrapper, mp, compress=3)
print(f"  Model: {mp} ({os.path.getsize(mp)/1e6:.1f} MB)")

cp = os.path.join(BASE_DIR, "image_classes.pkl")
with open(cp, "wb") as f:
    pickle.dump(all_classes, f)
print(f"  Classes: {cp}")

# Cleanup
import shutil
try:
    shutil.rmtree(CACHE_DIR)
    print("  Cleaned cache")
except Exception:
    pass

total_time = time.time() - t0
print(f"\n{'=' * 70}")
print(f"  FINAL ACCURACY: {ens_acc:.2f}%")
print(f"  Time: {total_time/60:.1f} min")
print(f"  Backbones: {', '.join(bnames)}")
print(f"  Features: {total_dim} dims")
print(f"  Ensemble: {len(mlp_models)} MLPs")
print(f"{'=' * 70}")
