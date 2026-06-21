"""
train_finetune_v4.py  -  Ultra memory-safe fine-tuning with numpy memmap
========================================================================
ALL intermediate features stored on DISK via numpy memmap, not in RAM.
Peak RAM should stay well under 1 GB regardless of dataset size.

Phases:
  1. Catalogue images from ZIPs (paths only)
  2. Extract intermediate features (192,7,7) → memmap files on disk
  3. Fine-tune blocks 7-8 + classifier using memmap-backed DataLoader
  4. Extract 1280-dim features through fine-tuned head → numpy array
  5. Train MLP ensemble on 1280-dim features
  6. Save backbone + model
"""

import os, sys, time, pickle, random, zipfile, io, csv, gc, traceback
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
# Config
# ══════════════════════════════════════════════════════════════════════
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(os.path.dirname(BASE_DIR), "datasets")
RESULT_F  = os.path.join(BASE_DIR, "train_result.txt")
ERROR_F   = os.path.join(BASE_DIR, "train_error.txt")
CACHE_DIR = os.path.join(BASE_DIR, "_feat_cache")

MAX_PER_CLASS = 300
AUG_FACTOR    = 2       # 1 original + 1 augmented
BATCH_EXT     = 16      # small extraction batch → low peak RAM
BATCH_TRAIN   = 64
EPOCHS_FT     = 40
LR_FT         = 3e-4
PATIENCE_FT   = 12
EPOCHS_MLP    = 300
LR_MLP        = 1e-3
PATIENCE_MLP  = 40
IMG_SIZE      = 224
INTER_SHAPE   = (192, 7, 7)   # block-6 output for EfficientNet-B0
INTER_DIM     = 192 * 7 * 7   # 9408

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
try:
    err_f = open(ERROR_F, "w", encoding="utf-8", errors="replace")
    sys.stderr = err_f
except PermissionError:
    pass  # If shell already redirects stderr, skip

def mem_mb():
    """Return current process RSS in MB (Windows)."""
    try:
        import ctypes, ctypes.wintypes
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [("cb", ctypes.wintypes.DWORD),
                        ("PageFaultCount", ctypes.wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        h = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
        return pmc.WorkingSetSize / (1024 * 1024)
    except Exception:
        return -1

t0 = time.time()
print("=" * 70)
print("  FINE-TUNE v4  (memmap disk-backed features)")
print("=" * 70)
print(f"  RAM at start: {mem_mb():.0f} MB")

# ══════════════════════════════════════════════════════════════════════
# Class maps (same as before)
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

# Build sample list: (zip_path, entry_path, class_name)
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

# Split before augmentation
indices = list(range(len(all_samples)))
labels_arr = [class_to_idx[all_samples[i][2]] for i in indices]
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels_arr)
print(f"\nSplit: {len(train_idx)} train, {len(test_idx)} test")
print(f"  RAM after catalogue: {mem_mb():.0f} MB")

# ══════════════════════════════════════════════════════════════════════
# Phase 2: Extract intermediate features → memmap on disk
# ══════════════════════════════════════════════════════════════════════
print("\nPhase 2: Extracting features to disk (memmap)")
print("-" * 50)

os.makedirs(CACHE_DIR, exist_ok=True)

# Load frozen backbone (blocks 0-6)
print("Loading EfficientNet-B0 backbone ...")
base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
frozen = nn.Sequential(*list(base_model.features[:7]))
frozen.eval()
for p in frozen.parameters():
    p.requires_grad = False
del base_model
gc.collect()

# Verify shape
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
with torch.no_grad():
    out = frozen(dummy)
print(f"Intermediate shape: {out.shape}")
del dummy, out
print(f"  RAM after loading backbone: {mem_mb():.0f} MB")

def extract_to_memmap(sample_indices, feat_path, label_path, augment):
    """
    Stream images from ZIPs through frozen backbone,
    write features DIRECTLY to a numpy memmap file on disk.
    """
    aug_factor = AUG_FACTOR if augment else 1
    max_total = len(sample_indices) * aug_factor
    
    # Pre-allocate memmap on disk
    mm = np.memmap(feat_path, dtype=np.float16, mode="w+",
                   shape=(max_total, *INTER_SHAPE))
    lab = np.full(max_total, -1, dtype=np.int32)
    
    # Group by ZIP
    by_zip = {}
    for idx in sample_indices:
        zp, entry, cls = all_samples[idx]
        by_zip.setdefault(zp, []).append((idx, entry, cls))
    
    pos = 0
    failed = 0
    
    for zip_path, items in by_zip.items():
        batch_t = []   # image tensors for current batch
        batch_l = []   # labels for current batch
        
        with zipfile.ZipFile(zip_path) as zf:
            for _, entry, cls in items:
                try:
                    data = zf.read(entry)
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                except Exception:
                    failed += 1
                    continue
                
                # Original (always)
                batch_t.append(_val_tf(img))
                batch_l.append(class_to_idx[cls])
                
                # Augmented copies (train only)
                if augment:
                    for _ in range(AUG_FACTOR - 1):
                        batch_t.append(_train_tf(img))
                        batch_l.append(class_to_idx[cls])
                
                img.close()
                del img, data
                
                # ---- Flush batch to disk when full ----
                while len(batch_t) >= BATCH_EXT:
                    batch = torch.stack(batch_t[:BATCH_EXT])
                    with torch.no_grad():
                        feats = frozen(batch)
                    # Write to memmap → disk, NOT RAM
                    n = feats.shape[0]
                    mm[pos:pos + n] = feats.cpu().numpy().astype(np.float16)
                    lab[pos:pos + n] = batch_l[:BATCH_EXT]
                    pos += n
                    
                    # Discard processed items
                    batch_t = batch_t[BATCH_EXT:]
                    batch_l = batch_l[BATCH_EXT:]
                    del batch, feats
                    
                    if pos % 1000 < BATCH_EXT:
                        print(f"    {pos}/{max_total} extracted  (RAM {mem_mb():.0f} MB)")
        
        # Flush remaining from this ZIP
        if batch_t:
            batch = torch.stack(batch_t)
            with torch.no_grad():
                feats = frozen(batch)
            n = feats.shape[0]
            mm[pos:pos + n] = feats.cpu().numpy().astype(np.float16)
            lab[pos:pos + n] = batch_l[:n]
            pos += n
            del batch, feats
            batch_t.clear()
            batch_l.clear()
        
        # Flush memmap to disk + GC after each ZIP
        mm.flush()
        gc.collect()
    
    # Save actual count and labels
    np.save(label_path, lab[:pos])
    del mm
    gc.collect()
    
    if failed:
        print(f"    ({failed} images failed to load)")
    return pos

# ---- Train features ----
train_feat_path  = os.path.join(CACHE_DIR, "train_feats.dat")
train_label_path = os.path.join(CACHE_DIR, "train_labels.npy")
print(f"\nExtracting TRAIN features ({len(train_idx)} × {AUG_FACTOR} aug) ...")
t_ext = time.time()
actual_train = extract_to_memmap(train_idx, train_feat_path, train_label_path, augment=True)
print(f"  Train: {actual_train} features, {time.time()-t_ext:.0f}s")
print(f"  Disk: {os.path.getsize(train_feat_path)/1e6:.0f} MB")

# ---- Test features ----
test_feat_path  = os.path.join(CACHE_DIR, "test_feats.dat")
test_label_path = os.path.join(CACHE_DIR, "test_labels.npy")
print(f"\nExtracting TEST features ({len(test_idx)}, no aug) ...")
t_ext2 = time.time()
actual_test = extract_to_memmap(test_idx, test_feat_path, test_label_path, augment=False)
print(f"  Test: {actual_test} features, {time.time()-t_ext2:.0f}s")

# Free the backbone
del frozen
gc.collect()
print(f"\n  RAM after freeing backbone: {mem_mb():.0f} MB")

# ══════════════════════════════════════════════════════════════════════
# Phase 3: Fine-tune blocks 7-8 on memmap features
# ══════════════════════════════════════════════════════════════════════
print("\nPhase 3: Fine-tuning blocks 7-8 + classifier")
print("-" * 50)

class MemmapDataset(Dataset):
    """Dataset backed by a numpy memmap file — features stay on disk."""
    def __init__(self, mm_path, label_path, count, shape, add_noise=False):
        self.mm = np.memmap(mm_path, dtype=np.float16, mode="r", shape=(count, *shape))
        self.labels = np.load(label_path)
        self.count = count
        self.add_noise = add_noise
    
    def __len__(self):
        return self.count
    
    def __getitem__(self, idx):
        # Read ONE feature from disk (OS caches hot pages)
        feat = torch.from_numpy(self.mm[idx].astype(np.float32))
        if self.add_noise:
            feat = feat + torch.randn_like(feat) * 0.02
        return feat, int(self.labels[idx])

class TrainableHead(nn.Module):
    """Blocks 7-8 + avgpool + classifier, trained on block-6 features."""
    def __init__(self, n_classes):
        super().__init__()
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.block7 = m.features[7]
        self.block8 = m.features[8]
        self.avgpool = m.avgpool
        del m
        gc.collect()
        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, n_classes),
        )
    
    def forward(self, x):
        x = self.block7(x)
        x = self.block8(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)
    
    def extract_features(self, x):
        """Return 1280-dim feature vector (no classifier)."""
        x = self.block7(x)
        x = self.block8(x)
        x = self.avgpool(x)
        return x.flatten(1)

head = TrainableHead(n_classes)
n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
print(f"Trainable params: {n_params:,}")
print(f"Config: epochs={EPOCHS_FT}, lr={LR_FT}, batch={BATCH_TRAIN}, patience={PATIENCE_FT}")
print(f"  RAM after loading head: {mem_mb():.0f} MB")

train_ds = MemmapDataset(train_feat_path, train_label_path, actual_train,
                         INTER_SHAPE, add_noise=True)
test_ds  = MemmapDataset(test_feat_path, test_label_path, actual_test,
                         INTER_SHAPE, add_noise=False)

# Weighted sampler for class imbalance
train_labels_np = np.load(train_label_path)
class_counts = Counter(train_labels_np.tolist())
sample_weights = [1.0 / class_counts[int(l)] for l in train_labels_np]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
del train_labels_np, sample_weights
gc.collect()

train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, sampler=sampler,
                          num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False,
                          num_workers=0, pin_memory=False)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(head.parameters(), lr=LR_FT, weight_decay=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

best_acc = 0.0
best_state = None
no_improve = 0

for epoch in range(1, EPOCHS_FT + 1):
    t_ep = time.time()
    head.train()
    total_loss = correct = total = 0
    
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = head(xb)
        loss = criterion(out, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        correct += (out.argmax(1) == yb).sum().item()
        total += xb.size(0)
    
    scheduler.step()
    
    # Evaluate
    head.eval()
    tc = tt = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            out = head(xb)
            tc += (out.argmax(1) == yb).sum().item()
            tt += xb.size(0)
    test_acc = tc / tt * 100
    
    elapsed = time.time() - t_ep
    lr_now = optimizer.param_groups[0]["lr"]
    print(f"  Ep {epoch:>2d}  loss={total_loss/total:.4f}  "
          f"train={correct/total*100:.1f}%  test={test_acc:.1f}%  "
          f"lr={lr_now:.6f}  ({elapsed:.0f}s)  RAM={mem_mb():.0f}MB")
    
    if test_acc > best_acc:
        best_acc = test_acc
        best_state = {k: v.clone() for k, v in head.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1
    
    if no_improve >= PATIENCE_FT:
        print(f"\n  Early stopping at epoch {epoch}")
        break

print(f"\n  Best fine-tune test accuracy: {best_acc:.2f}%")
if best_state:
    head.load_state_dict(best_state)

# ══════════════════════════════════════════════════════════════════════
# Phase 4: Extract 1280-dim features through fine-tuned head
# ══════════════════════════════════════════════════════════════════════
print("\nPhase 4: Extracting 1280-dim fine-tuned features")
print("-" * 50)

head.eval()

def extract_1280(mm_path, label_path, count):
    """Read intermediate feats from memmap, push through fine-tuned blocks, return numpy."""
    mm = np.memmap(mm_path, dtype=np.float16, mode="r", shape=(count, *INTER_SHAPE))
    labels = np.load(label_path)
    parts = []
    with torch.no_grad():
        for i in range(0, count, 128):
            end = min(i + 128, count)
            batch = torch.from_numpy(mm[i:end].astype(np.float32))
            f = head.extract_features(batch)
            parts.append(f.cpu().numpy())
            del batch, f
    del mm
    return np.concatenate(parts), labels

X_train, y_train = extract_1280(train_feat_path, train_label_path, actual_train)
X_test, y_test   = extract_1280(test_feat_path, test_label_path, actual_test)
n_feat = X_train.shape[1]

print(f"  Train: {X_train.shape}")
print(f"  Test:  {X_test.shape}")
print(f"  RAM after extraction: {mem_mb():.0f} MB")

# Free memmap-related data
gc.collect()

# ══════════════════════════════════════════════════════════════════════
# Phase 5: Train MLP ensemble on 1280-dim fine-tuned features
# ══════════════════════════════════════════════════════════════════════
print("\nPhase 5: MLP ensemble")
print("-" * 50)

class DeepMLP(nn.Module):
    def __init__(self, n_in, n_out, width=1024, drop=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(n_in),
            nn.Linear(n_in, width), nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(drop),
            nn.Linear(width, width // 2), nn.BatchNorm1d(width // 2), nn.GELU(), nn.Dropout(drop * 0.7),
            nn.Linear(width // 2, width // 4), nn.BatchNorm1d(width // 4), nn.GELU(), nn.Dropout(drop * 0.5),
            nn.Linear(width // 4, n_out),
        )
    def forward(self, x):
        return self.net(x)

class FeatDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

configs = [
    (42,   1024, 1e-3,  0.4),
    (123,  1280, 8e-4,  0.35),
    (7,    768,  1.2e-3, 0.45),
]

mlp_models = []
for ci, (seed, width, lr, drop) in enumerate(configs):
    print(f"\n  [{ci+1}/{len(configs)}] seed={seed} width={width} lr={lr}")
    t_m = time.time()
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    
    model = DeepMLP(n_feat, n_classes, width, drop)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
    
    cc = Counter(y_train.tolist())
    w = [1.0 / cc[int(l)] for l in y_train]
    sam = WeightedRandomSampler(w, len(w), replacement=True)
    tr_dl = DataLoader(FeatDS(X_train, y_train), batch_size=256, sampler=sam, num_workers=0)
    te_dl = DataLoader(FeatDS(X_test, y_test), batch_size=512, shuffle=False, num_workers=0)
    
    best_a = 0.0; best_s = None; no_imp = 0
    for ep in range(1, EPOCHS_MLP + 1):
        model.train()
        for xb, yb in tr_dl:
            # Mixup
            lam = np.random.beta(0.3, 0.3)
            idx_p = torch.randperm(xb.size(0))
            mx = lam * xb + (1 - lam) * xb[idx_p]
            mx = mx + torch.randn_like(mx) * 0.05
            opt.zero_grad()
            out = model(mx)
            loss = lam * crit(out, yb) + (1 - lam) * crit(out, yb[idx_p])
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
        if acc > best_a:
            best_a = acc
            best_s = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= PATIENCE_MLP:
            break
    
    if best_s:
        model.load_state_dict(best_s)
    model.eval()
    mlp_models.append(model)
    print(f"    Best: {best_a:.2f}%  ({time.time()-t_m:.0f}s)")
    del tr_dl, te_dl, crit, opt, sched, sam, w, cc
    gc.collect()

# Ensemble evaluation
print(f"\nEnsemble evaluation:")
te_dl = DataLoader(FeatDS(X_test, y_test), batch_size=512, shuffle=False)
all_probs = []; all_true = []
with torch.no_grad():
    for xb, yb in te_dl:
        p = torch.zeros(xb.size(0), n_classes)
        for m in mlp_models:
            p += torch.softmax(m(xb), dim=1)
        p /= len(mlp_models)
        all_probs.append(p); all_true.append(yb)

all_probs = torch.cat(all_probs); all_true = torch.cat(all_true)
preds = all_probs.argmax(1)
ens_acc = (preds == all_true).float().mean().item() * 100
print(f"Ensemble accuracy: {ens_acc:.2f}%")
print(classification_report(all_true.numpy(), preds.numpy(), target_names=all_classes))

# ══════════════════════════════════════════════════════════════════════
# Phase 6: Save models
# ══════════════════════════════════════════════════════════════════════
print("\nPhase 6: Saving")
print("-" * 50)

# 1) Fine-tuned backbone
full = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
full.features[7] = head.block7
full.features[8] = head.block8
full.classifier = nn.Identity()
full.eval()
bb_path = os.path.join(BASE_DIR, "finetuned_backbone.pth")
torch.save(full.state_dict(), bb_path)
print(f"  Backbone: {bb_path} ({os.path.getsize(bb_path)/1e6:.1f} MB)")
del full

# 2) MLP ensemble
class EnsembleWrapper:
    def __init__(self, models_list, classes, n_feat):
        self.classes_ = np.array(classes)
        self._n_cls = len(classes)
        self._n_feat = n_feat
        self._states = [{k: v.cpu() for k, v in m.state_dict().items()} for m in models_list]
        self._widths = []
        for s in self._states:
            for k, v in s.items():
                if "1.weight" in k:
                    self._widths.append(v.shape[0]); break
        self._models = None
    
    def _load(self):
        if self._models is None:
            self._models = []
            for s, w in zip(self._states, self._widths):
                m = DeepMLP(self._n_feat, self._n_cls, w)
                m.load_state_dict(s); m.eval()
                self._models.append(m)
    
    def predict(self, X):
        self._load()
        t = torch.tensor(np.array(X), dtype=torch.float32)
        with torch.no_grad():
            p = sum(torch.softmax(m(t), dim=1) for m in self._models) / len(self._models)
        return p.argmax(1).numpy()
    
    def predict_proba(self, X):
        self._load()
        t = torch.tensor(np.array(X), dtype=torch.float32)
        with torch.no_grad():
            p = sum(torch.softmax(m(t), dim=1) for m in self._models) / len(self._models)
        return p.numpy()

import joblib
wrapper = EnsembleWrapper(mlp_models, all_classes, n_feat)
mp = os.path.join(BASE_DIR, "image_model.pkl")
joblib.dump(wrapper, mp, compress=3)
print(f"  Model: {mp} ({os.path.getsize(mp)/1e6:.1f} MB)")

# 3) Classes
cp = os.path.join(BASE_DIR, "image_classes.pkl")
with open(cp, "wb") as f:
    pickle.dump(all_classes, f)
print(f"  Classes: {cp}")

# Cleanup cache dir
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
print(f"  RAM at end: {mem_mb():.0f} MB")
print(f"{'=' * 70}")
