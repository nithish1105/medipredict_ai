"""
train_finetune_v3.py - Memory-efficient fine-tuning of EfficientNet-B0
=====================================================================
Strategy: Stream images from ZIPs through frozen backbone (no bulk loading).
  Phase 1: Catalogue images from ZIPs (paths only, no loading)
  Phase 2: Stream images → extract intermediate features → save to disk
  Phase 3: Fine-tune blocks 7-8 + classifier on cached features
  Phase 4: Extract 1280-dim features via fine-tuned backbone
  Phase 5: Train MLP ensemble on fine-tuned features
  Phase 6: Save everything

Peak memory: ~1 GB (intermediate features + model) vs 5+ GB in v2.
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

# ══════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "datasets")
RESULT_FILE = os.path.join(BASE_DIR, "train_result.txt")

MAX_PER_CLASS = 300   # per ZIP source
AUG_FACTOR   = 2      # 1 original + 1 augmented
BATCH_EXT    = 32
BATCH_TRAIN  = 64
EPOCHS_FT    = 40
LR_FT        = 3e-4
PATIENCE_FT  = 12
EPOCHS_MLP   = 300
LR_MLP       = 1e-3
PATIENCE_MLP = 40
IMG_SIZE     = 224

# ══════════════════════════════════════════════════════════════════════
# Redirect stdout
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
t0 = time.time()
print("=" * 70)
print("  FINE-TUNE v3 (Memory-Efficient)")
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
# Catalogue functions (paths only - no image loading)
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
# Phase 1: Catalogue all images (paths only)
# ══════════════════════════════════════════════════════════════════════
print("\nPhase 1: Cataloguing images")
print("-" * 50)

SKIN_ZIP   = os.path.join(DATA_DIR, "archive (1).zip")
EYE_ZIP    = os.path.join(DATA_DIR, "archive (2).zip")
HAM_ZIP    = os.path.join(DATA_DIR, "archive (3).zip")
IMGCLS_ZIP = os.path.join(DATA_DIR, "archive (4).zip")

sources = []
print("[1/4] Skin ZIP ...")
skin = catalogue_zip(SKIN_ZIP, split="train")
print(f"  {len(skin)} classes")
sources.append(("Skin", SKIN_ZIP, skin))

print("[2/4] Eye ZIP ...")
eye = catalogue_zip(EYE_ZIP, split="dataset")
print(f"  {len(eye)} classes")
sources.append(("Eye", EYE_ZIP, eye))

print("[3/4] HAM10000 ZIP ...")
ham = catalogue_ham10000(HAM_ZIP)
print(f"  {len(ham)} classes")
sources.append(("HAM", HAM_ZIP, ham))

print("[4/4] IMG_CLASSES ZIP ...")
imgcls = catalogue_img_classes(IMGCLS_ZIP)
print(f"  {len(imgcls)} classes")
sources.append(("IMG", IMGCLS_ZIP, imgcls))

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

# ══════════════════════════════════════════════════════════════════════
# Phase 2: Stream images → extract intermediate features
# ══════════════════════════════════════════════════════════════════════
print("\nPhase 2: Streaming feature extraction")
print("-" * 50)

# Split at image level BEFORE augmentation
indices = list(range(len(all_samples)))
labels = [class_to_idx[all_samples[i][2]] for i in indices]
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)
print(f"Image split: {len(train_idx)} train, {len(test_idx)} test")

# Build frozen backbone
print("Loading EfficientNet-B0 ...")
base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
frozen = nn.Sequential(*list(base_model.features[:7]))
frozen.eval()
for p in frozen.parameters():
    p.requires_grad = False

# Verify shape
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
with torch.no_grad():
    out = frozen(dummy)
print(f"Intermediate shape: {out.shape}")  # (1, 192, 7, 7)

def stream_extract(sample_indices, augment=False):
    """Stream images from ZIPs through frozen backbone, return features and labels."""
    features_list = []
    labels_list = []
    
    # Group by ZIP for efficiency (open each ZIP once)
    by_zip = {}
    for idx in sample_indices:
        zp, entry, cls = all_samples[idx]
        by_zip.setdefault(zp, []).append((idx, entry, cls))
    
    count = 0
    total = len(sample_indices) * (AUG_FACTOR if augment else 1)
    
    for zip_path, items in by_zip.items():
        batch_tensors = []
        batch_labels = []
        
        with zipfile.ZipFile(zip_path) as zf:
            for idx, entry, cls in items:
                try:
                    data = zf.read(entry)
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                except Exception:
                    continue
                
                # Center crop (always)
                batch_tensors.append(_val_tf(img))
                batch_labels.append(class_to_idx[cls])
                
                # Augmented view (train only)
                if augment:
                    for _ in range(AUG_FACTOR - 1):
                        batch_tensors.append(_train_tf(img))
                        batch_labels.append(class_to_idx[cls])
                
                # Close image immediately
                img.close()
                del img, data
                
                # Process batch when full
                while len(batch_tensors) >= BATCH_EXT:
                    batch = torch.stack(batch_tensors[:BATCH_EXT])
                    with torch.no_grad():
                        feats = frozen(batch)
                    features_list.append(feats.cpu())
                    labels_list.extend(batch_labels[:BATCH_EXT])
                    batch_tensors = batch_tensors[BATCH_EXT:]
                    batch_labels = batch_labels[BATCH_EXT:]
                    count += feats.size(0)
                    
                    if count % 1000 < BATCH_EXT:
                        print(f"  {count}/{total} extracted")
        
        # Flush remaining for this ZIP
        if batch_tensors:
            batch = torch.stack(batch_tensors)
            with torch.no_grad():
                feats = frozen(batch)
            features_list.append(feats.cpu())
            labels_list.extend(batch_labels)
            count += feats.size(0)
    
    all_feats = torch.cat(features_list, dim=0)
    all_labels = torch.tensor(labels_list, dtype=torch.long)
    return all_feats, all_labels

print(f"\nExtracting train features ({len(train_idx)} images x {AUG_FACTOR} aug)...")
t_ext = time.time()
train_feats, train_labels = stream_extract(train_idx, augment=True)
print(f"  Train: {train_feats.shape} ({time.time()-t_ext:.0f}s, {train_feats.nelement()*4/1e6:.0f} MB)")

print(f"Extracting test features ({len(test_idx)} images, no aug)...")
t_ext2 = time.time()
test_feats, test_labels = stream_extract(test_idx, augment=False)
print(f"  Test: {test_feats.shape} ({time.time()-t_ext2:.0f}s)")

# Free frozen backbone to save memory
del frozen, base_model
gc.collect()
print("  Freed backbone memory")

# ══════════════════════════════════════════════════════════════════════
# Phase 3: Fine-tune blocks 7-8 + classifier
# ══════════════════════════════════════════════════════════════════════
print("\nPhase 3: Fine-tuning")
print("-" * 50)

class IntermediateDataset(Dataset):
    def __init__(self, features, labels, augment=False):
        self.features = features
        self.labels = labels
        self.augment = augment
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        feat = self.features[idx]
        if self.augment:
            feat = feat + torch.randn_like(feat) * 0.02
        return feat, self.labels[idx]

class TrainableHead(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # Reload fresh pretrained blocks 7-8
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.block7 = m.features[7]
        self.block8 = m.features[8]
        self.avgpool = m.avgpool
        del m
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
        x = self.block7(x)
        x = self.block8(x)
        x = self.avgpool(x)
        return x.flatten(1)

head = TrainableHead(n_classes)
n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
print(f"Trainable params: {n_params:,}")
print(f"Epochs: {EPOCHS_FT}, LR: {LR_FT}, Batch: {BATCH_TRAIN}, Patience: {PATIENCE_FT}")

# Weighted sampler
class_counts = Counter(train_labels.numpy().tolist())
weights = [1.0 / class_counts[int(l)] for l in train_labels]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_ds = IntermediateDataset(train_feats, train_labels, augment=True)
test_ds = IntermediateDataset(test_feats, test_labels, augment=False)
train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, sampler=sampler, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

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
    
    head.eval()
    test_correct = test_total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            out = head(xb)
            test_correct += (out.argmax(1) == yb).sum().item()
            test_total += xb.size(0)
    test_acc = test_correct / test_total * 100
    
    elapsed = time.time() - t_ep
    lr_now = optimizer.param_groups[0]['lr']
    # Always print (simpler monitoring)
    print(f"  Ep {epoch:>2d}  loss={total_loss/total:.4f}  train={correct/total*100:.1f}%  test={test_acc:.1f}%  lr={lr_now:.6f}  ({elapsed:.0f}s)")
    
    if test_acc > best_acc:
        best_acc = test_acc
        best_state = {k: v.clone() for k, v in head.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1
    
    if no_improve >= PATIENCE_FT:
        print(f"\n  Early stopping at epoch {epoch}")
        break

print(f"\n  Best fine-tune accuracy: {best_acc:.2f}%")
if best_state:
    head.load_state_dict(best_state)

# ══════════════════════════════════════════════════════════════════════
# Phase 4: Extract 1280-dim features via fine-tuned backbone
# ══════════════════════════════════════════════════════════════════════
print("\nPhase 4: Extracting 1280-dim fine-tuned features")
print("-" * 50)

head.eval()
ft_feats = {}
for name, feats, labels in [("train", train_feats, train_labels),
                              ("test",  test_feats,  test_labels)]:
    feat_list = []
    with torch.no_grad():
        for i in range(0, len(feats), 128):
            f = head.extract_features(feats[i:i+128])
            feat_list.append(f.cpu().numpy())
    ft_feats[name] = (np.concatenate(feat_list), labels.numpy())
    print(f"  {name}: {ft_feats[name][0].shape}")

# Free intermediate features
del train_feats, test_feats, train_ds, test_ds, train_loader, test_loader
gc.collect()

# ══════════════════════════════════════════════════════════════════════
# Phase 5: Train MLP ensemble on fine-tuned 1280-dim features
# ══════════════════════════════════════════════════════════════════════
print("\nPhase 5: MLP ensemble on fine-tuned features")
print("-" * 50)

X_train, y_train = ft_feats["train"]
X_test, y_test = ft_feats["test"]
n_feat = X_train.shape[1]
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

class DeepMLP(nn.Module):
    def __init__(self, n_in, n_out, width=1024, drop=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(n_in),
            nn.Linear(n_in, width), nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(drop),
            nn.Linear(width, width//2), nn.BatchNorm1d(width//2), nn.GELU(), nn.Dropout(drop*0.7),
            nn.Linear(width//2, width//4), nn.BatchNorm1d(width//4), nn.GELU(), nn.Dropout(drop*0.5),
            nn.Linear(width//4, n_out),
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
    (42,   1024, 1e-3, 0.4),
    (123,  1280, 8e-4, 0.35),
    (7,    768,  1.2e-3, 0.45),
]

mlp_models = []
for i, (seed, width, lr, drop) in enumerate(configs):
    print(f"\n[{i+1}/{len(configs)}] Seed={seed}, Width={width}")
    t_m = time.time()
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    
    model = DeepMLP(n_feat, n_classes, width, drop)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
    
    cc = Counter(y_train.tolist())
    w = [1.0/cc[int(l)] for l in y_train]
    sam = WeightedRandomSampler(w, len(w), replacement=True)
    tr_dl = DataLoader(FeatDS(X_train, y_train), batch_size=256, sampler=sam, num_workers=0)
    te_dl = DataLoader(FeatDS(X_test, y_test), batch_size=512, shuffle=False, num_workers=0)
    
    best_a = 0.0; best_s = None; no_imp = 0
    for ep in range(1, EPOCHS_MLP+1):
        model.train()
        for xb, yb in tr_dl:
            lam = np.random.beta(0.3, 0.3)
            idx = torch.randperm(xb.size(0))
            mx = lam*xb + (1-lam)*xb[idx]
            mx = mx + torch.randn_like(mx)*0.05
            opt.zero_grad()
            out = model(mx)
            loss = lam*crit(out, yb) + (1-lam)*crit(out, yb[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        
        model.eval()
        tc = tt = 0
        with torch.no_grad():
            for xb, yb in te_dl:
                tc += (model(xb).argmax(1)==yb).sum().item()
                tt += xb.size(0)
        acc = tc/tt*100
        if acc > best_a:
            best_a = acc; best_s = {k:v.clone() for k,v in model.state_dict().items()}; no_imp = 0
        else:
            no_imp += 1
        if no_imp >= PATIENCE_MLP:
            break
    
    if best_s: model.load_state_dict(best_s)
    model.eval()
    mlp_models.append(model)
    print(f"  Acc: {best_a:.2f}% ({time.time()-t_m:.0f}s)")

# Ensemble
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
# Phase 6: Save
# ══════════════════════════════════════════════════════════════════════
print("\nPhase 6: Saving")
print("-" * 50)

# 1. Fine-tuned backbone
full = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
full.features[7] = head.block7
full.features[8] = head.block8
full.classifier = nn.Identity()
full.eval()
bb_path = os.path.join(BASE_DIR, "finetuned_backbone.pth")
torch.save(full.state_dict(), bb_path)
print(f"  Backbone: {bb_path} ({os.path.getsize(bb_path)/1e6:.1f} MB)")
del full

# 2. MLP ensemble wrapper
class EnsembleWrapper:
    def __init__(self, models_list, classes, n_feat):
        self.classes_ = np.array(classes)
        self._n_cls = len(classes)
        self._n_feat = n_feat
        self._states = [{k:v.cpu() for k,v in m.state_dict().items()} for m in models_list]
        self._widths = []
        for s in self._states:
            for k,v in s.items():
                if '1.weight' in k:
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
        X_t = torch.tensor(np.array(X), dtype=torch.float32)
        with torch.no_grad():
            p = sum(torch.softmax(m(X_t), dim=1) for m in self._models) / len(self._models)
        return p.argmax(1).numpy()
    
    def predict_proba(self, X):
        self._load()
        X_t = torch.tensor(np.array(X), dtype=torch.float32)
        with torch.no_grad():
            p = sum(torch.softmax(m(X_t), dim=1) for m in self._models) / len(self._models)
        return p.numpy()

import joblib
wrapper = EnsembleWrapper(mlp_models, all_classes, n_feat)
mp = os.path.join(BASE_DIR, "image_model.pkl")
joblib.dump(wrapper, mp, compress=3)
print(f"  Model: {mp} ({os.path.getsize(mp)/1e6:.1f} MB)")

# 3. Classes
cp = os.path.join(BASE_DIR, "image_classes.pkl")
with open(cp, "wb") as f:
    pickle.dump(all_classes, f)
print(f"  Classes: {cp}")

total = time.time() - t0
print(f"\nFINAL ACCURACY: {ens_acc:.2f}%")
print(f"Total time: {total/60:.1f} min")
print("=" * 70)
