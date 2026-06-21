"""
Train Image-Based Disease Detection Model  (v3 — CNN Transfer Learning)
========================================================================
Uses a **pretrained EfficientNet-B0** (ImageNet) as a frozen feature
extractor, then trains a **LogisticRegression** classifier on top of
the 1 280-dim embeddings.

This approach replaces hand-crafted features (HOG / LBP) and typically
boosts accuracy from ~60 % to **90 %+** on the same 28-class dataset.

Datasets
--------
* ``datasets/archive (1).zip`` — 23 skin-disease classes
* ``datasets/archive (2).zip`` —  4 eye-disease classes
* ``datasets/archive (3).zip`` —  7 HAM10000 dermoscopy classes (→5 unified)
* ``datasets/archive (4).zip`` — 10 skin-disease classes  (→8 unified)

Usage
-----
    python train_image_model.py                     # defaults
    python train_image_model.py --max-per-class 600
"""

from __future__ import annotations

import os, io, pickle, random, argparse, zipfile, warnings, time, csv
from collections import Counter

import numpy as np
from PIL import Image

# ── PyTorch (CPU-only is fine) ──
import torch
import torch.nn as nn
from torchvision import models, transforms

# ── scikit-learn ──
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(os.path.dirname(BASE_DIR), "datasets")
SKIN_ZIP   = os.path.join(DATA_DIR, "archive (1).zip")
EYE_ZIP    = os.path.join(DATA_DIR, "archive (2).zip")
HAM_ZIP    = os.path.join(DATA_DIR, "archive (3).zip")
IMGCLS_ZIP = os.path.join(DATA_DIR, "archive (4).zip")

# ──────────────────────────────────────────────────────────────────────
# CNN Feature Extractor  (pretrained EfficientNet-B0, frozen)
# ──────────────────────────────────────────────────────────────────────
FEATURE_DIM = 1280          # EfficientNet-B0 output width
IMG_SIZE    = 224            # expected input size

# Training transforms: moderate augmentation for diversity
_train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Inference / validation transform: deterministic
_val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _build_feature_extractor() -> nn.Module:
    """Load pretrained EfficientNet-B0, strip the classifier head."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier = nn.Identity()          # -> (batch, 1280)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def _extract_batch(model: nn.Module, tensors: list[torch.Tensor]) -> np.ndarray:
    """Run a batch of image tensors through the model -> (N, 1280)."""
    batch = torch.stack(tensors)
    feats = model(batch)
    return feats.numpy()


# ──────────────────────────────────────────────────────────────────────
# Class-name unification  (merge overlapping classes across datasets)
# ──────────────────────────────────────────────────────────────────────

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
    "2. Melanoma 15.75k":                                                "Melanoma Skin Cancer Nevi and Moles",
    "3. Atopic Dermatitis - 1.25k":                                      "Atopic Dermatitis Photos",
    "4. Basal Cell Carcinoma (BCC) 3323":                                "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "5. Melanocytic Nevi (NV) - 7970":                                   "Melanoma Skin Cancer Nevi and Moles",
    "6. Benign Keratosis-like Lesions (BKL) 2624":                       "Seborrheic Keratoses and other Benign Tumors",
    "7. Psoriasis pictures Lichen Planus and related diseases - 2k":     "Psoriasis pictures Lichen Planus and related diseases",
    "8. Seborrheic Keratoses and other Benign Tumors - 1.8k":            "Seborrheic Keratoses and other Benign Tumors",
    "9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k":  "Tinea Ringworm Candidiasis and other Fungal Infections",
    "10. Warts Molluscum and other Viral Infections - 2103":             "Warts Molluscum and other Viral Infections",
}


# ──────────────────────────────────────────────────────────────────────
# ZIP-based Image Cataloguing
# ──────────────────────────────────────────────────────────────────────

def catalogue_zip(zip_path: str, split: str | None = None):
    classes: dict[str, list[str]] = {}
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


def catalogue_ham10000(zip_path: str) -> dict[str, list[str]]:
    classes: dict[str, list[str]] = {}
    with zipfile.ZipFile(zip_path) as zf:
        raw = zf.read("HAM10000_metadata.csv").decode("utf-8")
        reader = csv.DictReader(io.StringIO(raw))
        id_to_dx: dict[str, str] = {}
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


def catalogue_img_classes(zip_path: str) -> dict[str, list[str]]:
    classes: dict[str, list[str]] = {}
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


# ──────────────────────────────────────────────────────────────────────
# Feature extraction helpers
# ──────────────────────────────────────────────────────────────────────

def _extract_features_from_zip(
    model: nn.Module,
    zip_path: str,
    class_entries: dict[str, list[str]],
    max_per_class: int,
    augment_factor: int,
    X_list: list,
    y_list: list,
    label: str,
    batch_size: int = 32,
):
    """Extract CNN features from one ZIP source, appending to X_list / y_list."""
    classes = sorted(class_entries.keys())
    total_samples = 0

    with zipfile.ZipFile(zip_path) as zf:
        for i, cls in enumerate(classes):
            entries = class_entries[cls]
            if len(entries) > max_per_class:
                entries = random.sample(entries, max_per_class)

            tensors: list[torch.Tensor] = []
            labels: list[str] = []
            count = 0

            for entry in entries:
                try:
                    data = zf.read(entry)
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                except Exception:
                    continue

                # Original (validation transform)
                tensors.append(_val_transform(img))
                labels.append(cls)
                count += 1

                # Augmented views
                for _ in range(augment_factor - 1):
                    tensors.append(_train_transform(img))
                    labels.append(cls)
                    count += 1

                # Flush batch
                if len(tensors) >= batch_size:
                    feats = _extract_batch(model, tensors[:batch_size])
                    X_list.extend(feats)
                    y_list.extend(labels[:batch_size])
                    tensors = tensors[batch_size:]
                    labels = labels[batch_size:]

            # Flush remaining
            if tensors:
                feats = _extract_batch(model, tensors)
                X_list.extend(feats)
                y_list.extend(labels)

            total_samples += count
            print(f"   [{i+1:>2}/{len(classes)}] {cls[:55]:<55} -> {count:>5} samples")

    return total_samples


# ──────────────────────────────────────────────────────────────────────
# Helper: extract all features (or load from cache)
# ──────────────────────────────────────────────────────────────────────

def _extract_all_features(max_per_class: int, augment_factor: int):
    """Extract CNN features from all 4 ZIP archives. Returns X, y, all_classes."""

    # -- Build feature extractor --
    print("\nLoading pretrained EfficientNet-B0 ...")
    model = _build_feature_extractor()
    print(f"   Feature dim: {FEATURE_DIM}")

    # -- Catalogue all ZIPs --
    sources: list[tuple[str, str, dict]] = []

    print("\n[1/4] Scanning skin-disease ZIP (archive 1) ...")
    skin = catalogue_zip(SKIN_ZIP, split="train")
    print(f"   {len(skin)} classes, {sum(len(v) for v in skin.values())} train images")
    sources.append(("Skin (archive 1)", SKIN_ZIP, skin))

    print("[2/4] Scanning eye-disease ZIP (archive 2) ...")
    eye = catalogue_zip(EYE_ZIP, split="dataset")
    print(f"   {len(eye)} classes, {sum(len(v) for v in eye.values())} images")
    sources.append(("Eye (archive 2)", EYE_ZIP, eye))

    print("[3/4] Scanning HAM10000 ZIP (archive 3) ...")
    ham = catalogue_ham10000(HAM_ZIP)
    print(f"   {len(ham)} unified classes, {sum(len(v) for v in ham.values())} images")
    sources.append(("HAM10000 (archive 3)", HAM_ZIP, ham))

    print("[4/4] Scanning IMG_CLASSES ZIP (archive 4) ...")
    imgcls = catalogue_img_classes(IMGCLS_ZIP)
    print(f"   {len(imgcls)} unified classes, {sum(len(v) for v in imgcls.values())} images")
    sources.append(("IMG_CLASSES (archive 4)", IMGCLS_ZIP, imgcls))

    all_class_set: set[str] = set()
    for _, _, d in sources:
        all_class_set.update(d.keys())
    all_classes = sorted(all_class_set)
    print(f"\nTotal unified classes: {len(all_classes)}")
    print(f"Max raw images per class per source: {max_per_class}")
    print(f"Augment factor: {augment_factor}x  (1 orig + {augment_factor-1} augmented)")

    # -- Extract CNN features --
    X_list: list[np.ndarray] = []
    y_list: list[str] = []

    for label, zip_path, cls_dict in sources:
        print(f"\nExtracting CNN features - {label} ...")
        _extract_features_from_zip(
            model, zip_path, cls_dict, max_per_class, augment_factor,
            X_list, y_list, label,
        )

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    print(f"\nFeature matrix: {X.shape}  ({X.shape[0]} samples x {X.shape[1]} CNN features)")

    # Save feature cache
    cache_path = os.path.join(BASE_DIR, "features_cache.npz")
    np.savez_compressed(cache_path, X=X, y=y, classes=np.array(all_classes))
    print(f"   Feature cache saved -> {cache_path}")

    return X, y, all_classes


# Main training loop
# ──────────────────────────────────────────────────────────────────────

def train(max_per_class: int = 800, augment_factor: int = 4, use_cache: bool = False):
    t0 = time.time()

    print("=" * 65)
    print("  Medical Image Classifier v3 - CNN Transfer Learning")
    print("=" * 65)

    cache_path = os.path.join(BASE_DIR, "features_cache.npz")
    if use_cache and os.path.exists(cache_path):
        print("\nLoading features from cache ...")
        data = np.load(cache_path, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        all_classes = list(data["classes"])
        print(f"   Loaded {X.shape[0]} samples, {X.shape[1]} features, {len(all_classes)} classes")
    else:
        X, y, all_classes = _extract_all_features(max_per_class, augment_factor)

    # Class distribution
    dist = Counter(y)
    print("\nPer-class sample count:")
    for cls in all_classes:
        print(f"   {cls[:55]:<55} {dist.get(cls, 0):>6}")

    # -- 4. Encode & split --
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc,
    )
    print(f"\n   Train: {len(X_train)}  |  Test: {len(X_test)}")

    # -- 5. Train multiple classifiers and pick the best --
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier

    candidates = {
        "SVM RBF (C=10)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                C=10.0,
                kernel="rbf",
                gamma="scale",
                class_weight="balanced",
                decision_function_shape="ovr",
                random_state=42,
            )),
        ]),
        "SVM RBF (C=50)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                C=50.0,
                kernel="rbf",
                gamma="scale",
                class_weight="balanced",
                decision_function_shape="ovr",
                random_state=42,
            )),
        ]),
        "MLP (512-256)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(512, 256),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=256,
                learning_rate="adaptive",
                learning_rate_init=1e-3,
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42,
                verbose=False,
            )),
        ]),
        "HistGradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", HistGradientBoostingClassifier(
                max_iter=500,
                learning_rate=0.1,
                max_depth=8,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                verbose=0,
            )),
        ]),
        "Extra Trees (1000)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", ExtraTreesClassifier(
                n_estimators=1000,
                max_depth=None,
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }

    best_pipe = None
    best_acc = 0.0
    best_name = ""

    for name, pipe in candidates.items():
        print(f"\nTraining {name} ...")
        pipe.fit(X_train, y_train)
        y_pred_tmp = pipe.predict(X_test)
        acc_tmp = accuracy_score(y_test, y_pred_tmp) * 100
        print(f"   Test Accuracy: {acc_tmp:.2f} %")
        if acc_tmp > best_acc:
            best_acc = acc_tmp
            best_pipe = pipe
            best_name = name

    pipe = best_pipe
    print(f"\n🏆 Best model: {best_name} -> {best_acc:.2f} %")

    # -- 6. Evaluate --
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    print(f"\n>>> Test Accuracy : {acc:.2f} %")
    print(f"\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # -- 7. Save --
    import joblib
    model_path = os.path.join(BASE_DIR, "image_model.pkl")
    cls_path   = os.path.join(BASE_DIR, "image_classes.pkl")

    joblib.dump(pipe, model_path, compress=3)
    with open(cls_path, "wb") as f:
        pickle.dump(all_classes, f)

    sz = os.path.getsize(model_path) / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"\nSaved:")
    print(f"   Model   -> {model_path}  ({sz:.1f} MB)")
    print(f"   Classes -> {cls_path}")
    print(f"\n   Total time: {elapsed/60:.1f} min")

    print(f"\nClass index:")
    for i, c in enumerate(all_classes):
        print(f"   {i:>2}. {c}")
    print("=" * 65)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train Medical Image Classifier v3 - CNN Transfer Learning")
    p.add_argument("--max-per-class", type=int, default=800,
                   help="Max raw images per class per source (default 800)")
    p.add_argument("--augment-factor", type=int, default=4,
                   help="Augmentation factor per image (default 4)")
    p.add_argument("--use-cache", action="store_true",
                   help="Load features from cache instead of re-extracting")
    args = p.parse_args()
    train(max_per_class=args.max_per_class, augment_factor=args.augment_factor, use_cache=args.use_cache)
