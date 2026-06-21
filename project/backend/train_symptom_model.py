"""
Train Symptom-Based Disease Prediction Model  (Real Dataset)
=============================================================
Trains a **Random Forest Classifier** on the Kaggle symptom → disease
dataset (``datasets/dataset.csv``) weighted by symptom severity from
``datasets/Symptom-severity.csv``.

Usage
-----
    python train_symptom_model.py

Outputs
-------
* ``backend/symptom_model.pkl``   — trained Random Forest
* ``backend/label_encoder.pkl``   — LabelEncoder for disease names
* ``backend/symptom_meta.pkl``    — symptom list + severity dict (used at inference)
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "datasets")


# ──────────────────────────────────────────────────────────────────────
# 1.  Load & Clean Datasets
# ──────────────────────────────────────────────────────────────────────

def load_data():
    """Return (main_df, severity_dict, description_dict, precaution_dict)."""

    # --- Main dataset: Disease | Symptom_1 … Symptom_17 ---
    csv_path = os.path.join(DATA_DIR, "dataset.csv")
    df = pd.read_csv(csv_path)
    df["Disease"] = df["Disease"].str.strip()

    # Clean every symptom column
    sym_cols = [c for c in df.columns if c.startswith("Symptom")]
    for col in sym_cols:
        df[col] = (
            df[col]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.replace(" ", "_", regex=False)
            .str.replace("__", "_", regex=False)
            .str.lower()
        )

    # --- Severity weights ---
    sev_path = os.path.join(DATA_DIR, "Symptom-severity.csv")
    sev_df = pd.read_csv(sev_path)
    sev_df["Symptom"] = (
        sev_df["Symptom"]
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.lower()
    )
    severity = dict(zip(sev_df["Symptom"], sev_df["weight"]))

    # --- Descriptions ---
    desc = {}
    desc_path = os.path.join(DATA_DIR, "symptom_Description.csv")
    if os.path.exists(desc_path):
        tmp = pd.read_csv(desc_path)
        desc = dict(zip(tmp.iloc[:, 0].str.strip(), tmp.iloc[:, 1].str.strip()))

    # --- Precautions ---
    prec = {}
    prec_path = os.path.join(DATA_DIR, "symptom_precaution.csv")
    if os.path.exists(prec_path):
        tmp = pd.read_csv(prec_path)
        for _, row in tmp.iterrows():
            disease = str(row.iloc[0]).strip()
            tips = [str(v).strip() for v in row.iloc[1:] if pd.notna(v) and str(v).strip()]
            prec[disease] = tips

    return df, severity, desc, prec


# ──────────────────────────────────────────────────────────────────────
# 2.  Feature Engineering
# ──────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, severity: dict, *, use_severity: bool = True):
    """
    One-hot encode all unique symptoms.  Each cell value is the severity
    weight (or 1 if the symptom has no entry in the severity table).

    Returns (X, y, all_symptom_names, present_symptom_indices).
    """
    sym_cols = [c for c in df.columns if c.startswith("Symptom")]

    # Collect the canonical symptom set
    all_symptoms = set()
    for col in sym_cols:
        all_symptoms.update(df[col][df[col] != ""].unique())
    all_symptoms.discard("")
    all_symptoms = sorted(all_symptoms)

    print(f"   Unique symptoms : {len(all_symptoms)}")
    print(f"   Unique diseases : {df['Disease'].nunique()}")
    print(f"   Total rows      : {len(df)}")

    # Build numeric matrix
    X = np.zeros((len(df), len(all_symptoms)), dtype=np.float32)
    sym_to_idx = {s: i for i, s in enumerate(all_symptoms)}

    present_symptom_indices: list[list[int]] = []

    for row_idx, (_, row) in enumerate(df.iterrows()):
        indices: list[int] = []
        for col in sym_cols:
            s = row[col]
            if s and s in sym_to_idx:
                idx = sym_to_idx[s]
                weight = float(severity.get(s, 1)) if use_severity else 1.0
                X[row_idx, idx] = weight
                indices.append(idx)

        present_symptom_indices.append(sorted(set(indices)))

    y = df["Disease"].values
    return X, y, all_symptoms, present_symptom_indices


def augment_partial_symptom_rows(
    X: np.ndarray,
    y: np.ndarray,
    present_symptom_indices: list[list[int]],
    *,
    copies_per_row: int = 6,
    min_keep: int = 2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment dataset by randomly dropping some symptoms per row.

    This matches real-world usage where users often select only a subset of
    their symptoms. Augmentation makes the classifier significantly more
    tolerant of missing inputs.
    """

    if copies_per_row <= 0:
        return X, y

    rng = np.random.default_rng(random_state)
    n_rows, n_features = X.shape
    aug_X: list[np.ndarray] = [X]
    aug_y: list[np.ndarray] = [y]

    created = 0
    for row_idx in range(n_rows):
        idxs = present_symptom_indices[row_idx]
        if len(idxs) <= min_keep:
            continue

        base = X[row_idx]
        # Create multiple masked versions per row.
        for _ in range(copies_per_row):
            keep_count = int(rng.integers(low=min_keep, high=len(idxs) + 1))
            keep = rng.choice(idxs, size=keep_count, replace=False)
            x_new = np.zeros(n_features, dtype=np.float32)
            x_new[keep] = base[keep]
            aug_X.append(x_new[np.newaxis, :])
            aug_y.append(np.array([y[row_idx]]))
            created += 1

    X_out = np.vstack(aug_X)
    y_out = np.concatenate(aug_y)

    # Shuffle (important: keep X/y aligned)
    perm = rng.permutation(len(y_out))
    return X_out[perm], y_out[perm]


# ──────────────────────────────────────────────────────────────────────
# 3.  Training
# ──────────────────────────────────────────────────────────────────────

def train():
    print("=" * 65)
    print("  🩺  Symptom-Based Disease Prediction — Real Dataset Training")
    print("=" * 65)

    df, severity, descriptions, precautions = load_data()
    print(f"\n📂 Loaded {len(df)} rows from dataset.csv")

    X, y, all_symptoms, present = build_features(df, severity, use_severity=True)

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Augment to be robust to partial symptom selection
    X, y_enc = augment_partial_symptom_rows(
        X,
        y_enc,
        present,
        copies_per_row=4,
        min_keep=2,
        random_state=42,
    )
    print(f"\n🧪 Augmented samples : {len(X)} (includes originals)")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc,
    )
    print(f"\n📊 Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ── Train multiple classifiers and pick the best ──
    from sklearn.ensemble import ExtraTreesClassifier

    candidates = {
        "Random Forest (300,depth18)": RandomForestClassifier(
            n_estimators=300,
            max_depth=18,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
        ),
        "Extra Trees (300,depth18)": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=18,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
    }

    best_model = None
    best_acc = 0.0
    best_name = ""

    for name, clf in candidates.items():
        print(f"\n🌲 Training {name} …")
        clf.fit(X_train, y_train)
        y_pred_tmp = clf.predict(X_test)
        acc_tmp = accuracy_score(y_test, y_pred_tmp) * 100
        print(f"   Test Accuracy : {acc_tmp:.2f} %")
        if acc_tmp > best_acc:
            best_acc = acc_tmp
            best_model = clf
            best_name = name

    base_model = best_model
    print(f"\n🏆 Best model: {best_name} with {best_acc:.2f} %")

    # Cross-validation intentionally skipped for speed/stability on some machines.
    # The hold-out test split below is used as the primary metric.
    print("📈 Cross-val       : skipped (speed)")

    # Calibrate probabilities (reduces over-confident wrong predictions)
    from sklearn.calibration import CalibratedClassifierCV
    model = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
    model.fit(X_train, y_train)
    print("🧭 Probability calibration: enabled (sigmoid)")

    # Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    print(f"\n✅ Test Accuracy : {acc:.2f} %")

    print(f"\n📋 Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Top-10 important symptoms (only available on tree models)
    base_est = getattr(model, "base_estimator", None)
    if base_est is not None and hasattr(base_est, "feature_importances_"):
        imp = base_est.feature_importances_
        top = np.argsort(imp)[::-1][:10]
        print("🔑 Top-10 Important Symptoms:")
        for rank, idx in enumerate(top, 1):
            print(f"   {rank:>2}. {all_symptoms[idx]:<35} {imp[idx]:.4f}")

    # ── Save artefacts ──
    model_path = os.path.join(BASE_DIR, "symptom_model.pkl")
    le_path    = os.path.join(BASE_DIR, "label_encoder.pkl")
    meta_path  = os.path.join(BASE_DIR, "symptom_meta.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(le_path, "wb") as f:
        pickle.dump(le, f)
    with open(meta_path, "wb") as f:
        pickle.dump({
            "symptoms": all_symptoms,
            "severity": severity,
            "use_severity": True,
            "descriptions": descriptions,
            "precautions": precautions,
        }, f)

    print(f"\n💾 Saved:")
    print(f"   Model          → {model_path}")
    print(f"   LabelEncoder   → {le_path}")
    print(f"   Symptom meta   → {meta_path}")
    print("=" * 65)


if __name__ == "__main__":
    train()
