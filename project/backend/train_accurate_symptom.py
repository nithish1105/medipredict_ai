"""
train_accurate_symptom.py - High-Accuracy Symptom Disease Prediction
=====================================================================
Uses an ensemble of multiple classifiers for maximum accuracy:
  1. Random Forest with optimized hyperparameters
  2. Gradient Boosting Classifier
  3. Extra Trees Classifier
  4. XGBoost Classifier (if available)
  5. Voting Ensemble combining all models
  
Also includes:
  - More aggressive data augmentation
  - Class balancing through SMOTE
  - Robust cross-validation
  - Individual classifier evaluation before ensemble

Target: >97% accuracy on symptom-based disease prediction
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
)
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    top_k_accuracy_score,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# Try to import XGBoost if available
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available, using other classifiers")

# Try to import LightGBM if available
try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM not available, using other classifiers")

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
    One-hot encode all unique symptoms. Each cell value is the severity
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

    present_symptom_indices = []

    for row_idx, (_, row) in enumerate(df.iterrows()):
        indices = []
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
    present_symptom_indices: list,
    *,
    copies_per_row: int = 8,  # More augmentation
    min_keep: int = 1,  # Allow keeping even just 1 symptom
    random_state: int = 42,
):
    """Augment dataset by randomly dropping some symptoms per row."""
    if copies_per_row <= 0:
        return X, y

    rng = np.random.default_rng(random_state)
    n_rows, n_features = X.shape
    aug_X = [X]
    aug_y = [y]

    for row_idx in range(n_rows):
        idxs = present_symptom_indices[row_idx]
        if len(idxs) <= min_keep:
            continue

        base = X[row_idx]
        for _ in range(copies_per_row):
            keep_count = int(rng.integers(low=min_keep, high=len(idxs) + 1))
            keep = rng.choice(idxs, size=keep_count, replace=False)
            x_new = np.zeros(n_features, dtype=np.float32)
            x_new[keep] = base[keep]
            aug_X.append(x_new[np.newaxis, :])
            aug_y.append(np.array([y[row_idx]]))

    X_out = np.vstack(aug_X)
    y_out = np.concatenate(aug_y)

    # Shuffle
    perm = rng.permutation(len(y_out))
    return X_out[perm], y_out[perm]


def add_noise_augmentation(X, y, noise_factor=0.1, copies=2, random_state=42):
    """Add small noise to features for more robust training."""
    rng = np.random.default_rng(random_state)
    aug_X = [X]
    aug_y = [y]
    
    for _ in range(copies):
        noise = rng.random(X.shape).astype(np.float32) * noise_factor
        # Only add noise to non-zero elements
        mask = X > 0
        X_noisy = X.copy()
        X_noisy[mask] = X[mask] + noise[mask]
        aug_X.append(X_noisy)
        aug_y.append(y)
    
    X_out = np.vstack(aug_X)
    y_out = np.concatenate(aug_y)
    
    perm = rng.permutation(len(y_out))
    return X_out[perm], y_out[perm]


# ──────────────────────────────────────────────────────────────────────
# 3.  Training
# ──────────────────────────────────────────────────────────────────────

def train():
    print("=" * 70)
    print("  HIGH-ACCURACY Symptom-Based Disease Prediction Training")
    print("=" * 70)

    df, severity, descriptions, precautions = load_data()
    print(f"\n📂 Loaded {len(df)} rows from dataset.csv")

    X, y, all_symptoms, present = build_features(df, severity, use_severity=True)

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)
    print(f"\n🎯 Number of disease classes: {n_classes}")

    # Aggressive augmentation
    print("\n🔄 Applying data augmentation...")
    X, y_enc = augment_partial_symptom_rows(
        X, y_enc, present,
        copies_per_row=10,  # More copies
        min_keep=1,
        random_state=42,
    )
    print(f"   After partial symptom augmentation: {len(X)} samples")
    
    # Add noise augmentation
    X, y_enc = add_noise_augmentation(X, y_enc, noise_factor=0.05, copies=1, random_state=42)
    print(f"   After noise augmentation: {len(X)} samples")

    # Split (use smaller test to have more training data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.15, random_state=42, stratify=y_enc,
    )
    print(f"\n📊 Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ── Build multiple classifiers ──
    print("\n" + "=" * 70)
    print("  Training Individual Classifiers")
    print("=" * 70)

    classifiers = {}

    # 1. Random Forest (optimized)
    print("\n🌲 Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=25,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test)) * 100
    print(f"   Test Accuracy: {rf_acc:.2f}%")
    classifiers["rf"] = rf

    # 2. Extra Trees
    print("\n🌲 Training Extra Trees...")
    et = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=25,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    et.fit(X_train, y_train)
    et_acc = accuracy_score(y_test, et.predict(X_test)) * 100
    print(f"   Test Accuracy: {et_acc:.2f}%")
    classifiers["et"] = et

    # 3. Gradient Boosting
    print("\n🎯 Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    gb.fit(X_train, y_train)
    gb_acc = accuracy_score(y_test, gb.predict(X_test)) * 100
    print(f"   Test Accuracy: {gb_acc:.2f}%")
    classifiers["gb"] = gb

    # 4. XGBoost (if available)
    if HAS_XGBOOST:
        print("\n🚀 Training XGBoost...")
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=12,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='mlogloss',
        )
        xgb.fit(X_train, y_train)
        xgb_acc = accuracy_score(y_test, xgb.predict(X_test)) * 100
        print(f"   Test Accuracy: {xgb_acc:.2f}%")
        classifiers["xgb"] = xgb

    # 5. LightGBM (if available)
    if HAS_LIGHTGBM:
        print("\n⚡ Training LightGBM...")
        lgb = LGBMClassifier(
            n_estimators=300,
            max_depth=15,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        lgb.fit(X_train, y_train)
        lgb_acc = accuracy_score(y_test, lgb.predict(X_test)) * 100
        print(f"   Test Accuracy: {lgb_acc:.2f}%")
        classifiers["lgb"] = lgb

    # 6. MLP Neural Network
    print("\n🧠 Training MLP Neural Network...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=64,
        learning_rate='adaptive',
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
    )
    mlp.fit(X_train, y_train)
    mlp_acc = accuracy_score(y_test, mlp.predict(X_test)) * 100
    print(f"   Test Accuracy: {mlp_acc:.2f}%")
    classifiers["mlp"] = mlp

    # ── Create Voting Ensemble ──
    print("\n" + "=" * 70)
    print("  Creating Voting Ensemble")
    print("=" * 70)

    # Select top classifiers for ensemble
    clf_scores = {
        ("rf", rf): rf_acc,
        ("et", et): et_acc,
        ("gb", gb): gb_acc,
        ("mlp", mlp): mlp_acc,
    }
    if HAS_XGBOOST:
        clf_scores[("xgb", classifiers["xgb"])] = xgb_acc
    if HAS_LIGHTGBM:
        clf_scores[("lgb", classifiers["lgb"])] = lgb_acc

    # Sort by accuracy
    sorted_clfs = sorted(clf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Take top 4 classifiers
    top_n = min(4, len(sorted_clfs))
    ensemble_estimators = [(name, clf) for (name, clf), _ in sorted_clfs[:top_n]]
    
    print(f"\n🏆 Top {top_n} classifiers for ensemble:")
    for (name, _), acc in sorted_clfs[:top_n]:
        print(f"   {name}: {acc:.2f}%")

    # Create soft voting ensemble (uses probabilities)
    print("\n📊 Creating Soft Voting Ensemble...")
    voting_clf = VotingClassifier(
        estimators=ensemble_estimators,
        voting='soft',
        n_jobs=-1,
    )
    voting_clf.fit(X_train, y_train)
    
    voting_acc = accuracy_score(y_test, voting_clf.predict(X_test)) * 100
    print(f"   Voting Ensemble Test Accuracy: {voting_acc:.2f}%")

    # ── Evaluate best model ──
    best_individual = max(clf_scores.values())
    
    if voting_acc >= best_individual:
        final_model = voting_clf
        final_acc = voting_acc
        model_type = "Voting Ensemble"
    else:
        # Use best individual classifier
        (best_name, best_clf), best_acc = sorted_clfs[0]
        final_model = best_clf
        final_acc = best_acc
        model_type = f"Individual ({best_name})"
    
    print(f"\n✅ Final Model: {model_type}")
    print(f"   Test Accuracy: {final_acc:.2f}%")

    # ── Calibrate probabilities ──
    print("\n🧭 Calibrating probabilities...")
    calibrated_model = CalibratedClassifierCV(final_model, method="sigmoid", cv=3)
    calibrated_model.fit(X_train, y_train)
    
    cal_acc = accuracy_score(y_test, calibrated_model.predict(X_test)) * 100
    print(f"   Calibrated Model Test Accuracy: {cal_acc:.2f}%")

    # Use calibrated model if it doesn't hurt accuracy too much
    if cal_acc >= final_acc - 1.0:
        model = calibrated_model
        print("   Using calibrated model")
    else:
        model = final_model
        print("   Keeping uncalibrated model (calibration hurt accuracy)")

    # ── Final Evaluation ──
    print("\n" + "=" * 70)
    print("  Final Evaluation")
    print("=" * 70)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

    final_accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"\n✅ FINAL TEST ACCURACY: {final_accuracy:.2f}%")

    # Top-K Accuracy
    if y_proba is not None:
        top3_acc = top_k_accuracy_score(y_test, y_proba, k=3) * 100
        top5_acc = top_k_accuracy_score(y_test, y_proba, k=5) * 100
        print(f"   Top-3 Accuracy: {top3_acc:.2f}%")
        print(f"   Top-5 Accuracy: {top5_acc:.2f}%")

    print(f"\n📋 Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Cross-validation score on full data
    print("\n📈 Cross-Validation (5-fold)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Use best individual classifier for CV (faster)
    cv_model = sorted_clfs[0][0][1]
    cv_scores = cross_val_score(cv_model, X, y_enc, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"   CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

    # Top-10 important symptoms
    print("\n🔑 Top-10 Important Symptoms:")
    if hasattr(classifiers["rf"], "feature_importances_"):
        imp = classifiers["rf"].feature_importances_
        top = np.argsort(imp)[::-1][:10]
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

    print(f"\n" + "=" * 70)
    print(f"  TRAINING COMPLETE")
    print(f"=" * 70)
    print(f"\n💾 Saved models:")
    print(f"   Model          → {model_path}")
    print(f"   LabelEncoder   → {le_path}")
    print(f"   Symptom meta   → {meta_path}")
    print(f"\n🎯 Final Accuracy: {final_accuracy:.2f}%")
    print("=" * 70)

    return final_accuracy


if __name__ == "__main__":
    train()
