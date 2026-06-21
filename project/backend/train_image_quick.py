"""
Quick image model training on cached EfficientNet features.
Uses fast classifiers only - targets ~85%+ accuracy quickly.
"""
import os, sys, time, pickle, warnings
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def train():
    t0 = time.time()
    print("=" * 65)
    print("  Quick Image Model Training (from cached features)")
    print("=" * 65)

    cache_path = os.path.join(BASE_DIR, "features_cache.npz")
    if not os.path.exists(cache_path):
        print("ERROR: features_cache.npz not found. Run train_image_model.py first.")
        sys.exit(1)

    data = np.load(cache_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    all_classes = list(data["classes"])
    print(f"  Loaded {X.shape[0]} samples, {X.shape[1]} features, {len(all_classes)} classes\n")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc,
    )
    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    # ── Fast classifiers only ──
    candidates = {}

    # 1) HistGradientBoosting - fast and effective
    print("  Training HistGradientBoosting...")
    t1 = time.time()
    histgb = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.08,
            max_depth=12, min_samples_leaf=2,
            l2_regularization=0.01,
            class_weight="balanced", random_state=42, verbose=0,
        )),
    ])
    histgb.fit(X_train, y_train)
    y_pred = histgb.predict(X_test)
    acc1 = accuracy_score(y_test, y_pred) * 100
    print(f"    Accuracy: {acc1:.2f}%  ({time.time()-t1:.0f}s)")
    candidates["HistGB"] = (acc1, histgb)

    # 2) Extra Trees - fast with parallization
    print("  Training Extra Trees (1000 estimators)...")
    t1 = time.time()
    et = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", ExtraTreesClassifier(
            n_estimators=1000, max_depth=30,
            min_samples_leaf=1, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )),
    ])
    et.fit(X_train, y_train)
    y_pred = et.predict(X_test)
    acc2 = accuracy_score(y_test, y_pred) * 100
    print(f"    Accuracy: {acc2:.2f}%  ({time.time()-t1:.0f}s)")
    candidates["ExtraTrees"] = (acc2, et)

    # 3) Random Forest - fast with parallelization
    print("  Training Random Forest (1000 estimators)...")
    t1 = time.time()
    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=1000, max_depth=30,
            min_samples_leaf=1, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )),
    ])
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc3 = accuracy_score(y_test, y_pred) * 100
    print(f"    Accuracy: {acc3:.2f}%  ({time.time()-t1:.0f}s)")
    candidates["RandomForest"] = (acc3, rf)

    # Pick best
    best_name = max(candidates, key=lambda k: candidates[k][0])
    best_acc, best_pipe = candidates[best_name]
    
    print(f"\n  Best model: {best_name} with {best_acc:.2f}%")

    # Final evaluation
    y_final = best_pipe.predict(X_test)
    print("\n" + "=" * 65)
    print(f"  Final Test Accuracy: {best_acc:.2f}%")
    print("=" * 65)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_final, target_names=[all_classes[i] for i in range(len(all_classes))]))

    # Save model wrapper
    import model_defs
    wrapper = model_defs.ScikitLearnWrapper(best_pipe, all_classes)

    out_path = os.path.join(BASE_DIR, "image_model.pkl")
    joblib.dump(wrapper, out_path, compress=3)
    print(f"\n  Model saved to: {out_path}")
    
    # Also save image classes
    cls_path = os.path.join(BASE_DIR, "image_classes.pkl")
    with open(cls_path, "wb") as f:
        pickle.dump(all_classes, f)
    print(f"  Classes saved to: {cls_path}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    train()
