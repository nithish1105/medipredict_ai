"""
Quick training with scikit-learn only (no PyTorch needed)
Uses cached EfficientNet features to train accurate model in ~10 minutes
"""

import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 70)
print("  SKLEARN-ONLY IMAGE MODEL TRAINING")
print("=" * 70)

# Load cached features
cache_path = os.path.join(BASE_DIR, "features_cache.npz")
print(f"\nLoading features from: {cache_path}")

data = np.load(cache_path, allow_pickle=True)
X = data["X"]
y = data["y"]
all_classes = list(data["classes"])

print(f"  Features: {X.shape}")
print(f"  Classes: {len(all_classes)}")

# Encode labels
class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
y_encoded = np.array([class_to_idx[cls] for cls in y])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# Train multiple models
print("\n" + "=" * 70)
print("TRAINING MODELS")
print("=" * 70)

candidates = {
    "SVM (C=10)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(C=10, kernel="rbf", gamma="scale", class_weight="balanced",
                    decision_function_shape="ovr", random_state=42, probability=True))
    ]),
    
    "SVM (C=50)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(C=50, kernel="rbf", gamma="scale", class_weight="balanced",
                    decision_function_shape="ovr", random_state=42, probability=True))
    ]),
    
    "MLP (512-256)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(512, 256), activation="relu",
                             solver="adam", alpha=1e-4, batch_size=256,
                             learning_rate="adaptive", max_iter=300,
                             early_stopping=True, random_state=42, verbose=False))
    ]),
    
    "HistGradientBoosting": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(max_iter=500, learning_rate=0.1,
                                              max_depth=8, min_samples_leaf=5,
                                              class_weight="balanced", random_state=42))
    ]),
    
    "ExtraTrees (1000)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", ExtraTreesClassifier(n_estimators=1000, max_depth=None,
                                     min_samples_leaf=1, class_weight="balanced",
                                     random_state=42, n_jobs=-1))
    ]),
}

best_acc = 0
best_model = None
best_name = ""

for name, model in candidates.items():
    print(f"\n[{name}]")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    print(f"  Accuracy: {acc:.2f}%")
    
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

print("\n" + "=" * 70)
print(f"BEST MODEL: {best_name}")
print(f"Test Accuracy: {best_acc:.2f}%")
print("=" * 70)

# Full evaluation
y_pred_final = best_model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_final, target_names=all_classes))

# Make model compatible with app.py (import from model_defs)
from model_defs import ScikitLearnWrapper

# Wrap and save
wrapper = ScikitLearnWrapper(best_model, all_classes)

model_path = os.path.join(BASE_DIR, "image_model.pkl")
classes_path = os.path.join(BASE_DIR, "image_classes.pkl")

joblib.dump(wrapper, model_path, compress=3)
with open(classes_path, "wb") as f:
    pickle.dump(all_classes, f)

size_mb = os.path.getsize(model_path) / (1024 * 1024)

print(f"\n✅ Model saved:")
print(f"  Path: {model_path}")
print(f"  Size: {size_mb:.1f} MB")
print(f"  Accuracy: {best_acc:.2f}%")
print(f"  Classes: {len(all_classes)}")
print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print("\nYou can now use the model in your Flask app!")
print("Just restart: python app.py")
