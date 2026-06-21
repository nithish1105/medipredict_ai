"""
train_fast.py - Quick training using HistGradientBoosting
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Import wrapper from model_defs so it can be found when loading
from model_defs import ScikitLearnWrapper

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(BASE, "features_cache.npz")
OUTPUT_PATH = os.path.join(BASE, "image_model.pkl")

print("=" * 60)
print("  FAST IMAGE MODEL TRAINING")
print("=" * 60)

# Load cached features
data = np.load(CACHE_PATH, allow_pickle=True)
X = data["X"]
y = data["y"]
classes = list(data["classes"])

print(f"\nFeatures: {X.shape}, Classes: {len(classes)}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# Fast model
print("\nTraining HistGradientBoosting...")
model = HistGradientBoostingClassifier(
    max_iter=200,
    max_depth=12,
    learning_rate=0.1,
    random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model.fit(X_train_s, y_train)
acc = model.score(X_test_s, y_test)
print(f"  Accuracy: {acc*100:.2f}%")

# Create pipeline
pipeline = Pipeline([
    ("scaler", scaler),
    ("model", model)
])

# Wrap and save
wrapper = ScikitLearnWrapper(pipeline, classes)
joblib.dump(wrapper, OUTPUT_PATH)
print(f"\n✅ Saved to {OUTPUT_PATH}")
print(f"   Size: {os.path.getsize(OUTPUT_PATH) / 1024 / 1024:.1f} MB")
