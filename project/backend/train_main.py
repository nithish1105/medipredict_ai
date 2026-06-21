"""
Image model trainer — uses HistGradientBoosting and MLP on scaled CNN features.
Saves to image_model.pkl compatible with app.py.
"""
import os, pickle, warnings, time, sys
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(BASE, "train_result.txt")

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")

open(LOG, "w").close()

log("=" * 60)
log("Image Model Training — HistGB + MLP on Scaled Features")
log("=" * 60)

data = np.load(os.path.join(BASE, "features_cache.npz"), allow_pickle=True)
X, y = data["X"], data["y"]
classes = list(data["classes"])

le = LabelEncoder()
y_enc = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
log(f"Samples: {X.shape[0]}, Train: {len(X_train)}, Test: {len(X_test)}, Classes: {len(classes)}")

# Standardize features (critical for CNN features!)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

best_acc = 0.0
best_model = None
best_name = ""

# --- 1) HistGradientBoosting (fast, good on standardized features) ---
log("\n[1] Training HistGradientBoosting (1000 iters, lr=0.05)...")
t1 = time.time()
hgb = HistGradientBoostingClassifier(
    max_iter=1000, learning_rate=0.05, max_depth=10,
    min_samples_leaf=3, l2_regularization=0.01,
    class_weight="balanced", random_state=42
)
hgb.fit(X_train_s, y_train)
acc = accuracy_score(y_test, hgb.predict(X_test_s)) * 100
log(f"    HistGB accuracy: {acc:.2f}% ({time.time()-t1:.0f}s)")
if acc > best_acc:
    best_acc = acc
    best_model = Pipeline([("scaler", StandardScaler()), ("clf", hgb)])
    best_name = "HistGB"

# --- 2) MLP (wider, more iterations) ---
log("\n[2] Training MLP (1024-512-256, max_iter=1000)...")
t1 = time.time()
mlp = MLPClassifier(
    hidden_layer_sizes=(1024, 512, 256),
    activation="relu", solver="adam",
    alpha=1e-4, batch_size=256,
    learning_rate="adaptive", learning_rate_init=1e-3,
    max_iter=1000, early_stopping=True,
    validation_fraction=0.15, random_state=42, verbose=False,
)
mlp.fit(X_train_s, y_train)
acc = accuracy_score(y_test, mlp.predict(X_test_s)) * 100
log(f"    MLP accuracy: {acc:.2f}% ({time.time()-t1:.0f}s)")
if acc > best_acc:
    best_acc = acc
    best_model = Pipeline([("scaler", StandardScaler()), ("clf", mlp)])
    best_name = "MLP"

# --- 3) Larger MLP ---
log("\n[3] Training MLP (2048-1024-512-256, max_iter=1000)...")
t1 = time.time()
mlp2 = MLPClassifier(
    hidden_layer_sizes=(2048, 1024, 512, 256),
    activation="relu", solver="adam",
    alpha=5e-5, batch_size=256,
    learning_rate="adaptive", learning_rate_init=5e-4,
    max_iter=1000, early_stopping=True,
    validation_fraction=0.15, random_state=42, verbose=False,
)
mlp2.fit(X_train_s, y_train)
acc = accuracy_score(y_test, mlp2.predict(X_test_s)) * 100
log(f"    MLP-large accuracy: {acc:.2f}% ({time.time()-t1:.0f}s)")
if acc > best_acc:
    best_acc = acc
    best_model = Pipeline([("scaler", StandardScaler()), ("clf", mlp2)])
    best_name = "MLP-large"

# --- Summary and save ---
log(f"\nBest model: {best_name} -> {best_acc:.2f}%")

# Refit the pipeline (scaler + classifier) on full training data
log("Refitting pipeline on training data...")
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
final_acc = accuracy_score(y_test, y_pred) * 100
log(f"Final accuracy after refit: {final_acc:.2f}%")

report = classification_report(y_test, y_pred, target_names=le.classes_)
log(report)

log("Saving model (no compression for speed)...")
joblib.dump(best_model, os.path.join(BASE, "image_model.pkl"), compress=1)
with open(os.path.join(BASE, "image_classes.pkl"), "wb") as f:
    pickle.dump(classes, f)

sz = os.path.getsize(os.path.join(BASE, "image_model.pkl")) / (1024*1024)
log(f"\nSaved: image_model.pkl ({sz:.1f} MB)")
log(f"FINAL ACCURACY: {final_acc:.2f}%")
log("=" * 60)
