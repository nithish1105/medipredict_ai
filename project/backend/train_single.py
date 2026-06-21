"""Train single best classifier on cached EfficientNet features."""
import os, pickle, warnings, json, sys
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(BASE, "train_result.txt")

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")

# Clear log
open(LOG, "w").close()

log("Loading features...")
data = np.load(os.path.join(BASE, "features_cache.npz"), allow_pickle=True)
X, y = data["X"], data["y"]
classes = list(data["classes"])

le = LabelEncoder()
y_enc = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
log(f"Samples: {X.shape[0]}, Train: {len(X_train)}, Test: {len(X_test)}, Classes: {len(classes)}")

# ExtraTrees is fast and effective for CNN features
log("Training ExtraTrees (2000 estimators)...")
clf = ExtraTreesClassifier(
    n_estimators=2000,
    max_depth=None,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
log(f"ExtraTrees accuracy: {acc:.2f}%")

report = classification_report(y_test, y_pred, target_names=le.classes_)
log(report)

# Save
joblib.dump(clf, os.path.join(BASE, "image_model.pkl"), compress=3)
with open(os.path.join(BASE, "image_classes.pkl"), "wb") as f:
    pickle.dump(classes, f)

log(f"DONE. Accuracy: {acc:.2f}%")
log(f"Model saved to image_model.pkl")
