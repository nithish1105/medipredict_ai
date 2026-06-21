"""
Minimal image model trainer — uses only fast classifiers on cached features.
Writes results to train_result.txt for easy checking.
"""
import os, sys, time, pickle, warnings, json
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))

# Load cached features
data = np.load(os.path.join(BASE, "features_cache.npz"), allow_pickle=True)
X, y = data["X"], data["y"]
classes = list(data["classes"])
le = LabelEncoder()
y_enc = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

results = {}

# 1) Extra Trees
print("ExtraTrees...", flush=True)
et = ExtraTreesClassifier(n_estimators=1500, class_weight="balanced", random_state=42, n_jobs=-1)
et.fit(X_train, y_train)
acc = accuracy_score(y_test, et.predict(X_test)) * 100
results["ExtraTrees"] = acc
print(f"  {acc:.2f}%", flush=True)

# 2) Random Forest
print("RandomForest...", flush=True)
rf = RandomForestClassifier(n_estimators=1500, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
acc = accuracy_score(y_test, rf.predict(X_test)) * 100
results["RandomForest"] = acc
print(f"  {acc:.2f}%", flush=True)

# 3) HistGradientBoosting
print("HistGB...", flush=True)
sc = StandardScaler()
Xts = sc.fit_transform(X_train)
Xtes = sc.transform(X_test)
hgb = HistGradientBoostingClassifier(max_iter=800, learning_rate=0.05, max_depth=10, min_samples_leaf=3, class_weight="balanced", random_state=42)
hgb.fit(Xts, y_train)
acc = accuracy_score(y_test, hgb.predict(Xtes)) * 100
results["HistGB"] = acc
print(f"  {acc:.2f}%", flush=True)

# 4) MLP
print("MLP...", flush=True)
mlp = MLPClassifier(hidden_layer_sizes=(1024,512,256), activation="relu", solver="adam", alpha=1e-4, batch_size=256, learning_rate="adaptive", learning_rate_init=1e-3, max_iter=500, early_stopping=True, validation_fraction=0.15, random_state=42)
mlp.fit(Xts, y_train)
acc = accuracy_score(y_test, mlp.predict(Xtes)) * 100
results["MLP"] = acc
print(f"  {acc:.2f}%", flush=True)

# Pick best
best_name = max(results, key=results.get)
best_acc = results[best_name]
print(f"\nBest: {best_name} -> {best_acc:.2f}%", flush=True)

# Save best model
if best_name == "ExtraTrees":
    best_model = et
elif best_name == "RandomForest":
    best_model = rf
elif best_name == "HistGB":
    best_model = Pipeline([("scaler", StandardScaler()), ("clf", HistGradientBoostingClassifier(max_iter=800, learning_rate=0.05, max_depth=10, min_samples_leaf=3, class_weight="balanced", random_state=42))])
    best_model.fit(X_train, y_train)
else:
    best_model = Pipeline([("scaler", StandardScaler()), ("clf", MLPClassifier(hidden_layer_sizes=(1024,512,256), activation="relu", solver="adam", alpha=1e-4, batch_size=256, learning_rate="adaptive", learning_rate_init=1e-3, max_iter=500, early_stopping=True, validation_fraction=0.15, random_state=42))])
    best_model.fit(X_train, y_train)

joblib.dump(best_model, os.path.join(BASE, "image_model.pkl"), compress=3)
with open(os.path.join(BASE, "image_classes.pkl"), "wb") as f:
    pickle.dump(classes, f)

# Write results to file for easy checking
with open(os.path.join(BASE, "train_result.txt"), "w") as f:
    f.write(json.dumps(results, indent=2))
    f.write(f"\nBest: {best_name} -> {best_acc:.2f}%\n")

print("SAVED.", flush=True)
