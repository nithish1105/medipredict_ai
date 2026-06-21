"""Quick script to build image_model.pkl from checkpoint files."""
import os, sys, pickle, numpy as np, torch
import joblib
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from model_defs import DeepMLP, EnsembleWrapper

# Load checkpoints
ckpts = []
for i in range(10):
    p = os.path.join(BASE, f"ckpt_{i}.pt")
    if os.path.exists(p):
        c = torch.load(p, map_location="cpu", weights_only=False)
        print(f"  ckpt_{i}.pt: width={c['width']}, acc={c['accuracy']:.2f}%")
        ckpts.append(c)
    else:
        break

if not ckpts:
    print("No checkpoints found!"); sys.exit(1)

# Rebuild scaler from first checkpoint
c0 = ckpts[0]
scaler = StandardScaler()
scaler.mean_ = c0["scaler_mean"]
scaler.scale_ = c0["scaler_scale"]
scaler.var_ = c0["scaler_scale"] ** 2
scaler.n_features_in_ = c0["n_features"]
classes = list(c0["classes"])
n_features = c0["n_features"]
n_classes = c0["n_classes"]

# Build model data for EnsembleWrapper
model_data = []
for c in ckpts:
    model_data.append((c["state_dict"], c["width"]))

# EnsembleWrapper is imported from model_defs

wrapper = EnsembleWrapper(model_data, scaler, classes, n_features, n_classes)

# Verify on test split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
data = np.load(os.path.join(BASE, "features_cache.npz"), allow_pickle=True)
X, y = data["X"], data["y"]
le = LabelEncoder(); y_enc = le.fit_transform(y)
_, X_test, _, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

proba = wrapper.predict_proba(X_test)
acc = np.mean(proba.argmax(1) == y_test) * 100
print(f"\nEnsemble test accuracy: {acc:.2f}% ({len(ckpts)} models)")

outpath = os.path.join(BASE, "image_model.pkl")
joblib.dump(wrapper, outpath, compress=1)
sz = os.path.getsize(outpath) / (1024*1024)
print(f"Saved: image_model.pkl ({sz:.1f} MB)")

with open(os.path.join(BASE, "image_classes.pkl"), "wb") as f:
    pickle.dump(classes, f)
print("Saved: image_classes.pkl")
print("DONE!")
