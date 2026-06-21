"""
model_defs.py  -  Shared model class definitions for training and inference
===========================================================================
All model architectures used in image_model.pkl are defined here so that
joblib/pickle can find them when loading the saved model in app.py.

Supports both PyTorch models and sklearn-only models.
PyTorch is optional - sklearn models work without it.
"""

import os
import numpy as np
from PIL import Image

# PyTorch is optional - only needed for PyTorch-based models
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


# ═══════════════════════════════════════════════════════════════════
# Sklearn Model Wrapper (works without PyTorch)
# ═══════════════════════════════════════════════════════════════════

class ScikitLearnWrapper:
    """Wrapper to make sklearn model compatible with existing app.py."""
    def __init__(self, pipeline, classes):
        self.pipeline = pipeline
        self.classes_ = np.array(classes)
    
    def predict(self, X):
        """X can be feature array or list of image paths (for compatibility)."""
        if isinstance(X, (list, tuple)) and len(X) > 0 and isinstance(X[0], str):
            # If paths provided, extract features (requires utils.py)
            from utils import extract_image_features
            X = np.array([extract_image_features(p) for p in X])
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if isinstance(X, (list, tuple)) and len(X) > 0 and isinstance(X[0], str):
            from utils import extract_image_features
            X = np.array([extract_image_features(p) for p in X])
        if hasattr(self.pipeline, 'predict_proba'):
            return self.pipeline.predict_proba(X)
        else:
            # For models without predict_proba, use decision_function
            decision = self.pipeline.decision_function(X)
            from scipy.special import softmax
            return softmax(decision, axis=1)


# ═══════════════════════════════════════════════════════════════════
# PyTorch Models (only defined if torch is available)
# ═══════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    class DeepMLP(nn.Module):
        """Standard deep MLP with batch norm and dropout."""
        def __init__(self, nf, nc, width=1024):
            super().__init__()
            self.bn = nn.BatchNorm1d(nf)
            self.net = nn.Sequential(
                nn.Linear(nf, width), nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(0.4),
                nn.Linear(width, width // 2), nn.BatchNorm1d(width // 2), nn.GELU(), nn.Dropout(0.35),
                nn.Linear(width // 2, width // 4), nn.BatchNorm1d(width // 4), nn.GELU(), nn.Dropout(0.25),
                nn.Linear(width // 4, nc),
            )
        def forward(self, x):
            return self.net(self.bn(x))


    class ResidualMLP(nn.Module):
        """Deep MLP with residual connections and squeeze-excitation attention."""
        def __init__(self, n_in, n_out, width=1024, drop=0.4):
            super().__init__()
            self.input_bn = nn.BatchNorm1d(n_in)
            self.input_proj = nn.Linear(n_in, width)
            
            self.block1 = nn.Sequential(
                nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(drop),
                nn.Linear(width, width),
            )
            self.block2 = nn.Sequential(
                nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(drop * 0.8),
                nn.Linear(width, width),
            )
            self.block3 = nn.Sequential(
                nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(drop * 0.6),
                nn.Linear(width, width // 2),
            )
            
            self.se = nn.Sequential(
                nn.Linear(width, width // 8),
                nn.GELU(),
                nn.Linear(width // 8, width),
                nn.Sigmoid(),
            )
            
            self.head = nn.Sequential(
                nn.BatchNorm1d(width // 2),
                nn.GELU(),
                nn.Dropout(drop * 0.4),
                nn.Linear(width // 2, n_out),
            )
        
        def forward(self, x):
            x = self.input_bn(x)
            x = self.input_proj(x)
            r = x
            x = self.block1(x) + r
            att = self.se(x)
            x = x * att
            r = x
            x = self.block2(x) + r
            x = self.block3(x)
            return self.head(x)


    class MultiModelEnsemble:
        """
        Inference wrapper: extracts features from multiple backbones,
        concatenates them, and runs MLP ensemble for prediction.
        """
        
        def __init__(self, model_states, widths, classes, total_feat_dim,
                     backbone_names, backbone_dims):
            self.classes_ = np.array(classes)
            self._n_cls = len(classes)
            self._total_dim = total_feat_dim
            self._states = model_states
            self._widths = widths
            self._backbone_names = backbone_names
            self._backbone_dims = backbone_dims
            self._models = None
            self._backbones = None
            self._transform = None
        
        def _load_mlps(self):
            if self._models is None:
                self._models = []
                for s, w in zip(self._states, self._widths):
                    m = ResidualMLP(self._total_dim, self._n_cls, w)
                    m.load_state_dict(s)
                    m.eval()
                    self._models.append(m)
        
        def _load_backbones(self):
            if self._backbones is not None:
                return
            
            from torchvision import models as tv_models, transforms as tv_transforms
            
            self._transform = tv_transforms.Compose([
                tv_transforms.Resize((224, 224)),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            
            self._backbones = []
            base_dir = os.path.dirname(os.path.abspath(__file__))
            
            for bname in self._backbone_names:
                bn = bname.lower()
                if "efficientnet" in bn:
                    m = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.DEFAULT)
                    m.classifier = nn.Identity()
                elif "resnet" in bn:
                    m = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
                    m.fc = nn.Identity()
                elif "densenet" in bn:
                    m = tv_models.densenet121(weights=tv_models.DenseNet121_Weights.DEFAULT)
                    m.classifier = nn.Identity()
                elif "mobilenet" in bn:
                    m = tv_models.mobilenet_v3_large(weights=tv_models.MobileNet_V3_Large_Weights.DEFAULT)
                    m.classifier = nn.Identity()
                else:
                    continue
                
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
                self._backbones.append(m)
        
        def extract_features(self, img_path):
            self._load_backbones()
            
            img = Image.open(img_path).convert("RGB")
            tensor = self._transform(img).unsqueeze(0)
            img.close()
            
            parts = []
            with torch.no_grad():
                for m in self._backbones:
                    f = m(tensor).cpu().numpy().flatten()
                    parts.append(f)
            
            return np.concatenate(parts)
        
        def predict(self, X):
            self._load_mlps()
            if isinstance(X, (list, tuple)) and len(X) > 0 and isinstance(X[0], str):
                X = np.array([self.extract_features(p) for p in X])
            X_t = torch.tensor(np.array(X), dtype=torch.float32)
            with torch.no_grad():
                p = sum(torch.softmax(m(X_t), dim=1) for m in self._models) / len(self._models)
            return p.argmax(1).numpy()
        
        def predict_proba(self, X):
            self._load_mlps()
            if isinstance(X, (list, tuple)) and len(X) > 0 and isinstance(X[0], str):
                X = np.array([self.extract_features(p) for p in X])
            X_t = torch.tensor(np.array(X), dtype=torch.float32)
            with torch.no_grad():
                p = sum(torch.softmax(m(X_t), dim=1) for m in self._models) / len(self._models)
            return p.numpy()


    class EnsembleWrapper:
        """
        Inference wrapper for MLP ensemble with built-in scaler.
        """
        def __init__(self, model_data, scaler_obj, all_classes_list, nf, nc):
            self.classes_ = np.array(all_classes_list)
            self._model_data = model_data
            self._scaler = scaler_obj
            self._nf = nf
            self._nc = nc
            self._models = None

        def _ensure(self):
            if self._models is None:
                self._models = []
                for state, width in self._model_data:
                    m = DeepMLP(self._nf, self._nc, width=width)
                    m.load_state_dict(state)
                    m.eval()
                    self._models.append(m)

        def predict(self, X):
            self._ensure()
            Xs = self._scaler.transform(np.array(X).astype(np.float32))
            Xt = torch.tensor(Xs)
            probs = []
            with torch.no_grad():
                for m in self._models:
                    probs.append(torch.softmax(m(Xt), 1).numpy())
            return np.mean(probs, axis=0).argmax(axis=1)

        def predict_proba(self, X):
            self._ensure()
            Xs = self._scaler.transform(np.array(X).astype(np.float32))
            Xt = torch.tensor(Xs)
            probs = []
            with torch.no_grad():
                for m in self._models:
                    probs.append(torch.softmax(m(Xt), 1).numpy())
            return np.mean(probs, axis=0)


    class SingleModelEnsemble:
        """
        Inference wrapper for single-backbone MLP ensemble (B0 features only).
        """
        
        def __init__(self, model_states, widths, classes, n_feat):
            self.classes_ = np.array(classes)
            self._n_cls = len(classes)
            self._n_feat = n_feat
            self._states = model_states
            self._widths = widths
            self._models = None
        
        def _load(self):
            if self._models is None:
                self._models = []
                for s, w in zip(self._states, self._widths):
                    m = DeepMLP(self._n_feat, self._n_cls, w)
                    m.load_state_dict(s)
                    m.eval()
                    self._models.append(m)
        
        def predict(self, X):
            self._load()
            X_t = torch.tensor(np.array(X), dtype=torch.float32)
            with torch.no_grad():
                p = sum(torch.softmax(m(X_t), dim=1) for m in self._models) / len(self._models)
            return p.argmax(1).numpy()
        
        def predict_proba(self, X):
            self._load()
            X_t = torch.tensor(np.array(X), dtype=torch.float32)
            with torch.no_grad():
                p = sum(torch.softmax(m(X_t), dim=1) for m in self._models) / len(self._models)
            return p.numpy()
