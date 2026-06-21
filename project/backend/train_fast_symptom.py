#!/usr/bin/env python3
"""
Fast High-Accuracy Symptom-Based Disease Prediction Training
Uses optimized ensemble with faster training time
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier, 
    ExtraTreesClassifier,
    VotingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Set paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, '..', 'datasets')
DATASET_PATH = os.path.join(DATASET_DIR, 'dataset.csv')

def load_and_prepare_data():
    """Load and prepare the symptom-disease dataset."""
    df = pd.read_csv(DATASET_PATH)
    
    print(f"📂 Loaded {len(df)} rows from dataset.csv")
    
    # Get symptom columns (all columns except 'Disease' and unnamed)
    symptom_cols = [c for c in df.columns if c not in ['Disease', 'Unnamed: 133'] and not c.startswith('Unnamed')]
    
    # Get unique symptoms across all symptom columns
    all_symptoms = set()
    for col in symptom_cols:
        unique_vals = df[col].dropna().unique()
        all_symptoms.update([s.strip() for s in unique_vals if isinstance(s, str) and s.strip()])
    
    all_symptoms = sorted(list(all_symptoms))
    print(f"   Unique symptoms : {len(all_symptoms)}")
    
    # Create binary symptom matrix
    symptom_to_idx = {s: i for i, s in enumerate(all_symptoms)}
    X = np.zeros((len(df), len(all_symptoms)), dtype=np.float32)
    
    for row_idx, row in df.iterrows():
        for col in symptom_cols:
            symptom = row[col]
            if isinstance(symptom, str) and symptom.strip():
                symptom = symptom.strip()
                if symptom in symptom_to_idx:
                    X[row_idx, symptom_to_idx[symptom]] = 1.0
    
    # Encode diseases
    le = LabelEncoder()
    y = le.fit_transform(df['Disease'])
    
    print(f"   Unique diseases : {len(le.classes_)}")
    print(f"   Total rows      : {len(df)}")
    
    return X, y, le, all_symptoms

def augment_data(X, y, copies=3):
    """Light augmentation by dropping random symptoms."""
    augmented_X = [X]
    augmented_y = [y]
    
    for _ in range(copies):
        X_aug = X.copy()
        # Random mask: keep 70-100% of symptoms per sample
        mask = np.random.random(X_aug.shape) > 0.15
        X_aug = X_aug * mask
        augmented_X.append(X_aug)
        augmented_y.append(y.copy())
    
    return np.vstack(augmented_X), np.concatenate(augmented_y)

def main():
    print("=" * 70)
    print("  FAST High-Accuracy Symptom-Based Disease Prediction Training")
    print("=" * 70)
    print()
    
    # Load data
    X, y, label_encoder, symptom_list = load_and_prepare_data()
    
    n_classes = len(label_encoder.classes_)
    print(f"\n🎯 Number of disease classes: {n_classes}")
    
    # Light augmentation
    print("\n🔄 Applying light data augmentation...")
    X_aug, y_aug = augment_data(X, y, copies=3)
    print(f"   After augmentation: {len(X_aug)} samples")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_aug, y_aug, test_size=0.15, random_state=42, stratify=y_aug
    )
    print(f"\n📊 Train: {len(X_train)}  |  Test: {len(X_test)}")
    
    # Train classifiers
    print("\n" + "=" * 70)
    print("  Training Classifiers")
    print("=" * 70)
    
    classifiers = {}
    
    # Random Forest
    print("\n🌲 Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=1,  # Single threaded to avoid multiprocessing issues
        random_state=42
    )
    rf.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"   Test Accuracy: {acc * 100:.2f}%")
    classifiers['rf'] = rf
    
    # Extra Trees (fast and accurate for this type of data)
    print("\n🌲 Training Extra Trees...")
    et = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=1,  # Single threaded to avoid multiprocessing issues
        random_state=42
    )
    et.fit(X_train, y_train)
    acc = accuracy_score(y_test, et.predict(X_test))
    print(f"   Test Accuracy: {acc * 100:.2f}%")
    classifiers['et'] = et
    
    # MLP Neural Network (fast with limited iterations)
    print("\n🧠 Training MLP Neural Network...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )
    mlp.fit(X_train, y_train)
    acc = accuracy_score(y_test, mlp.predict(X_test))
    print(f"   Test Accuracy: {acc * 100:.2f}%")
    classifiers['mlp'] = mlp
    
    # Create Voting Ensemble
    print("\n" + "=" * 70)
    print("  Creating Voting Ensemble")
    print("=" * 70)
    
    ensemble = VotingClassifier(
        estimators=[
            ('rf', classifiers['rf']),
            ('et', classifiers['et']),
            ('mlp', classifiers['mlp'])
        ],
        voting='soft',
        n_jobs=1  # Single threaded to avoid multiprocessing issues
    )
    
    # Fit ensemble (required for VotingClassifier)
    print("\n🔧 Fitting ensemble...")
    ensemble.fit(X_train, y_train)
    
    # Evaluate
    y_pred = ensemble.predict(X_test)
    ensemble_acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Ensemble Test Accuracy: {ensemble_acc * 100:.2f}%")
    
    # Classification report
    print("\n" + "=" * 70)
    print("  Classification Report (Top diseases)")
    print("=" * 70)
    print(classification_report(y_test, y_pred, 
                                target_names=label_encoder.classes_,
                                zero_division=0))
    
    # Save models
    print("\n" + "=" * 70)
    print("  Saving Models")
    print("=" * 70)
    
    model_path = os.path.join(SCRIPT_DIR, 'symptom_model.pkl')
    encoder_path = os.path.join(SCRIPT_DIR, 'label_encoder.pkl')
    meta_path = os.path.join(SCRIPT_DIR, 'symptom_meta.pkl')
    
    joblib.dump(ensemble, model_path)
    joblib.dump(label_encoder, encoder_path)
    joblib.dump({
        'symptom_list': symptom_list,
        'n_classes': n_classes,
        'accuracy': ensemble_acc
    }, meta_path)
    
    print(f"\n✅ Model saved to: {model_path}")
    print(f"✅ Encoder saved to: {encoder_path}")
    print(f"✅ Metadata saved to: {meta_path}")
    
    print("\n" + "=" * 70)
    print(f"  TRAINING COMPLETE - Final Accuracy: {ensemble_acc * 100:.2f}%")
    print("=" * 70)
    
    return ensemble_acc

if __name__ == "__main__":
    main()
