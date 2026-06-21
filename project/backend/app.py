"""
Disease Prediction System — Flask Backend
==========================================
Main application serving the web interface and providing REST API
endpoints for disease prediction using symptom data and medical images.

Endpoints:
    GET  /                   → Home page
    GET  /symptoms           → Symptom prediction page
    GET  /image              → Image detection page
    GET  /combined           → Combined prediction page
    GET  /about              → About page
    POST /predict_symptoms   → Predict from symptoms (JSON)
    POST /predict_image      → Predict from image upload
    POST /predict_combined   → Predict from symptoms + image
"""

import os
import json
import pickle
import random
import logging
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from utils import (
    preprocess_image,
    validate_medical_image,
    encode_symptoms,
    get_symptom_list,
    get_image_classes,
    format_prediction_result,
    get_disease_description,
    get_disease_precautions,
    extract_image_features,
    get_ai_cure_assistance,
)

# Optional import for PyTorch-based models (fallback to demo mode if unavailable)
try:
    from train_fast_accurate import ModelWrapper, AccurateClassifier
except ImportError:
    ModelWrapper = None
    AccurateClassifier = None

# ──────────────────────────────────────────────────────────────────────
# App Configuration
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_DIR, "templates"),
    static_folder=os.path.join(PROJECT_DIR, "static"),
)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024          # 16 MB
app.config["UPLOAD_FOLDER"] = os.path.join(PROJECT_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Load ML Models (graceful fallback to demo mode)
# ──────────────────────────────────────────────────────────────────────
SYMPTOM_MODEL = None
LABEL_ENCODER = None
IMAGE_MODEL = None


def load_models():
    """Load pre-trained models from disk if available."""
    global SYMPTOM_MODEL, LABEL_ENCODER, IMAGE_MODEL
    import joblib

    # --- Symptom model (Extra Trees, .pkl via joblib) ---
    symptom_path = os.path.join(BASE_DIR, "symptom_model.pkl")
    if os.path.exists(symptom_path):
        SYMPTOM_MODEL = joblib.load(symptom_path)
        logger.info("✅ Symptom model loaded from %s", symptom_path)
    else:
        logger.warning("⚠️  symptom_model.pkl not found — running in demo mode.")

    # --- Label encoder (.pkl via joblib) ---
    le_path = os.path.join(BASE_DIR, "label_encoder.pkl")
    if os.path.exists(le_path):
        LABEL_ENCODER = joblib.load(le_path)
        logger.info("✅ LabelEncoder loaded from %s", le_path)
    else:
        logger.warning("⚠️  label_encoder.pkl not found.")

    # --- Image model (scikit-learn pipeline, .pkl — joblib compressed) ---
    import sys
    import model_defs  # noqa: F401 – needed for pickle to resolve classes
    # Inject model_defs classes into __main__ for pickle compatibility
    sys.modules['__main__'].ScikitLearnWrapper = model_defs.ScikitLearnWrapper
    
    # Try to inject PyTorch wrapper classes if available
    try:
        from train_accurate_image import PyTorchModelWrapper as AccuratePyTorchWrapper
        sys.modules['__main__'].PyTorchModelWrapper = AccuratePyTorchWrapper
    except (ImportError, Exception):
        pass
    try:
        from finetune_image import PyTorchModelWrapper as FinetunePyTorchWrapper
        sys.modules['__main__'].PyTorchModelWrapper = FinetunePyTorchWrapper
    except (ImportError, Exception):
        pass
    
    image_path = os.path.join(BASE_DIR, "image_model.pkl")
    if os.path.exists(image_path):
        try:
            IMAGE_MODEL = joblib.load(image_path)
            logger.info("✅ Image model loaded from %s", image_path)
        except Exception as e:
            logger.warning("⚠️  Failed to load image model: %s — running in demo mode.", e)
    else:
        logger.warning("⚠️  image_model.pkl not found — running in demo mode.")


load_models()

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def map_image_class_to_disease(class_name: str):
    """Map a CNN class label to (detected_symptom, predicted_disease).

    Works with the 27 real classes from the training datasets.
    """
    mapping = {
        # ── 23 Skin-disease classes ──
        "Acne and Rosacea Photos":                                      ("Acne / Rosacea Lesions",         "Acne / Rosacea"),
        "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": ("Malignant Skin Lesion",   "Actinic Keratosis / Basal Cell Carcinoma"),
        "Atopic Dermatitis Photos":                                     ("Atopic Skin Inflammation",       "Atopic Dermatitis"),
        "Bullous Disease Photos":                                       ("Blisters / Bullae",              "Bullous Disease"),
        "Cellulitis Impetigo and other Bacterial Infections":            ("Bacterial Skin Infection",       "Cellulitis / Impetigo"),
        "Eczema Photos":                                                ("Dry / Flaky Skin",               "Eczema"),
        "Exanthems and Drug Eruptions":                                  ("Drug Eruption / Rash",           "Exanthem / Drug Eruption"),
        "Hair Loss Photos Alopecia and other Hair Diseases":             ("Hair Loss / Alopecia",           "Alopecia"),
        "Herpes HPV and other STDs Photos":                              ("Viral Skin Lesion",              "Herpes / HPV"),
        "Light Diseases and Disorders of Pigmentation":                  ("Pigmentation Disorder",          "Vitiligo / Pigmentation Disorder"),
        "Lupus and other Connective Tissue diseases":                    ("Connective Tissue Inflammation", "Lupus"),
        "Melanoma Skin Cancer Nevi and Moles":                           ("Abnormal Mole / Lesion",         "Melanoma (suspected)"),
        "Nail Fungus and other Nail Disease":                            ("Nail Abnormality / Fungus",      "Nail Fungus"),
        "Poison Ivy Photos and other Contact Dermatitis":                ("Contact Dermatitis Rash",        "Contact Dermatitis"),
        "Psoriasis pictures Lichen Planus and related diseases":         ("Scaly Patches / Plaques",        "Psoriasis / Lichen Planus"),
        "Scabies Lyme Disease and other Infestations and Bites":         ("Infestation / Bite Marks",       "Scabies / Lyme Disease"),
        "Seborrheic Keratoses and other Benign Tumors":                  ("Benign Skin Growth",             "Seborrheic Keratosis"),
        "Systemic Disease":                                             ("Systemic Skin Manifestation",    "Systemic Disease"),
        "Tinea Ringworm Candidiasis and other Fungal Infections":        ("Fungal Skin Infection",          "Tinea / Ringworm / Candidiasis"),
        "Urticaria Hives":                                              ("Hives / Urticaria",              "Urticaria"),
        "Vascular Tumors":                                              ("Vascular Skin Growth",           "Vascular Tumor"),
        "Vasculitis Photos":                                            ("Vasculitis Inflammation",        "Vasculitis"),
        "Warts Molluscum and other Viral Infections":                    ("Warts / Viral Growth",           "Warts / Molluscum"),
        # ── 4 Eye-disease classes ──
        "cataract":                  ("Lens Opacity / Clouding",     "Cataract"),
        "diabetic_retinopathy":      ("Retinal Damage",              "Diabetic Retinopathy"),
        "glaucoma":                  ("Optic Nerve Damage",          "Glaucoma"),
        "normal":                    ("No abnormality detected",     "Healthy Eye"),
        # ── HAM10000 (archive 3) — merged to existing + 1 new ──
        "Dermatofibroma":            ("Firm Skin Nodule",            "Dermatofibroma"),
    }
    return mapping.get(class_name, (class_name, class_name))


# ──────────────────────────────────────────────────────────────────────
# Demo / Fallback Predictions
# ──────────────────────────────────────────────────────────────────────

def demo_symptom_prediction(symptoms: list) -> dict:
    """Return a plausible prediction when the real model is not trained."""
    disease_map = {
        frozenset(["itching", "skin_rash"]):                       ("Fungal Infection", 87.5),
        frozenset(["continuous_sneezing", "shivering", "chills"]):  ("Allergy", 82.3),
        frozenset(["cough", "high_fever", "fatigue"]):              ("Pneumonia", 78.1),
        frozenset(["high_fever", "headache", "muscle_pain"]):       ("Malaria", 84.6),
        frozenset(["joint_pain", "fatigue"]):                       ("Arthritis", 76.2),
        frozenset(["stomach_pain", "vomiting", "nausea"]):          ("Gastroenteritis", 81.4),
        frozenset(["headache", "dizziness", "nausea"]):             ("Migraine", 79.8),
        frozenset(["cough", "sore_throat"]):                        ("Common Cold", 85.3),
        frozenset(["high_fever", "cough", "shortness_of_breath"]):  ("Tuberculosis", 73.9),
        frozenset(["chest_pain", "shortness_of_breath"]):           ("Bronchitis", 77.5),
    }

    symptom_set = set(symptoms)
    best_match, best_overlap = None, 0
    for keys, value in disease_map.items():
        overlap = len(symptom_set & keys)
        if overlap > best_overlap:
            best_overlap = overlap
            best_match = value

    if best_match is None:
        best_match = (random.choice(["Common Cold", "Allergy", "Viral Fever"]),
                      round(random.uniform(60, 85), 2))

    # Get AI cure assistance
    cure_assistance = get_ai_cure_assistance(best_match[0])
    
    return {
        "success": True,
        "prediction": best_match[0],
        "confidence": best_match[1],
        "top_predictions": [
            {"disease": best_match[0], "confidence": best_match[1]},
            {"disease": "Viral Fever",  "confidence": round(best_match[1] - 15, 2)},
            {"disease": "Common Cold",  "confidence": round(best_match[1] - 25, 2)},
        ],
        "symptoms_used": symptoms,
        "note": "Demo mode — train models for real predictions.",
        "cure_assistance": cure_assistance,
    }


def demo_image_prediction(filename: str) -> dict:
    """Return a plausible image prediction when the CNN is not trained."""
    choices = [
        {"detected_symptom": "Skin Rash / Lesion",   "prediction": "Eczema",       "confidence": 82.4},
        {"detected_symptom": "Acne / Pustules",       "prediction": "Acne Vulgaris","confidence": 78.9},
        {"detected_symptom": "Redness / Inflammation", "prediction": "Dermatitis",  "confidence": 75.6},
        {"detected_symptom": "Fungal Patches",         "prediction": "Ringworm",    "confidence": 80.1},
        {"detected_symptom": "Discoloration",          "prediction": "Psoriasis",   "confidence": 71.3},
    ]
    result = random.choice(choices)
    result["success"] = True
    result["note"] = "Demo mode — train CNN model for real predictions."
    
    # Get AI cure assistance
    result["cure_assistance"] = get_ai_cure_assistance(result["prediction"])
    
    return result


def combine_predictions(symptom_result: dict | None, image_result: dict | None) -> dict:
    """Weighted combination: 60 % symptom · 40 % image."""
    if symptom_result and image_result:
        s_conf = symptom_result.get("confidence", 0)
        i_conf = image_result.get("confidence", 0)
        if s_conf >= i_conf:
            primary = symptom_result.get("prediction", symptom_result.get("disease", "Unknown"))
            combined = round(s_conf * 0.6 + i_conf * 0.4, 2)
        else:
            primary = image_result.get("prediction", image_result.get("disease", "Unknown"))
            combined = round(i_conf * 0.6 + s_conf * 0.4, 2)
        return {"disease": primary, "confidence": combined, "method": "Combined (Symptom + Image)"}
    elif symptom_result:
        return {
            "disease": symptom_result.get("prediction", symptom_result.get("disease", "Unknown")),
            "confidence": symptom_result.get("confidence", 0),
            "method": "Symptom-based only",
        }
    elif image_result:
        return {
            "disease": image_result.get("prediction", image_result.get("disease", "Unknown")),
            "confidence": image_result.get("confidence", 0),
            "method": "Image-based only",
        }
    return {"disease": "Unable to predict", "confidence": 0, "method": "No input provided"}


# ──────────────────────────────────────────────────────────────────────
# Page Routes
# ──────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/symptoms")
def symptoms_page():
    return render_template("symptoms.html", symptoms=get_symptom_list())


@app.route("/image")
def image_page():
    return render_template("image.html")


@app.route("/combined")
def combined_page():
    return render_template("combined.html", symptoms=get_symptom_list())


@app.route("/about")
def about_page():
    return render_template("about.html")


# ──────────────────────────────────────────────────────────────────────
# API — Symptom Prediction
# ──────────────────────────────────────────────────────────────────────

@app.route("/predict_symptoms", methods=["POST"])
def predict_symptoms():
    """Predict disease from a list of selected symptoms (JSON body)."""
    try:
        data = request.get_json(force=True)
        symptoms = data.get("symptoms", [])
        if not symptoms:
            return jsonify({"error": "No symptoms provided."}), 400

        # --- Real model ---
        if SYMPTOM_MODEL is not None:
            feature_vector = encode_symptoms(symptoms)
            pred_encoded = SYMPTOM_MODEL.predict([feature_vector])[0]
            probabilities = SYMPTOM_MODEL.predict_proba([feature_vector])[0]
            confidence = float(max(probabilities)) * 100

            # Decode label
            if LABEL_ENCODER is not None:
                prediction = LABEL_ENCODER.inverse_transform([pred_encoded])[0]
                all_classes = LABEL_ENCODER.classes_
            else:
                prediction = str(pred_encoded)
                all_classes = SYMPTOM_MODEL.classes_

            top_idx = np.argsort(probabilities)[::-1][:5]
            top_predictions = []
            for i in top_idx:
                name = str(all_classes[i]) if LABEL_ENCODER is not None else str(SYMPTOM_MODEL.classes_[i])
                top_predictions.append({
                    "disease": name,
                    "confidence": round(float(probabilities[i]) * 100, 2),
                })

            # Get AI cure assistance
            cure_assistance = get_ai_cure_assistance(str(prediction))
            
            return jsonify({
                "success": True,
                "prediction": str(prediction),
                "confidence": round(confidence, 2),
                "top_predictions": top_predictions,
                "symptoms_used": symptoms,
                "description": get_disease_description(str(prediction)),
                "precautions": get_disease_precautions(str(prediction)),
                "cure_assistance": cure_assistance,
            })

        # --- Demo mode ---
        return jsonify(demo_symptom_prediction(symptoms))

    except Exception as exc:
        logger.error("Symptom prediction error: %s", exc)
        return jsonify({"error": str(exc)}), 500


# ──────────────────────────────────────────────────────────────────────
# API — Image Prediction
# ──────────────────────────────────────────────────────────────────────

@app.route("/predict_image", methods=["POST"])
def predict_image():
    """Predict disease from an uploaded medical image."""
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded."}), 400
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected."}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Use JPG or PNG."}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        image_url = f"/static/uploads/{filename}"

        # --- Real model ---
        if IMAGE_MODEL is not None:
            is_valid, msg = validate_medical_image(filepath)
            if not is_valid:
                return jsonify({"error": msg, "image_url": image_url}), 400

            # Multi-model: pass filepath; single-model: pass features
            if hasattr(IMAGE_MODEL, '_backbone_names'):
                model_input = [filepath]
            else:
                model_input = [extract_image_features(filepath)]
            # Prefer class order embedded in the model to avoid mismatch.
            classes = getattr(IMAGE_MODEL, "classes_", None)
            if classes is None:
                classes = get_image_classes()
            else:
                classes = list(classes)

            if hasattr(IMAGE_MODEL, 'predict_proba'):
                probs = IMAGE_MODEL.predict_proba(model_input)[0]
                idx = int(np.argmax(probs))
                conf = float(probs[idx]) * 100
            else:
                pred = IMAGE_MODEL.predict(model_input)[0]
                idx = int(pred)
                conf = 85.0  # SVM has no probabilities

            detected, disease = map_image_class_to_disease(classes[idx])
            
            # Get AI cure assistance
            cure_assistance = get_ai_cure_assistance(disease)
            
            return jsonify({
                "success": True,
                "detected_symptom": detected,
                "prediction": disease,
                "confidence": round(conf, 2),
                "image_class": classes[idx],
                "image_url": image_url,
                "cure_assistance": cure_assistance,
            })

        # --- Demo mode ---
        result = demo_image_prediction(filename)
        result["image_url"] = image_url
        return jsonify(result)

    except Exception as exc:
        logger.error("Image prediction error: %s", exc)
        return jsonify({"error": str(exc)}), 500


# ──────────────────────────────────────────────────────────────────────
# API — Combined Prediction
# ──────────────────────────────────────────────────────────────────────

@app.route("/predict_combined", methods=["POST"])
def predict_combined():
    """Predict disease using both symptoms and an uploaded image."""
    try:
        symptoms = request.form.getlist("symptoms")

        # ---- Image part ----
        image_result = None
        if "image" in request.files and request.files["image"].filename:
            file = request.files["image"]
            if allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)
                img_url = f"/static/uploads/{filename}"

                if IMAGE_MODEL is not None:
                    if hasattr(IMAGE_MODEL, '_backbone_names'):
                        model_input = [filepath]
                    else:
                        model_input = [extract_image_features(filepath)]
                    classes = getattr(IMAGE_MODEL, "classes_", None)
                    if classes is None:
                        classes = get_image_classes()
                    else:
                        classes = list(classes)
                    if hasattr(IMAGE_MODEL, 'predict_proba'):
                        probs = IMAGE_MODEL.predict_proba(model_input)[0]
                        idx = int(np.argmax(probs))
                        conf = float(probs[idx]) * 100
                    else:
                        pred = IMAGE_MODEL.predict(model_input)[0]
                        idx = int(pred)
                        conf = 85.0
                    detected, img_disease = map_image_class_to_disease(classes[idx])
                    image_result = {
                        "detected_symptom": detected,
                        "disease": img_disease,
                        "confidence": round(conf, 2),
                        "image_url": img_url,
                    }
                else:
                    image_result = demo_image_prediction(filename)
                    image_result["image_url"] = img_url

        # ---- Symptom part ----
        symptom_result = None
        if symptoms:
            if SYMPTOM_MODEL is not None:
                fv = encode_symptoms(symptoms)
                pred_enc = SYMPTOM_MODEL.predict([fv])[0]
                probs = SYMPTOM_MODEL.predict_proba([fv])[0]
                if LABEL_ENCODER is not None:
                    disease_name = LABEL_ENCODER.inverse_transform([pred_enc])[0]
                else:
                    disease_name = str(pred_enc)
                symptom_result = {
                    "disease": disease_name,
                    "confidence": round(float(max(probs)) * 100, 2),
                    "description": get_disease_description(disease_name),
                    "precautions": get_disease_precautions(disease_name),
                }
            else:
                symptom_result = demo_symptom_prediction(symptoms)

        # ---- Combine ----
        final = combine_predictions(symptom_result, image_result)
        
        # Get AI cure assistance for the final combined disease
        cure_assistance = get_ai_cure_assistance(final.get("disease", "Unknown"))
        
        return jsonify({
            "success": True,
            "symptom_result": symptom_result,
            "image_result": image_result,
            "combined_prediction": final,
            "cure_assistance": cure_assistance,
        })

    except Exception as exc:
        logger.error("Combined prediction error: %s", exc)
        return jsonify({"error": str(exc)}), 500


# ──────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5005)
