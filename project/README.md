# 🩺 Disease Prediction System using Machine Learning with Image-Based Symptom Detection

A responsive medical web application that predicts diseases using **user-selected symptoms**, **uploaded medical images**, or **both combined**.

> **⚠ Disclaimer:** This is an educational prototype (final-year major project). It is **not** a substitute for professional medical advice.

---

## 📂 Project Structure

```
project/
├── backend/
│   ├── app.py                    # Flask backend & API endpoints
│   ├── utils.py                  # Image preprocessing & symptom encoding
│   ├── train_symptom_model.py    # Random Forest training script
│   ├── train_image_model.py      # CNN (TensorFlow/Keras) training script
│   ├── symptom_model.pkl         # Trained symptom model (generated)
│   └── image_model.h5            # Trained image model  (generated)
├── datasets/
│   └── symptom_disease.csv       # Symptom→Disease dataset (auto-generated)
├── templates/
│   ├── base.html                 # Jinja2 base layout
│   ├── index.html                # Home page
│   ├── symptoms.html             # Symptom prediction page
│   ├── image.html                # Image detection page
│   ├── combined.html             # Combined prediction page
│   └── about.html                # About & disclaimer
├── static/
│   ├── css/style.css             # Medical-themed stylesheet
│   ├── js/app.js                 # Client-side logic
│   └── uploads/                  # Uploaded images (auto-created)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & install dependencies

```bash
cd project
pip install -r requirements.txt
```

### 2. Train the Symptom Model (Random Forest)

```bash
cd backend
python train_symptom_model.py
```

This generates `symptom_model.pkl` and `datasets/symptom_disease.csv`.

### 3. Train the Image Model (CNN) — optional

```bash
# Demo model (random synthetic data, quick)
python train_image_model.py

# Real dataset (folder with sub-directories per class)
python train_image_model.py --dataset ../datasets/images/
```

This generates `image_model.h5`.

> **Note:** TensorFlow is only required for image predictions. The symptom predictor works without it.

### 4. Run the Flask Server

```bash
cd backend
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 🖥 Pages

| Page | URL | Description |
|------|-----|-------------|
| Home | `/` | Project overview, hero section, quick links |
| Symptoms | `/symptoms` | Multi-select symptom checklist → prediction |
| Image | `/image` | Upload medical image → CNN analysis |
| Combined | `/combined` | Image + symptoms → fused prediction |
| About | `/about` | Technologies, how it works, disclaimer |

---

## ⚙ API Endpoints

| Method | Endpoint | Input | Output |
|--------|----------|-------|--------|
| POST | `/predict_symptoms` | JSON `{ "symptoms": ["cough", "fever"] }` | Disease + confidence + top-3 |
| POST | `/predict_image` | `multipart/form-data` with `image` file | Detected symptom + disease + confidence |
| POST | `/predict_combined` | `multipart/form-data` with `image` + `symptoms` | Fused prediction |

---

## 🧠 Machine Learning

### Symptom Model
- **Algorithm:** Random Forest (200 estimators)
- **Dataset:** 1 000 synthetic samples across 20 diseases, 30 symptoms
- **Metrics:** ~88 % cross-validation accuracy

### Image Model
- **Architecture:** 4-block CNN → GlobalAveragePooling → Dense
- **Framework:** TensorFlow / Keras
- **Input:** 224 × 224 × 3 (resized, normalised)
- **Classes:** skin_rash, acne, eczema, psoriasis, melanoma, eye_infection, tongue_disease, wound_infection, xray_abnormal, healthy
- **Augmentation:** rotation, shift, flip, brightness, zoom

---

## ✨ Features

- 🎨 Medical theme (blue / green / white) with dark-mode toggle
- 📱 Fully responsive (Bootstrap 5)
- 🖼 Drag-and-drop image upload with preview
- 📊 Chart.js confidence visualisation
- ⚡ Loading animations during prediction
- 🔄 Graceful demo mode when models are not trained
- 🛡 Input validation & error messages

---

## 🛠 Technologies

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5, CSS3, JavaScript, Bootstrap 5, Chart.js, Font Awesome |
| Backend | Python 3.10+, Flask, Flask-CORS |
| ML (Symptoms) | scikit-learn (Random Forest) |
| ML (Images) | TensorFlow / Keras (CNN) |
| Data | Pandas, NumPy, Pillow |

---

## 📜 License

MIT — free for educational and personal use.
