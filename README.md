<<<<<<< HEAD
# 💊 UmutiSafe — Medicine Classification & Disposal API

This repository contains the FastAPI backend for UmutiSafe — an AI-powered system that extracts text from medicine labels (OCR) and predicts safe disposal categories and risk levels for medicines.

This README provides developer-focused instructions: how the models work, how to run and test the app locally, how it was deployed, and recommended improvements.

Table of contents
- Overview
- Model internals (how it works)
- Quick start (run locally)
- Tests and smoke checks
- Deployment (Docker & Render)
- Repo cleanup and .gitignore notes
- Improvements & roadmap
- Where to find more detailed docs

---

## Overview

The FastAPI app exposes endpoints for:

- Image-based prediction: `POST /api/predict/image`
- Text-based prediction: `POST /api/predict/text`
- Guidelines: `GET /api/guidelines`
- Health checks: `GET /health` and `GET /api/health`

The app integrates an OCR processor (`enhanced_ocr_processor.py`) and a prediction engine (`medicine_disposal_predictor.py`) and serves model artifacts from the repository by default (or downloads from S3 when configured).

---

## Model internals — how the models work

High-level pipeline:

1. OCR: `enhanced_ocr_processor.py` (EasyOCR-based) extracts text blocks from images and attempts to find fields like generic name, brand name, dosage form, and dosage strength.
2. Feature construction: The predictor combines text fields into a single `combined_text` string and computes TF-IDF features (using `tfidf_vectorizer.pkl`) plus simple text statistics:
   - `word_count`
   - `char_count`
   - `special_chars`
3. Feature matrix: TF-IDF vector (sparse) is converted to dense and horizontally stacked with the numeric text statistics to form the final feature matrix `X`.
4. Predictions:
   - Category model (`best_category_model.pkl`) predicts one of the disposal categories (e.g., Solids, Liquids, Semisolids, Aerosols, Biological Waste).
   - Risk model (`best_risk_model.pkl`) predicts a risk level (Low/Medium/High).
   - Label encoders (`le_category.pkl`, `le_risk.pkl`) are used to map encoded labels back to human-readable classes.

Files involved:

- `best_category_model.pkl`, `best_risk_model.pkl`: trained sklearn-compatible estimators saved with `joblib`.
- `tfidf_vectorizer.pkl`: TF-IDF vectorizer saved with `joblib`.
- `le_category.pkl`, `le_risk.pkl`: sklearn LabelEncoder objects saved with `joblib`.

Inputs / Outputs (contract):

- Input (image): multipart/form-data field `file` — image bytes. The OCR extracts text and the predictor returns classification and guidance.
- Input (text): form fields `generic_name`, `brand_name`, `dosage_form`, `packaging_type`.
- Output (JSON): contains `success`, `medicine_info`, `predictions` (category, risk, confidence, all_probabilities), `safety_guidance` (text and structured fields), and `ocr_info` for image calls.

Edge cases and error handling:

- If OCR fails, the `/api/predict/image` endpoint returns a structured failure with `success: False` and an error message.
- If model artifacts are missing, the app attempts to download them from S3 when `MODELS_S3_BUCKET` is set; otherwise startup reports missing model files and the service marks itself degraded.

---

## Quick start — run locally (developer)

Prerequisites

- Python 3.11
- pip
- Optional: Docker (for containerized runs)

Commands

1. Create & activate virtualenv (recommended):

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the server locally:

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

3. Try endpoints:

```powershell
curl -X GET http://127.0.0.1:8000/health
curl -X POST "http://127.0.0.1:8000/api/predict/text" -F "generic_name=Amoxicillin 500mg" -F "brand_name=Amoxil" -F "dosage_form=Tablet"
```

Notes

- The repo defaults to loading model files from the repository root (`./`), `./models/`, or `./app/models/`. If your files are elsewhere, either update `load_system_components()` or set S3 env vars and let the startup downloader fetch them.

---

## Tests and smoke checks

Automated tests and quick checks included in the repo:

- `smoke_test_load_models.py`: quick script to test loading model artifacts locally.
- `test_api.py`, `test_image_upload.py`: pytest-based tests that exercise endpoints (requires test server or mocking).

Run tests locally:

```powershell
# Make sure server isn't binding to the same port (tests may start a test client)
pytest -q
```

If you want a simple smoke script (curl-based), use the examples in the Quick start section.

---

## Deployment

Docker & Render (what we implemented)

- The repo includes a `Dockerfile` that installs system packages required by OpenCV (libGL, libglib, etc.) and then `pip install -r requirements.txt`.
- Model files are copied into the image if committed to the repo. To avoid large images, you can store model artifacts in S3 and set these environment variables on the host/service:

  - `MODELS_S3_BUCKET` (required when using S3 downloader)
  - `MODELS_S3_PREFIX` (optional)
  - `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (or use an IAM role on the host)

- Health check path for Render: `/health` (or `/api/health`).

Render-specific notes

- Ensure Render's build picks up `requirements.txt` and the Dockerfile. We fixed prior issues in build logs by cleaning `requirements.txt` and installing `boto3` to support S3.
- The Dockerfile now installs system libraries required for OpenCV so `import cv2` succeeds in the runtime.

Vercel frontend

- The frontend preview (Vercel) calls the backend health endpoint. We added explicit CORS entries for the preview domain and the production frontend:`https://umuti-safe-app.vercel.app`.

---

## Repo cleanup & `.gitignore`

The repository now includes a strengthened `.gitignore` that excludes:

- virtualenv directories: `env/`, `venv/`, `.venv/`
- Python caches: `__pycache__/`, `*.pyc`
- environment files: `.env`
- model artifacts and `*.pkl` (so future commits don't accidentally re-add large models)

Important: adding files to `.gitignore` does not remove files already tracked by git. If you want to remove model files from the repo history and keep them out of future commits, run locally:

```powershell
git rm --cached best_category_model.pkl best_risk_model.pkl le_category.pkl le_risk.pkl tfidf_vectorizer.pkl
git commit -m "Remove model artifacts from repo (keep them in S3 or external storage)"
git push origin main
```

Use the `APP_DOCUMENTATION.md` file (in this repo) for step-by-step details and troubleshooting logs.

---

## Improvements & Roadmap

Short-term

- Move model artifacts to S3 and load at startup to keep Docker images small.
- Add batching and async inference for higher throughput.
- Add structured logging and Sentry integration for runtime monitoring.

Medium-term

- Add GPU support (CUDA) for faster OCR and inference when available.
- Add more tests for OCR edge cases (blurry text, rotated labels, multiple languages).
- Add Canary deployments and blue/green rollouts on Render or another platform.

Long-term

- Build a management UI for adding/removing model versions and for human-in-the-loop feedback.

---

## Where to find more details

- App internals and run/test/deploy checklist: `APP_DOCUMENTATION.md` (new file in repo)
- Source files:
  - `main.py` — FastAPI entrypoint and request routing
  - `enhanced_ocr_processor.py` — OCR extraction and image preprocessing
  - `medicine_disposal_predictor.py` — prediction pipeline and model glue
  - `disposal_guidelines_db.py` — disposal guideline content

If you want, I can also add a short `deploy.md` with the exact Render UI steps and screenshots, or a GitHub Action workflow to build and push images automatically.

---

If you'd like me to commit/remove tracked model files from the git history (or create a Git LFS setup), tell me which option you prefer and I will prepare the exact commands or automate the change.

---

Thank you — ready to help with the next step (clean history, move models to S3, add CI/CD, or add a compact front-end health-check script).

Sure — here’s a refined and polished version of your **README.md** with clearer structure, improved formatting, consistent tone, and a few readability and professionalism tweaks:

---

=======
>>>>>>> 5b48d510c9b5db75710a727c68e5e7960d892365
# 💊 UmutiSafe — Intelligent Medicine Disposal Classification & Guidance System

## 🧠 Overview

**UmutiSafe** is an AI-powered system that classifies and recommends safe medicine disposal methods using **computer vision** and **machine learning**.
It extracts text from medicine labels through OCR (Optical Character Recognition) and uses predictive models to determine appropriate disposal categories and risk levels — reducing **environmental contamination** and **public health risks**.

**GitHub Repository:** [https://github.com/Jules-gatete/Mission_Capstone](https://github.com/Jules-gatete/Mission_Capstone)

---

## ⚡ Quick Start

### ✅ Prerequisites

* Python `3.11.9`
* pip package manager

### 🧩 Installation

```bash
# Clone the repository
git clone https://github.com/Jules-gatete/Mission_Capstone.git
cd Mission_Capstone

# Install dependencies
pip install -r requirements.txt

# Run the API
python run.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 🌐 Access the API

* **API Docs (Swagger UI):** [https://mission-capstone.onrender.com/docs](https://mission-capstone.onrender.com/docs)
* **Health Check Endpoint:** [https://mission-capstone.onrender.com/health](https://mission-capstone.onrender.com/health)

---

## 🚀 Key Features

### 🔍 Smart OCR Processing

* Extracts and annotates text from medicine labels
* Detects key pharmaceutical terms
* Handles multiple image formats
* Integrates OCR results for AI inference

### 🤖 AI-Powered Classification

* **Dual Prediction Models:**

  * **Disposal Category:** 5 predefined classes
  * **Risk Level:** 3 levels — *Low, Medium, High*
* Accepts both image and text inputs
* Batch processing support
* Provides confidence scoring for predictions

### 🧴 Safety & Disposal Guidance

* Category-based disposal recommendations
* Risk-sensitive handling instructions
* Highlighted prohibitions and safety alerts

---

## 🧪 API Usage Examples

### 1. 🖼️ Image-Based Prediction

```bash
curl -X POST "http://localhost:8000/api/predict/image" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@medicine_label.jpg"
```

### 2. 🧾 Text-Based Prediction

```bash
curl -X POST "http://localhost:8000/api/predict/text" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "generic_name=Amoxicillin 500mg&brand_name=Amoxil&dosage_form=Capsules"
```

### 3. 📦 Batch Prediction

```bash
curl -X POST "http://localhost:8000/api/predict/batch" \
  -H "Content-Type: application/json" \
  -d '[
    {"generic_name": "Amoxicillin 500mg", "brand_name": "Amoxil", "dosage_form": "Capsules"},
    {"generic_name": "Paracetamol 500mg", "brand_name": "Panadol", "dosage_form": "Tablets"}
  ]'
```

---

## 🧭 Project Structure

```
umutisafe/
├── main.py                        # FastAPI entry point
├── run.py                         # Application runner
├── enhanced_ocr_processor.py      # OCR extraction & preprocessing logic
├── medicine_disposal_predictor.py # Core ML prediction engine
├── requirements.txt               # Dependencies list
                        # Pretrained models
│── best_category_model.pkl
│── best_risk_model.pkl
│── le_category.pkl
│── le_risk.pkl
│── tfidf_vectorizer.pkl
└── disposal_guidelines_db.py
```

---

## 🎥 Demo

🎬 [Watch the demo video](https://youtu.be/BR4Ove8GRzY)

---

## ☁️ Deployment Plan

### Phase 1 — API Deployment

* FastAPI backend with interactive Swagger UI
* Dockerized & cloud-ready deployment
* Scalable architecture for AI inference

### Phase 2 — Web Interface

* Interactive frontend for uploading images
* Real-time classification visualization
* Responsive design for mobile & desktop
* Integration with CHWs (Community Health Workers) for take-back guidance

---

## ⚙️ Technical Details

### Machine Learning Components

* **Text Feature Extraction:** TF-IDF Vectorizer
* **Category Prediction:** Ensemble Classifier
* **Risk Assessment:** Multi-class Classifier
* **OCR Engine:** EasyOCR with custom pre-processing pipeline

### Supported Medicine Forms

* Tablets, Capsules, Injections
* Solutions, Suspensions, Creams
* Syrups, Drops, Sprays
* Ampoules, Vials, Patches

---

## 📊 Performance Highlights

| Metric                   | Result                                   |
| ------------------------ | ---------------------------------------- |
| OCR Accuracy             | >85% on medicine labels                  |
| Classification Accuracy  | >90% for trained classes                 |
| Risk Assessment Accuracy | >85% alignment with regulatory standards |
| Average Processing Time  | <5 seconds per image                     |

**Data Sources:**

* Rwanda FDA Medicine Dataset
* Open-source OCR & ML libraries
* Validation with healthcare professionals

---

### 🩺 “Making medicine disposal safer, one classification at a time.”
