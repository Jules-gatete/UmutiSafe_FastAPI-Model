# 💊 UmutiSafe — Intelligent Medicine Disposal Classification & Guidance System

##  Overview

**UmutiSafe** is an AI-powered system that classifies medicines and recommends safe disposal methods using **computer vision**, **OCR**, and **machine learning**.  
It extracts text from medicine labels, predicts disposal categories and risk levels, and provides safety guidance to reduce **environmental contamination** and **public health risks**.

This repository contains the **FastAPI backend**, including OCR processing, ML prediction pipelines, API endpoints, deployment instructions, and development tooling.

---

## Table of Contents

1. Overview
2. API Endpoints
3. Model Architecture & Pipeline
4. Quick Start (Run Locally)
5. Testing
6. Deployment (Docker & Render)
7. Repo Cleanup & `.gitignore`
8. Improvements & Roadmap
9. Project Structure
10. API Usage Examples
11. Key Features
12. Performance Metrics
13. Additional Docs / Links

---

## 🔗 API Endpoints

| Type | Endpoint | Description |
|------|----------|-------------|
| `POST` | `/api/predict/image` | Predict from uploaded image (OCR + ML) |
| `POST` | `/api/predict/text` | Predict from text fields |
| `GET` | `/api/guidelines` | Get disposal guidelines |
| `GET` | `/health` / `/api/health` | Health check |

---

##  Model Architecture — How It Works

**Pipeline Overview**

1. **OCR Extraction**  
   `enhanced_ocr_processor.py` (EasyOCR) extracts medicine text (brand, generic name, dosage, etc.)

2. **Feature Engineering**  
   - TF-IDF (`tfidf_vectorizer.pkl`)
   - Word / char counts
   - Special character count  
   → Combined into the final feature matrix

3. **Dual ML Models**
   - `best_category_model.pkl` → Disposal Category (Solids, Liquids, etc.)
   - `best_risk_model.pkl` → Risk Level (Low, Medium, High)

4. **Encoders**
   - `le_category.pkl`
   - `le_risk.pkl`

**Inputs / Outputs**

| Input Type | Payload | Output |
|------------|---------|---------|
| Image | `file=@image.jpg` | OCR + prediction + safety guidance |
| Text | `generic_name`, `brand_name`, `dosage_form`, `packaging_type` | Prediction + guidance |

---

## ⚡ Quick Start — Run Locally

### ✅ Prerequisites
- Python **3.11**
- `pip`
- *(optional)* Docker

### ▶️ Setup & Run

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
pip install -r requirements.txt

uvicorn main:app --reload --host 0.0.0.0 --port 8000
````

### Test API

```powershell
curl -X GET http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/api/predict/text `
-F "generic_name=Amoxicillin 500mg" `
-F "brand_name=Amoxil" `
-F "dosage_form=Tablet"
```

---

## 🧪 Tests & Smoke Checks

```powershell
pytest -q
```

Included test scripts:

* `smoke_test_load_models.py` – verifies model loading
* `test_api.py`, `test_image_upload.py` – endpoint tests

---

## ☁️ Deployment

### Docker & Render

* Prebuilt `Dockerfile` with OpenCV dependencies
* Add model files OR load from S3 on startup

Required env vars when using S3:

```
MODELS_S3_BUCKET=
MODELS_S3_PREFIX=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
```

Health check path: `/health`

### Frontend (Vercel)

CORS allowed:

```
https://umuti-safe-app.vercel.app
```


## 🚀 Key Features

### 🔍 OCR Processing

* Extract text from medicine labels
* Supports multiple formats (JPG, PNG, PDF images)
* Annotates extracted blocks

### 🤖 AI Classification

* 5 Disposal Categories
* 3 Risk Levels (Low / Medium / High)
* Accepts text or images
* Confidence + probability outputs

### 🧴 Disposal Guidance

* Safety notes
* Risk-based recommendations
* Prohibited actions highlighted

---

## 📊 Performance Highlights

| Metric              | Result        |
| ------------------- | ------------- |
| OCR Accuracy        | > 85%         |
| Category Prediction | > 90%         |
| Risk Assessment     | > 85%         |
| Avg Processing Time | < 5 sec/image |

**Datasets Used:**
Rwanda FDA Medicine Dataset + verified medical sources

---

## 🗃️ Project Structure

```
umutisafe/
├── main.py
├── run.py
├── enhanced_ocr_processor.py
├── medicine_disposal_predictor.py
├── disposal_guidelines_db.py
├── requirements.txt
│── best_category_model.pkl
│── best_risk_model.pkl
│── le_category.pkl
│── le_risk.pkl
└── tfidf_vectorizer.pkl
```

---

## 🧪 API Usage Examples

Image Prediction:

```bash
curl -X POST "http://localhost:8000/api/predict/image" \
  -F "file=@medicine.jpg"
```

Text Prediction:

```bash
curl -X POST "http://localhost:8000/api/predict/text" \
  -d "generic_name=Paracetamol 500mg&brand_name=Panadol&dosage_form=Tablets"
```

Batch:

```bash
curl -X POST "http://localhost:8000/api/predict/batch" \
  -H "Content-Type: application/json" \
  -d '[{"generic_name": "Amoxicillin"}, {"generic_name": "Panadol"}]'
```

---

## 📍 Additional Documentation

* Full system documentation: `APP_DOCUMENTATION.md`
* Demo video: [https://youtu.be/BR4Ove8GRzY](https://youtu.be/BR4Ove8GRzY)
* GitHub: [https://github.com/Jules-gatete/Mission_Capstone](https://github.com/Jules-gatete/Mission_Capstone)
* API Docs: [https://mission-capstone.onrender.com/docs](https://mission-capstone.onrender.com/docs)

---

## 🛣️ Roadmap

### Short-Term

* Move models fully to S3
* Async inference + batching
* Structured logging + Sentry

### Medium-Term

* GPU/CUDA support
* Multilingual OCR support
* Blue/Green deployments

### Long-Term

* Model version management UI
* Human-in-the-loop feedback
* CHW (Community Health Worker) integration

---

### 🩺 *“Making medicine disposal safer, one classification at a time.”*
