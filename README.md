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
