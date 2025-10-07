# UmutiSafe -  Medicine Disposal Classification/Guidance System

## Description
UmutiSafe is an intelligent system that classifies medicine disposal methods using AI and computer vision. It analyzes medicine labels through OCR and machine learning to provide safe disposal recommendations, helping prevent environmental contamination and public health risks.

**GitHub Repository:** [https://github.com/Jules-gatete/Mission_Capstone.git](https://github.com/Jules-gatete/Mission_Capstone.git)

##  Quick Start

### Prerequisites
- Python version = python-3.11.9
- pip package manager

### Installation
```bash
# Clone the repository
git clone [https://github.com/Jules-gatete/Mission_Capstone.git](https://github.com/Jules-gatete/Mission_Capstone.git)


# Install dependencies
pip install -r requirements.txt

# Run the API
python run.py or uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Access the API
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

##  Features

###  Smart OCR Processing
- Advanced text extraction from medicine labels
- Medicine-specific keyword detection
- Visual annotation of detected text
- Support for various image formats

###  AI Classification
- **Dual-Model Prediction**:
  - Disposal Category (5 classes)
  - Risk Level (3 levels)[Low,Medium and HIGH]
- Multiple input formats (image & text)
- Batch processing capabilities
- Confidence scoring

###  Safety Guidance
- Category-specific disposal instructions
- Risk-based safety protocols
- Clear prohibitions and warnings

##  API Usage

### Image-Based Prediction
```bash
curl -X POST "http://localhost:8000/api/predict/image" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@medicine_label.jpg"
```

### Text-Based Prediction
```bash
curl -X POST "http://localhost:8000/api/predict/text" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "generic_name=Amoxicillin 500mg&brand_name=Amoxil&dosage_form=Capsules"
```

### Batch Prediction
```bash
curl -X POST "http://localhost:8000/api/predict/batch" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "generic_name": "Amoxicillin 500mg",
      "brand_name": "Amoxil",
      "dosage_form": "Capsules"
    },
    {
      "generic_name": "Paracetamol 500mg", 
      "brand_name": "Panadol",
      "dosage_form": "Tablets"
    }
  ]'
```

## API Structure

```
umutisafe/
├── main.py                 # FastAPI application
├── enhanced_ocr_processor.py    # OCR processing class
├── medicine_disposal_predictor.py # Prediction engine
├── requirements.txt        # Python dependencies
├── run.py                 # Application runner
├── models/                # Trained ML models
│   ├── best_category_model.pkl
│   ├── best_risk_model.pkl
│   ├── le_category.pkl
│   ├── le_risk.pkl
│   └── tfidf_vectorizer.pkl
└── data/
    └── processed/
        └── disposal_guidelines_db.py
```

##  Demo Video

[Link demo video](https://youtu.be/BR4Ove8GRzY)



##  Deployment Plan

### Phase 1: API Deployment 
- FastAPI backend with Swagger UI
- Containerized deployment
- Cloud-ready architecture

### Phase 2: Web Interface 
-  frontend application
- Image upload interface
- Real-time results display
- Mobile-responsive design
- take back guidance with CHW will be added


##  Technical Details

### Machine Learning Models
- **Text Feature Extraction**: TF-IDF Vectorizer
- **Category Prediction**: Ensemble Classifier
- **Risk Assessment**: Multi-class Classifier
- **OCR Engine**: EasyOCR with custom preprocessing

### Supported Medicine Types
- Tablets, Capsules, Injections
- Solutions, Suspensions, Creams
- Syrups, Drops, Sprays
- Ampoules, Vials, Patches

##  Performance Metrics

- OCR Accuracy: >85% on medicine labels
- Classification Accuracy: >90% on trained categories
- Risk Assessment: >85% match with regulatory guidelines
- Processing Time: <5 seconds per image

- Rwanda FDA for medicine dataset
- Open-source OCR and ML libraries
- Healthcare professionals for validation

---

**💊 Making medicine disposal safer, one classification at a time!**




