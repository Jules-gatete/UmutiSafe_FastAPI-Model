# main.py
import PIL.Image
import PIL

# Fix for PIL ANTIALIAS issue - MUST BE AT THE TOP
if not hasattr(PIL.Image, 'ANTIALIAS'):
    if hasattr(PIL.Image, 'Resampling'):
        PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS
        print("✅ Patched ANTIALIAS to Resampling.LANCZOS")
    elif hasattr(PIL.Image, 'LANCZOS'):
        PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
        print("✅ Patched ANTIALIAS to LANCZOS")
    else:
        PIL.Image.ANTIALIAS = None
        print("⚠️  Could not patch ANTIALIAS")

print(f"📦 PIL version: {PIL.__version__}")

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import pandas as pd
import numpy as np
import re
import json
from datetime import datetime
import joblib
import cv2
import easyocr
import os
import importlib.util

# Import your existing classes
from enhanced_ocr_processor import EnhancedOCRProcessor

# JSON Serialization Helpers
class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif hasattr(obj, 'item'):
            return obj.item()
        return super(NumpyEncoder, self).default(obj)

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types, including dictionary keys"""
    if isinstance(obj, dict):
        converted_dict = {}
        for key, value in obj.items():
            # Convert key to string if it's a numpy type
            if isinstance(key, (np.integer, np.int32, np.int64)):
                converted_key = int(key)
            elif isinstance(key, (np.floating, np.float32, np.float64)):
                converted_key = float(key)
            elif hasattr(key, 'item'):
                converted_key = key.item()
            else:
                converted_key = key
            
            # Ensure key is string for JSON serialization
            if not isinstance(converted_key, (str, int, float, bool)) and converted_key is not None:
                converted_key = str(converted_key)
            
            converted_dict[converted_key] = convert_numpy_types(value)
        return converted_dict
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    elif hasattr(obj, 'item'):
        return obj.item()
    else:
        return obj

# Load the disposal guidelines from your Python file
disposal_guidelines = {}
try:
    disposal_guidelines_db_path = './data/processed/disposal_guidelines_db.py'
    if os.path.exists(disposal_guidelines_db_path):
        # Import the disposal_guidelines dictionary properly
        spec = importlib.util.spec_from_file_location("disposal_guidelines_db", disposal_guidelines_db_path)
        disposal_guidelines_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(disposal_guidelines_module)
        disposal_guidelines = disposal_guidelines_module.disposal_guidelines
        print("✅ Disposal guidelines database loaded successfully")
    else:
        print("⚠️  Disposal guidelines file not found")
        disposal_guidelines = {}
except Exception as e:
    print(f"⚠️  Error loading disposal guidelines: {e}")
    disposal_guidelines = {}

# Initialize FastAPI app
app = FastAPI(
    title="UmutiSafe Medicine Classification/Disposal API",
    description="Medicine disposal classification and safety guidance system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MedicineDisposalPredictor:
    """Predictor that requires ML models to work"""
    
    def __init__(self, components):
        if not components:
            raise ValueError("ML components are required")
        
        self.components = components
        self.ocr_processor = EnhancedOCRProcessor()
        
        # Validate that all required components are present
        required_components = ['category_model', 'risk_model', 'le_category', 'le_risk', 'tfidf_vectorizer']
        for comp in required_components:
            if comp not in components or components[comp] is None:
                raise ValueError(f"Required component '{comp}' is missing or None")
        
        print("✅ Predictor initialized with ML models")
    
    def predict_from_image(self, uploaded_image):
        """Predict disposal from uploaded image"""
        
        # Extract text with visualization
        ocr_result = self.ocr_processor.extract_text_with_visualization(uploaded_image)
        
        if not ocr_result['success']:
            return self._create_error_response("OCR processing failed")
        
        # Use extracted info for prediction
        extracted_info = ocr_result['extracted_info']
        
        # Create prediction data
        generic_name = extracted_info.get('generic_name', 'Unknown Medicine')
        if extracted_info.get('dosage_strength'):
            generic_name += f" {extracted_info['dosage_strength']}"
        dosage_form = extracted_info.get('dosage_form', '')
        brand_name = extracted_info.get('brand_name', '')
        
        medicine_data = pd.DataFrame([{
            'Generic Name': generic_name,
            'Product Brand Name': brand_name,
            'Dosage Form': dosage_form,
            'Packaging Type': 'Unknown'
        }])
        
        # Get predictions using ML models
        category_pred, risk_pred, probabilities = self._get_predictions(medicine_data)
        
        # Get guidelines
        guidelines = self._get_disposal_guidelines(category_pred, risk_pred)
        
        # Create comprehensive response
        response = {
            'success': True,
            'ocr_info': {
                'confidence': float(ocr_result['confidence_avg']),  # Convert to float
                'extracted_info': extracted_info,
                'medicine_texts_found': int(len(ocr_result['medicine_texts'])),  # Convert to int
            },
            'medicine_info': {
                'generic_name': generic_name,
                'brand_name': brand_name,
                'dosage_form': dosage_form,
            },
            'predictions': {
                'disposal_category': category_pred,
                'risk_level': risk_pred,
                'confidence': float(max(probabilities.values()) if probabilities else 0),
                'all_probabilities': probabilities
            },
            'safety_guidance': guidelines,
            'timestamp': datetime.now().isoformat()
        }
        
        return response
    
    def process_text_input(self, generic_name, brand_name="", dosage_form="", packaging_type=""):
        """Process text input"""
        
        medicine_data = pd.DataFrame([{
            'Generic Name': generic_name,
            'Product Brand Name': brand_name,
            'Dosage Form': dosage_form,
            'Packaging Type': packaging_type
        }])
        
        # Get predictions using ML models
        category_pred, risk_pred, probabilities = self._get_predictions(medicine_data)
        
        # Get guidelines
        guidelines = self._get_disposal_guidelines(category_pred, risk_pred)
        
        response = {
            'success': True,
            'medicine_info': {
                'generic_name': generic_name,
                'brand_name': brand_name,
                'dosage_form': dosage_form
            },
            'predictions': {
                'disposal_category': category_pred,
                'risk_level': risk_pred,
                'confidence': float(max(probabilities.values()) if probabilities else 0),
                'all_probabilities': probabilities
            },
            'safety_guidance': guidelines,
            'timestamp': datetime.now().isoformat()
        }
        
        return response
    
    def _get_predictions(self, medicine_data):
        """Get predictions using saved ML models"""
        try:
            # Combine text features as per training
            combined_text = (medicine_data['Product Brand Name'].astype(str) + ' ' +
                            medicine_data['Generic Name'].astype(str) + ' ' +
                            medicine_data['Dosage Form'].astype(str) + ' ' +
                            medicine_data['Packaging Type'].astype(str))
            
            # TF-IDF features using saved vectorizer
            tfidf = self.components['tfidf_vectorizer']
            X_text_tfidf = tfidf.transform(combined_text)
            
            # Text statistics
            word_count = combined_text.apply(lambda x: len(str(x).split()))
            char_count = combined_text.apply(lambda x: len(str(x)))
            special_chars = combined_text.apply(lambda x: len(re.findall(r'[^a-zA-Z0-9\s]', str(x))))
            
            text_features_df = pd.DataFrame({
                'word_count': word_count,
                'char_count': char_count,
                'special_chars': special_chars
            })
            
            # Combine features
            X = np.hstack((X_text_tfidf.toarray(), text_features_df.values))
            
            # Predict category using saved model
            category_model = self.components['category_model']
            le_category = self.components['le_category']
            
            category_pred_encoded = category_model.predict(X)[0]
            category_proba = category_model.predict_proba(X)[0]
            category_pred = le_category.inverse_transform([category_pred_encoded])[0]
            
            # Predict risk level using the risk model
            risk_model = self.components['risk_model']
            le_risk = self.components['le_risk']
            
            risk_pred_encoded = risk_model.predict(X)[0]
            risk_pred = le_risk.inverse_transform([risk_pred_encoded])[0]
            
            # Create probabilities dictionary for categories - ensure native Python types
            probabilities = {}
            for i, prob in enumerate(category_proba):
                category_name = str(le_category.classes_[i])  # Ensure string key
                # Convert numpy float to Python float
                probabilities[category_name] = float(round(prob, 3))
            
            # Convert predictions to native Python types
            category_pred = str(category_pred)
            risk_pred = str(risk_pred)
            
            return category_pred, risk_pred, probabilities
            
        except Exception as e:
            print(f"❌ ML prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"ML prediction failed: {str(e)}")
    
    def _get_disposal_guidelines(self, category, risk_level):
        """Get disposal guidelines"""
        try:
            if category in disposal_guidelines:
                guideline_data = disposal_guidelines[category].copy()
                
                # Handle different data types for prohibitions and risks
                if isinstance(guideline_data.get('prohibitions'), list):
                    guideline_data['prohibitions'] = '. '.join(guideline_data['prohibitions'])
                
                if isinstance(guideline_data.get('risks'), list):
                    guideline_data['risks'] = '. '.join(guideline_data['risks'])
                
                # Add risk-specific information
                guideline_data['risk_level'] = risk_level
                return guideline_data
            else:
                return self._get_fallback_guidelines(category, risk_level)
                
        except Exception as e:
            print(f"⚠️  Error accessing guidelines: {e}")
            return self._get_fallback_guidelines(category, risk_level)
    
    def _get_fallback_guidelines(self, category, risk_level):
        """Fallback guidelines"""
        return {
            'category_name': category,
            'risk_level': risk_level,
            'prohibitions': 'Follow general medicine disposal guidelines. Consult healthcare provider for specific instructions.',
            'risks': 'Unknown risks. Professional guidance recommended for safe disposal.',
            'procedure': 'Contact local pharmacy or healthcare facility for disposal instructions.',
            'special_instructions': 'Always prioritize safety and environmental protection.'
        }
    
    def _create_error_response(self, message):
        return {
            'success': False,
            'error': message,
            'timestamp': datetime.now().isoformat()
        }

# Global variables
predictor = None
components_loaded = False
startup_error = None

# Load system components at startup
@app.on_event("startup")
async def startup_event():
    global predictor, components_loaded, startup_error
    try:
        components = load_system_components()
        if components:
            predictor = MedicineDisposalPredictor(components)
            components_loaded = True
            startup_error = None
            print("✅ UmutiSafe API started successfully with ML models!")
        else:
            components_loaded = False
            startup_error = "ML models failed to load"
            print("❌ UmutiSafe API started without ML models")
    except Exception as e:
        components_loaded = False
        startup_error = str(e)
        print(f"❌ Startup error: {e}")

def load_system_components():
    """Load all system components - returns None if fails"""
    try:
        base_path = './models/'
        
        # Check if model files exist
        required_files = [
            'best_category_model.pkl',
            'best_risk_model.pkl', 
            'le_category.pkl',
            'le_risk.pkl',
            'tfidf_vectorizer.pkl'
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(f'{base_path}{file}'):
                missing_files.append(file)
        
        if missing_files:
            print(f"⚠️  Missing model files: {', '.join(missing_files)}")
            return None
        
        components = {
            'category_model': joblib.load(f'{base_path}best_category_model.pkl'),
            'risk_model': joblib.load(f'{base_path}best_risk_model.pkl'),
            'le_category': joblib.load(f'{base_path}le_category.pkl'),
            'le_risk': joblib.load(f'{base_path}le_risk.pkl'),
            'tfidf_vectorizer': joblib.load(f'{base_path}tfidf_vectorizer.pkl')
        }
        
        print("✅ All system components loaded successfully")
        return components
        
    except Exception as e:
        print(f"❌ Error loading components: {e}")
        return None

# Health check endpoint
@app.get("/")
async def root():
    status_info = {
        "message": "Welcome to UmutiSafe Medicine Classification/Disposal API",
        "status": "operational" if components_loaded else "degraded",
        "version": "1.0.0",
        "ml_models_loaded": components_loaded,
        "startup_error": startup_error
    }
    
    if components_loaded:
        status_info["endpoints"] = {
            "health": "/health",
            "docs": "/docs",
            "predict_image": "/api/predict/image",
            "predict_text": "/api/predict/text",
            "batch_predict": "/api/predict/batch"
        }
    else:
        status_info["endpoints"] = {
            "health": "/health",
            "docs": "/docs",
            "guidelines": "/api/guidelines"
        }
        status_info["note"] = "Prediction endpoints unavailable - ML models not loaded"
    
    return status_info

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if components_loaded else "degraded",
        "timestamp": datetime.now().isoformat(),
        "ml_models_loaded": components_loaded,
        "predictor_ready": predictor is not None,
        "startup_error": startup_error
    }

# Image prediction endpoint
@app.post("/api/predict/image")
async def predict_from_image(file: UploadFile = File(...)):
    """
    Predict disposal category and risk level from medicine label image
    
    - **file**: Upload a clear image of medicine label (JPEG, PNG, JPG)
    """
    if not components_loaded or predictor is None:
        raise HTTPException(
            status_code=503, 
            detail="Service unavailable - ML models not loaded. Please check server logs."
        )
    
    # Validate file type
    allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
        )
    
    try:
        # Read image data
        image_data = await file.read()
        
        # Process with predictor
        result = predictor.predict_from_image(image_data)
        
        # Convert numpy types to native Python types
        result = convert_numpy_types(result)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# Text prediction endpoint
@app.post("/api/predict/text")
async def predict_from_text(
    generic_name: str = Form(..., description="Medicine generic name (e.g., Amoxicillin 500mg)"),
    brand_name: str = Form("", description="Brand name (optional)"),
    dosage_form: str = Form("", description="Dosage form (e.g., Tablets, Capsules, Injection)"),
    packaging_type: str = Form("Unknown", description="Packaging type (optional)")
):
    """
    Predict disposal category and risk level from text input
    
    - **generic_name**: Required - Medicine generic name with strength
    - **brand_name**: Optional - Brand name
    - **dosage_form**: Optional - Dosage form
    - **packaging_type**: Optional - Packaging type
    """
    if not components_loaded or predictor is None:
        raise HTTPException(
            status_code=503, 
            detail="Service unavailable - ML models not loaded. Please check server logs."
        )
    
    try:
        result = predictor.process_text_input(
            generic_name=generic_name,
            brand_name=brand_name,
            dosage_form=dosage_form,
            packaging_type=packaging_type
        )
        
        # Convert numpy types to native Python types
        result = convert_numpy_types(result)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Get disposal guidelines
@app.get("/api/guidelines")
async def get_guidelines():
    """Get all available disposal guidelines"""
    try:
        return {
            "total_categories": len(disposal_guidelines),
            "categories": list(disposal_guidelines.keys()),
            "guidelines": disposal_guidelines
        }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading guidelines: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )