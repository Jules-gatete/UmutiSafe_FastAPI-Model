# main.py
import os
import sys
import PIL
from PIL import Image

# Comprehensive PIL ANTIALIAS fix
try:
    # Try different possible locations for the resampling attribute
    if hasattr(Image, 'Resampling'):
        PIL.Image.ANTIALIAS = Image.Resampling.LANCZOS
    elif hasattr(Image, 'LANCZOS'):
        PIL.Image.ANTIALIAS = Image.LANCZOS
    else:
        # If neither is available, define a fallback
        PIL.Image.ANTIALIAS = 1
        print("⚠️  Using fallback for ANTIALIAS")
    
    print("✅ PIL ANTIALIAS patched successfully")
except Exception as e:
    print(f"⚠️  PIL patch warning: {e}")
    PIL.Image.ANTIALIAS = 1  # Fallback value

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
import importlib.util

# Import your existing classes
from enhanced_ocr_processor import EnhancedOCRProcessor
from medicine_disposal_predictor import MedicineDisposalPredictor

# JSON Serialization Helpers
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
        
        print("🔍 Starting system components loading...")
        
        # Test NumPy import first with detailed diagnostics
        try:
            import numpy as np
            print(f"✅ NumPy version: {np.__version__}")
            print(f"✅ NumPy path: {np.__file__}")
            
            # Test if we can access core modules
            try:
                import numpy.core as core
                print("✅ NumPy core module accessible")
            except ImportError as e:
                print(f"❌ NumPy core issue: {e}")
                
        except ImportError as e:
            print(f"❌ NumPy import failed: {e}")
            return None
        
        # Test scikit-learn import
        try:
            import sklearn
            print(f"✅ scikit-learn version: {sklearn.__version__}")
        except ImportError as e:
            print(f"❌ scikit-learn import failed: {e}")
            return None
        
        # Test joblib import
        try:
            import joblib as jb
            print(f"✅ joblib version: {jb.__version__}")
        except ImportError as e:
            print(f"❌ joblib import failed: {e}")
            return None
        
        # Check if model files exist with detailed info
        required_files = [
            'best_category_model.pkl',
            'best_risk_model.pkl', 
            'le_category.pkl',
            'le_risk.pkl',
            'tfidf_vectorizer.pkl'
        ]
        
        missing_files = []
        available_files = []
        
        for file in required_files:
            file_path = f'{base_path}{file}'
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                available_files.append(file)
                print(f"✅ Found: {file} ({file_size} bytes)")
            else:
                missing_files.append(file)
                print(f"❌ Missing: {file}")
        
        if missing_files:
            print(f"❌ Missing model files: {', '.join(missing_files)}")
            print(f"✅ Available files: {', '.join(available_files)}")
            return None
        
        print("🔄 Starting to load individual components...")
        
        # Load components one by one with detailed error handling
        components = {}
        
        # 1. Load label encoders first (usually simpler)
        try:
            print("🔄 Loading label encoders...")
            components['le_category'] = joblib.load(f'{base_path}le_category.pkl')
            print("✅ LabelEncoder for category loaded")
            
            components['le_risk'] = joblib.load(f'{base_path}le_risk.pkl')
            print("✅ LabelEncoder for risk loaded")
        except Exception as e:
            print(f"❌ Failed to load label encoders: {e}")
            import traceback
            print(f"Label encoder error details: {traceback.format_exc()}")
            return None
        
        # 2. Load TF-IDF vectorizer
        try:
            print("🔄 Loading TF-IDF vectorizer...")
            components['tfidf_vectorizer'] = joblib.load(f'{base_path}tfidf_vectorizer.pkl')
            print("✅ TF-IDF vectorizer loaded")
        except Exception as e:
            print(f"❌ Failed to load TF-IDF vectorizer: {e}")
            import traceback
            print(f"TF-IDF error details: {traceback.format_exc()}")
            return None
        
        # 3. Load ML models (most likely to have compatibility issues)
        try:
            print("🔄 Loading category model...")
            components['category_model'] = joblib.load(f'{base_path}best_category_model.pkl')
            print(f"✅ Category model loaded: {type(components['category_model'])}")
        except Exception as e:
            print(f"❌ Failed to load category model: {e}")
            import traceback
            print(f"Category model error details: {traceback.format_exc()}")
            return None
        
        try:
            print("🔄 Loading risk model...")
            components['risk_model'] = joblib.load(f'{base_path}best_risk_model.pkl')
            print(f"✅ Risk model loaded: {type(components['risk_model'])}")
        except Exception as e:
            print(f"❌ Failed to load risk model: {e}")
            import traceback
            print(f"Risk model error details: {traceback.format_exc()}")
            return None
        
        # Verify all components are loaded
        loaded_components = list(components.keys())
        print(f"🎉 All system components loaded successfully: {', '.join(loaded_components)}")
        
        return components
        
    except Exception as e:
        print(f"❌ Critical error loading components: {e}")
        import traceback
        print(f"Detailed traceback: {traceback.format_exc()}")
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
            "guidelines": "/api/guidelines",
            "model_status": "/api/models/status"
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

# Model status endpoint
@app.get("/api/models/status")
async def model_status():
    """Check the status of all ML models"""
    base_path = './models/'
    required_files = [
        'best_category_model.pkl',
        'best_risk_model.pkl', 
        'le_category.pkl',
        'le_risk.pkl',
        'tfidf_vectorizer.pkl'
    ]
    
    available_files = []
    for file in required_files:
        if os.path.exists(f'{base_path}{file}'):
            file_size = os.path.getsize(f'{base_path}{file}')
            available_files.append({"file": file, "size_bytes": file_size})
    
    missing_files = [f for f in required_files if f not in [af["file"] for af in available_files]]
    
    # Get package versions
    package_versions = {}
    try:
        import numpy as np
        package_versions['numpy'] = np.__version__
    except ImportError:
        package_versions['numpy'] = 'Not available'
    
    try:
        import sklearn
        package_versions['scikit-learn'] = sklearn.__version__
    except ImportError:
        package_versions['scikit-learn'] = 'Not available'
    
    try:
        import joblib as jb
        package_versions['joblib'] = jb.__version__
    except ImportError:
        package_versions['joblib'] = 'Not available'
    
    return {
        "models_loaded": components_loaded,
        "available_files": available_files,
        "missing_files": missing_files,
        "package_versions": package_versions,
        "components_loaded": list(predictor.components.keys()) if components_loaded and predictor else []
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

# Batch prediction endpoint
@app.post("/api/predict/batch")
async def batch_predict(medicines: list):
    """
    Batch prediction for multiple medicines
    
    Request body example:
    ```json
    [
        {
            "generic_name": "Amoxicillin 500mg",
            "brand_name": "Amoxil",
            "dosage_form": "Capsules",
            "packaging_type": "Blister"
        },
        {
            "generic_name": "Paracetamol 500mg",
            "brand_name": "Panadol",
            "dosage_form": "Tablets", 
            "packaging_type": "Bottle"
        }
    ]
    ```
    """
    if not components_loaded or predictor is None:
        raise HTTPException(
            status_code=503, 
            detail="Service unavailable - ML models not loaded. Please check server logs."
        )
    
    try:
        results = []
        for medicine in medicines:
            result = predictor.process_text_input(
                generic_name=medicine.get('generic_name', ''),
                brand_name=medicine.get('brand_name', ''),
                dosage_form=medicine.get('dosage_form', ''),
                packaging_type=medicine.get('packaging_type', 'Unknown')
            )
            # Convert numpy types for each result
            results.append(convert_numpy_types(result))
        
        return {
            "batch_id": f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "total_medicines": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")

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

# Statistics endpoint
@app.get("/api/statistics")
async def get_statistics():
    """Get API usage statistics"""
    return {
        "api_version": "1.0.0",
        "status": "operational" if components_loaded else "degraded",
        "ml_models_loaded": components_loaded,
        "predictor_ready": predictor is not None,
        "startup_error": startup_error,
        "startup_time": datetime.now().isoformat(),
        "supported_features": {
            "image_processing": components_loaded,
            "text_prediction": components_loaded,
            "batch_processing": components_loaded,
            "guidelines_lookup": True
        }
    }

# Error handlers
@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )

@app.exception_handler(503)
async def service_unavailable_handler(request, exc):
    return JSONResponse(
        status_code=503,
        content={"detail": str(exc.detail)}
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload in production
        log_level="info"
    )
