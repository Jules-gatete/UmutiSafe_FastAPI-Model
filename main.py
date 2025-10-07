# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
import uvicorn
import io
import pandas as pd
import numpy as np
import re
import json
from datetime import datetime
import joblib
import cv2
from PIL import Image
import easyocr
import os
import sys
from typing import Optional, Dict, Any
import importlib.util

# Import your existing classes
from enhanced_ocr_processor import EnhancedOCRProcessor
from medicine_disposal_predictor import MedicineDisposalPredictor

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
        
        # Debug: Print the structure of the first guideline to understand the format
        if disposal_guidelines and len(disposal_guidelines) > 0:
            first_key = list(disposal_guidelines.keys())[0]
            print(f"📋 Sample guideline structure for '{first_key}':")
            print(f"   Type: {type(disposal_guidelines[first_key])}")
            if isinstance(disposal_guidelines[first_key], dict):
                for k, v in disposal_guidelines[first_key].items():
                    print(f"   {k}: {type(v)} - {v}")
    else:
        print("⚠️  Disposal guidelines file not found")
        disposal_guidelines = {}
except Exception as e:
    print(f"⚠️  Error loading disposal guidelines: {e}")
    disposal_guidelines = {}

# Initialize FastAPI app
app = FastAPI(
    title="UmutiSafe Medicine  Classification/Disposal API",
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

# Load system components at startup
@app.on_event("startup")
async def startup_event():
    global predictor, components_loaded
    try:
        components = load_system_components()
        if components:
            predictor = MedicineDisposalPredictor(components)
            components_loaded = True
            print("✅ UmutiSafe API started successfully!")
        else:
            print("❌ Failed to load system components")
    except Exception as e:
        print(f"❌ Startup error: {e}")

def load_system_components():
    """Load all system components"""
    try:
        # Update paths for deployment
        base_path = './models/'
        
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
    return {
        "message": "Welcome to UmutiSafe Medicine Classification/Disposal API",
        "status": "operational" if components_loaded else "initializing",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "predict_image": "/api/predict/image",
            "predict_text": "/api/predict/text",
            "batch_predict": "/api/predict/batch"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if components_loaded else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "components_loaded": components_loaded
    }

# Image prediction endpoint
@app.post("/api/predict/image")
async def predict_from_image(file: UploadFile = File(...)):
    """
    Predict disposal category and risk level from medicine label image
    
    - **file**: Upload a clear image of medicine label (JPEG, PNG, JPG)
    """
    if not components_loaded:
        raise HTTPException(status_code=503, detail="Service not ready")
    
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
    if not components_loaded:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        result = predictor.process_text_input(
            generic_name=generic_name,
            brand_name=brand_name,
            dosage_form=dosage_form,
            packaging_type=packaging_type
        )
        
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
    if not components_loaded:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        results = []
        for medicine in medicines:
            result = predictor.process_text_input(
                generic_name=medicine.get('generic_name', ''),
                brand_name=medicine.get('brand_name', ''),
                dosage_form=medicine.get('dosage_form', ''),
                packaging_type=medicine.get('packaging_type', 'Unknown')
            )
            results.append(result)
        
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
        "status": "operational",
        "components_loaded": components_loaded,
        "startup_time": datetime.now().isoformat(),
        "supported_features": {
            "image_processing": True,
            "text_prediction": True,
            "batch_processing": True,
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

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False  # Disable reload in production,
        log_level="info"
    )
