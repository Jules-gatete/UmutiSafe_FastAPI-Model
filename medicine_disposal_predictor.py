# medicine_disposal_predictor.py
import pandas as pd
import numpy as np
import re
import json
from datetime import datetime
import os
import importlib.util

class MedicineDisposalPredictor:
    """Complete predictor using all saved components"""
    
    def __init__(self, components):
        self.components = components
        from enhanced_ocr_processor import EnhancedOCRProcessor
        self.ocr_processor = EnhancedOCRProcessor()
        
        # Load disposal guidelines
        self.disposal_guidelines = self._load_disposal_guidelines()
    
    def _load_disposal_guidelines(self):
        """Load disposal guidelines"""
        try:
            disposal_guidelines_db_path = './data/processed/disposal_guidelines_db.py'
            if os.path.exists(disposal_guidelines_db_path):
                # Import the disposal_guidelines dictionary properly
                spec = importlib.util.spec_from_file_location("disposal_guidelines_db", disposal_guidelines_db_path)
                disposal_guidelines_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(disposal_guidelines_module)
                disposal_guidelines = disposal_guidelines_module.disposal_guidelines
                print("✅ Disposal guidelines database loaded successfully")
                return disposal_guidelines
            else:
                print("⚠️  Disposal guidelines file not found")
                return self._create_fallback_guidelines()
        except Exception as e:
            print(f"⚠️  Error loading disposal guidelines: {e}")
            return self._create_fallback_guidelines()
        
    def predict_from_image(self, uploaded_image):
        """Predict disposal from uploaded image"""
        
        # Extract text with visualization
        ocr_result = self.ocr_processor.extract_text_with_visualization(uploaded_image)
        
        if not ocr_result['success']:
            return self._create_error_response("OCR processing failed")
        
        # Use extracted info for prediction
        extracted_info = ocr_result['extracted_info']
        
        # Create prediction data
        generic_name = extracted_info['generic_name'] or "Unknown Medicine"
        if extracted_info['dosage_strength']:
            generic_name += f" {extracted_info['dosage_strength']}"
        dosage_form = extracted_info.get('dosage_form', '')
        brand_name = extracted_info.get('brand_name', '')
        
        medicine_data = pd.DataFrame([{
            'Generic Name': generic_name,
            'Product Brand Name': brand_name,
            'Dosage Form': dosage_form,
            'Packaging Type': 'Unknown'
        }])
        
        # Get predictions using saved models
        category_pred, risk_pred, probabilities = self._get_predictions(medicine_data)
        
        # Get guidelines
        guidelines = self._get_disposal_guidelines(category_pred, risk_pred)
        
        # Create comprehensive response
        response = {
            'success': True,
            'ocr_info': {
                'confidence': ocr_result['confidence_avg'],
                'extracted_info': extracted_info,
                'medicine_texts_found': len(ocr_result['medicine_texts']),
            },
            'medicine_info': {
                'generic_name': generic_name,
                'brand_name': brand_name,
                'dosage_form': dosage_form,
            },
            'predictions': {
                'disposal_category': category_pred,
                'risk_level': risk_pred,
                'confidence': max(probabilities.values()) if probabilities else 0,
                'all_probabilities': probabilities
            },
            'safety_guidance': guidelines,
            'timestamp': datetime.now().isoformat()
        }
        
        return response
    
    def _get_predictions(self, medicine_data):
        """Get predictions using saved models - FIXED for risk prediction"""
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
            
            # Create probabilities dictionary for categories
            probabilities = {
                le_category.classes_[i]: round(prob, 3)
                for i, prob in enumerate(category_proba)
            }
            
            return category_pred, risk_pred, probabilities
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return "Unknown", "Unknown", {}
    
    def _get_disposal_guidelines(self, category, risk_level):
        """Get disposal guidelines using saved database"""
        try:
            if category in self.disposal_guidelines:
                guideline_data = self.disposal_guidelines[category].copy()
                
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
    
    def process_text_input(self, generic_name, brand_name="", dosage_form="", packaging_type=""):
        """Process text input"""
        
        medicine_data = pd.DataFrame([{
            'Generic Name': generic_name,
            'Product Brand Name': brand_name,
            'Dosage Form': dosage_form,
            'Packaging Type': packaging_type
        }])
        
        # Get predictions
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
                'confidence': max(probabilities.values()) if probabilities else 0,
                'all_probabilities': probabilities
            },
            'safety_guidance': guidelines,
            'timestamp': datetime.now().isoformat()
        }
        
        return response
    
    def _create_error_response(self, message):
        return {
            'success': False,
            'error': message,
            'timestamp': datetime.now().isoformat()
        }