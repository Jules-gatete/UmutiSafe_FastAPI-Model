# enhanced_ocr_processor.py
import cv2
import numpy as np
import easyocr
from PIL import Image
import matplotlib.pyplot as plt
import re
import io
import base64

class EnhancedInputExtractor:
    """Enhanced input extraction with robust fallbacks and better medicine name detection"""
    
    def __init__(self):
        self.medicine_patterns = {
            'generic_names': [
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*(?:\d+\s*(?:mg|mcg|g|ml))?',
                r'([A-Z]{3,}[a-z]*\d*[a-z]*)',
                r'(\b(?:paracetamol|amoxicillin|insulin|warfarin|metformin|dapagliflozin|gliflozin)\b)'
            ],
            'dosage_forms': {
                'tablet': ['tablet', 'tab', 'tabs', 'pill', 'oral'],
                'capsule': ['capsule', 'cap', 'caps', 'gelcap'],
                'injection': ['injection', 'inj', 'vial', 'ampoule', 'syringe'],
                'liquid': ['solution', 'sol', 'liquid', 'syrup', 'suspension'],
                'cream': ['cream', 'ointment', 'gel', 'lotion', 'balm'],
                'inhaler': ['inhaler', 'aerosol', 'spray', 'puffer']
            },
            'strengths': [
                r'(\d+\s*(?:mg|mcg|g|ml|%))',
                r'(\d+\s*mg/\s*ml)',
                r'(\d+\s*IU)',
                r'(\d+\s*units?)'
            ]
        }
        
        self.medicine_database = {
            'dapagliflozin': {'type': 'antidiabetic', 'risk': 'LOW', 'forms': ['tablet']},
            'metformin': {'type': 'antidiabetic', 'risk': 'LOW', 'forms': ['tablet']},
            'insulin': {'type': 'antidiabetic', 'risk': 'HIGH', 'forms': ['injection']},
            'warfarin': {'type': 'anticoagulant', 'risk': 'HIGH', 'forms': ['tablet']},
            'paracetamol': {'type': 'analgesic', 'risk': 'LOW', 'forms': ['tablet']},
            'amoxicillin': {'type': 'antibiotic', 'risk': 'MEDIUM', 'forms': ['capsule', 'suspension']}
        }
    
    def extract_comprehensive_info(self, ocr_texts):
        """Extract comprehensive medicine information with enhanced fallbacks"""
        extracted = {
            'generic_name': 'Unknown Medicine',
            'brand_name': '',
            'dosage_form': 'Unknown Form',
            'dosage_strength': '',
            'active_ingredients': [],
            'confidence': 0.0,
            'extraction_method': 'direct',
            'manufacturer': ''
        }
        
        all_text = ' '.join([item['text'] for item in ocr_texts])
        individual_texts = [item['text'] for item in ocr_texts]
        
        # Extract Generic Name
        generic_name = self._extract_generic_name(individual_texts, all_text)
        if generic_name:
            extracted['generic_name'] = generic_name
            extracted['active_ingredients'].append(generic_name)
            extracted['extraction_method'] = 'enhanced_pattern_matching'
        
        # Extract Dosage Form
        dosage_form = self._extract_dosage_form(individual_texts, all_text)
        if dosage_form:
            extracted['dosage_form'] = dosage_form
        
        # Extract Dosage Strength
        strength = self._extract_dosage_strength(all_text)
        if strength:
            extracted['dosage_strength'] = strength
        
        # Extract Brand Name
        brand_name = self._extract_brand_name(individual_texts)
        if brand_name:
            extracted['brand_name'] = brand_name
        
        # Extract Manufacturer
        manufacturer = self._extract_manufacturer(individual_texts)
        if manufacturer:
            extracted['manufacturer'] = manufacturer
        
        # Apply fallbacks if needed
        if extracted['generic_name'] == 'Unknown Medicine':
            extracted = self._apply_intelligent_fallbacks(extracted, individual_texts, all_text)
        
        # Calculate confidence
        extracted['confidence'] = self._calculate_extraction_confidence(extracted)
        
        return extracted
    
    def _extract_generic_name(self, texts, all_text):
        """Enhanced generic name extraction"""
        all_text_lower = all_text.lower()
        
        # Look for known medicine names
        for med_name in self.medicine_database.keys():
            if med_name in all_text_lower:
                return med_name.capitalize()
        
        # Look for medicine name patterns
        for text in texts:
            clean_text = self._clean_text(text)
            if self._is_likely_medicine_name(clean_text):
                return clean_text
        
        # Look for capitalized word sequences
        medicine_like_patterns = [
            r'([A-Z][a-z]{4,}(?:[a-z]+)?)',
            r'([A-Z][a-z]{3,}\s+[A-Z][a-z]{3,})',
        ]
        
        for pattern in medicine_like_patterns:
            matches = re.findall(pattern, all_text)
            if matches:
                for match in matches:
                    if self._is_likely_medicine_name(match):
                        return match
        
        return None
    
    def _extract_dosage_form(self, texts, all_text):
        """Extract dosage form"""
        all_text_lower = all_text.lower()
        
        for form, keywords in self.medicine_patterns['dosage_forms'].items():
            for keyword in keywords:
                if keyword in all_text_lower:
                    return form.capitalize()
        return None
    
    def _extract_dosage_strength(self, all_text):
        """Extract dosage strength"""
        for pattern in self.medicine_patterns['strengths']:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            if matches:
                return matches[0]
        return None
    
    def _extract_brand_name(self, texts):
        """Extract brand name"""
        for text in texts:
            clean_text = text.strip()
            if (2 < len(clean_text) < 25 and 
                clean_text.isupper() and 
                not any(char.isdigit() for char in clean_text) and
                not self._is_common_acronym(clean_text) and
                len(clean_text) > 3):
                return clean_text
        return None
    
    def _extract_manufacturer(self, texts):
        """Extract manufacturer information"""
        manufacturer_keywords = ['ltd', 'inc', 'corporation', 'pharma', 'laboratories', 'company']
        
        for text in texts:
            text_lower = text.lower()
            if any(keyword in text_lower for keyword in manufacturer_keywords):
                return text
        return None
    
    def _is_likely_medicine_name(self, text):
        """Check if text is likely a medicine name"""
        if len(text) < 4:
            return False
        
        exclude_patterns = [
            'IIH)', 'LTD', 'INC', 'CORP', 'PHARMA', 'LABS', 'COMPANY',
            'MG', 'MCG', 'ML', 'IU', 'BP', 'USP', 'FDA', 'FOR', 'AND', 'THE'
        ]
        
        if any(pattern in text.upper() for pattern in exclude_patterns):
            return False
        
        medicine_indicators = [
            len(text) > 4,
            text[0].isupper(),
            any(c.islower() for c in text),
            not text.isupper(),
            not any(char.isdigit() for char in text),
        ]
        
        return all(medicine_indicators)
    
    def _clean_text(self, text):
        """Clean text for better extraction"""
        cleaned = re.sub(r'[^\w\s]', '', text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    
    def _is_common_acronym(self, text):
        """Check if text is a common acronym"""
        common_acronyms = {'MG', 'MCG', 'ML', 'IU', 'BP', 'USP', 'FDA', 'LTD', 'INC', 'CORP'}
        return text in common_acronyms
    
    def _calculate_extraction_confidence(self, extracted):
        """Calculate confidence score for extraction"""
        score = 0.0
        max_score = 5.0
        
        if extracted['generic_name'] != 'Unknown Medicine':
            score += 2.0
        if extracted['dosage_form'] != 'Unknown Form':
            score += 1.0
        if extracted['dosage_strength']:
            score += 1.0
        if extracted['brand_name']:
            score += 0.5
        if extracted['manufacturer']:
            score += 0.5
        
        return min(1.0, score / max_score)
    
    def _apply_intelligent_fallbacks(self, extracted, texts, all_text):
        """Apply intelligent fallback strategies"""
        # Fallback 1: Look for any word that might be a medicine name
        for text in texts:
            words = text.split()
            for word in words:
                clean_word = self._clean_text(word)
                if (len(clean_word) > 5 and 
                    clean_word[0].isupper() and 
                    any(c.islower() for c in clean_word) and
                    not self._is_common_acronym(clean_word)):
                    extracted['generic_name'] = clean_word
                    extracted['extraction_method'] = 'fallback_capitalized_word'
                    return extracted
        
        # Fallback 2: Use context from dosage form to infer
        if extracted['dosage_form'] != 'Unknown Form':
            extracted['generic_name'] = f"Unknown {extracted['dosage_form']}"
            extracted['extraction_method'] = 'fallback_dosage_context'
        
        return extracted

class EnhancedOCRProcessor:
    """Enhanced OCR processor with improved input extraction"""
    
    def __init__(self):
        self.reader = easyocr.Reader(['en'])
        self.extractor = EnhancedInputExtractor()
    
    def extract_text_with_visualization(self, image_data):
        """Extract text with comprehensive visualization and enhanced extraction"""
        try:
            # Convert image
            if isinstance(image_data, bytes):
                image_array = np.frombuffer(image_data, np.uint8)
                original_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            else:
                original_image = image_data

            # Enhanced preprocessing
            gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            denoised = cv2.medianBlur(gray, 5)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            _, processed_image = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Perform OCR
            results = self.reader.readtext(processed_image, detail=1, paragraph=False)

            # Create visualization
            annotated_image = original_image.copy()
            medicine_texts = []
            all_texts = []

            for (bbox, text, confidence) in results:
                if confidence > 0.1:
                    text_clean = text.strip()
                    if len(text_clean) > 1:
                        all_texts.append({'text': text_clean, 'confidence': confidence, 'bbox': bbox})

                        # Draw bounding box
                        top_left = tuple(map(int, bbox[0]))
                        bottom_right = tuple(map(int, bbox[2]))

                        is_medicine_related = self._is_medicine_related(text_clean)

                        if is_medicine_related:
                            medicine_texts.append({'text': text_clean, 'confidence': confidence})
                            color = (255, 0, 0)
                            thickness = 3
                        else:
                            color = (0, 255, 0)
                            thickness = 2

                        cv2.rectangle(annotated_image, top_left, bottom_right, color, thickness)

            # Enhanced structured information extraction
            extracted_info = self.extractor.extract_comprehensive_info(all_texts)

            return {
                'success': True,
                'extracted_info': extracted_info,
                'medicine_texts': medicine_texts,
                'all_texts': all_texts,
                'confidence_avg': np.mean([t['confidence'] for t in all_texts]) if all_texts else 0,
                'extraction_confidence': extracted_info['confidence']
            }

        except Exception as e:
            print(f"❌ OCR processing error: {e}")
            return {
                'success': False,
                'error': str(e),
                'extracted_info': self._create_fallback_info(),
                'medicine_texts': [],
                'all_texts': []
            }

    def _is_medicine_related(self, text):
        """Enhanced medicine text detection"""
        text_lower = text.lower()

        medicine_keywords = [
            'tablet', 'capsule', 'injection', 'solution', 'suspension', 'cream', 'ointment',
            'syrup', 'drops', 'spray', 'inhaler', 'patch', 'suppository', 'powder', 'liquid',
            'pill', 'caps', 'cap', 'tab', 'inj', 'sol', 'susp', 'ampoule', 'vial', 'ampule',
            'mg', 'ml', 'mcg', 'iu', 'g', 'gram', 'milligram', 'microgram', 'unit', '%',
            'strength', 'dose', 'dosage', 'contains', 'ingredients', 'active', 'composition',
            'formula', 'prescription', 'medicine', 'drug', 'pharmaceutical'
        ]
        
        if any(keyword in text_lower for keyword in medicine_keywords):
            return True

        if re.search(r'\d+\s*(?:mg|mcg|ml|g|%|iu|units?)', text_lower):
            return True

        return False

    def _create_fallback_info(self):
        """Create fallback information when extraction fails"""
        return {
            'generic_name': 'Unknown Medicine',
            'brand_name': '',
            'dosage_form': 'Unknown Form', 
            'dosage_strength': '',
            'active_ingredients': [],
            'confidence': 0.0,
            'extraction_method': 'error_fallback',
            'manufacturer': ''
        }