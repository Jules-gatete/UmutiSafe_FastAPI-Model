# enhanced_ocr_processor.py
import cv2
import numpy as np
import re
import matplotlib.pyplot as plt
import easyocr
from PIL import Image

class EnhancedOCRProcessor:
    """OCR processor optimized for medicine label extraction"""
    
    def __init__(self):
        self.reader = easyocr.Reader(['en'])
        # Comprehensive medicine-related keywords - EXPANDED
        self.medicine_keywords = [
            # Dosage forms - EXPANDED
            'tablet', 'capsule', 'injection', 'solution', 'suspension', 'cream', 'ointment',
            'syrup', 'drops', 'spray', 'inhaler', 'patch', 'suppository', 'powder', 'liquid',
            'pill', 'caps', 'cap', 'tab', 'inj', 'sol', 'susp', 'ampoule', 'vial', 'ampule',
            'injectable', 'infusion', 'emulsion', 'gel', 'lotion', 'liniment', 'paste',
            'aerosol', 'nebulizer', 'implant', 'pellet', 'granules', 'sachet', 'cachet',
            
            # Units and measurements
            'mg', 'ml', 'mcg', 'iu', 'g', 'gram', 'milligram', 'microgram', 'unit', '%', 'ppm',
            'milliliter', 'liter', 'international', 'units', 'cc', 'cm³', 'kilogram', 'kg',
            
            # Medicine-related terms
            'strength', 'dose', 'dosage', 'contains', 'ingredients', 'active', 'composition',
            'formula', 'prescription', 'medicine', 'drug', 'pharmaceutical', 'therapy',
            'treatment', 'medication', 'pill', 'drops', 'bottle', 'vial', 'ampoule',
            'blister', 'pack', 'strip', 'bottle', 'container', 'pharmacy', 'dispense',
            'administer', 'apply', 'inject', 'oral', 'topical', 'parenteral', 'intravenous',
            'intramuscular', 'subcutaneous', 'ophthalmic', 'otic', 'nasal', 'rectal', 'vaginal',
            
            # Common medicine names from Rwanda FDA dataset - EXPANDED
            'paracetamol', 'amoxicillin', 'ibuprofen', 'aspirin', 'insulin', 'morphine', 'warfarin',
            'metformin', 'diazepam', 'codeine', 'ciprofloxacin', 'levothyroxine', 'atorvastatin',
            'simvastatin', 'omeprazole', 'losartan', 'metoprolol', 'amlodipine', 'prednisone',
            'hydrochlorothiazide', 'lisinopril', 'albuterol', 'fluoxetine', 'sertraline',
            'glimepiride', 'gliclazide', 'aflibercept', 'moxifloxacin', 'perindopril',
            'indapamide', 'doxorubicin', 'cisplatin', 'fluorouracil', 'cyclophosphamide',
            'vincristine', 'buprenorphine', 'oxycodone', 'methylphenidate', 'heparin',
            'rivaroxaban', 'ceftriaxone', 'azithromycin', 'clindamycin', 'doxycycline',
            'erythromycin', 'gentamicin', 'vancomycin', 'phenytoin', 'carbamazepine',
            'valproate', 'lamotrigine', 'furosemide', 'spironolactone', 'digoxin',
            'nitroglycerin', 'salbutamol', 'beclomethasone', 'fluticasone', 'montelukast',
            'loratadine', 'cetirizine', 'ranitidine', 'cimetidine', 'pantoprazole',
            'esomeprazole', 'bisacodyl', 'loperamide', 'senna', 'lactulose'
        ]
    
    def extract_text_with_visualization(self, image_data):
        """Extract text with comprehensive visualization"""
        
        # Convert and display original image
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
        
        # Perform OCR with lower confidence threshold
        results = self.reader.readtext(processed_image, detail=1, paragraph=False)
        
        # Create visualization
        annotated_image = original_image.copy()
        medicine_texts = []
        all_texts = []
        
        for (bbox, text, confidence) in results:
            if confidence > 0.1:  # Lower threshold for more text capture
                text_clean = text.strip()
                if len(text_clean) > 1:
                    all_texts.append({'text': text_clean, 'confidence': confidence, 'bbox': bbox})
                    
                    # Draw bounding box
                    top_left = tuple(map(int, bbox[0]))
                    bottom_right = tuple(map(int, bbox[2]))
                    
                    # Enhanced medicine detection
                    is_medicine_related = self._is_medicine_related(text_clean)
                    
                    if is_medicine_related:
                        medicine_texts.append({'text': text_clean, 'confidence': confidence})
                        color = (255, 0, 0)  # Blue for medicine-related
                        thickness = 3
                    else:
                        color = (0, 255, 0)  # Green for other text
                        thickness = 2
                    
                    cv2.rectangle(annotated_image, top_left, bottom_right, color, thickness)
                    cv2.putText(annotated_image, f'{confidence:.2f}', 
                               (top_left[0], top_left[1] - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Extract structured information
        extracted_info = self._extract_medicine_info(all_texts)
        
        return {
            'success': True,
            'extracted_info': extracted_info,
            'medicine_texts': medicine_texts,
            'all_texts': all_texts,
            'confidence_avg': np.mean([t['confidence'] for t in all_texts]) if all_texts else 0,
        }
    
    def _is_medicine_related(self, text):
        """Enhanced medicine text detection"""
        text_lower = text.lower()
        
        # Check for medicine keywords
        if any(keyword in text_lower for keyword in self.medicine_keywords):
            return True
        
        # Check for dosage patterns
        if re.search(r'\d+\s*(?:mg|mcg|ml|g|%|iu|units?)', text_lower):
            return True
        
        # Check for common medicine patterns
        if re.search(r'[a-z]+\d*[a-z]*\s*\d+', text_lower):
            return True
            
        return False
    
    def _extract_medicine_info(self, ocr_results):
        """Extract structured medicine information"""
        info = {
            'generic_name': '',
            'brand_name': '',
            'dosage_strength': '',
            'dosage_form': '',
            'active_ingredients': [],
            'manufacturer': '',
            'other_info': []
        }
        
        all_text = ' '.join([item['text'] for item in ocr_results])
        individual_texts = [item['text'] for item in ocr_results]
        
        # Enhanced dosage strength extraction
        strength_patterns = [
            r'(\d+\s*(?:mg|mcg|ml|g|%))',
            r'(\d+\s*mg/\s*ml)',
            r'(\d+\s*IU)',
            r'(\d+\s*units?)',
            r'strength\s*[:]?\s*(\d+\s*(?:mg|mcg|ml))',
            r'(\d+\s*mg\s*tablet)',
            r'(\d+\s*mg\s*capsule)',
            r'(\d+\s*mg/\s*\d+\s*ml)',  # Added ratio patterns
            r'(\d+\s*mcg/\s*actuation)'  # Added for sprays/inhalers
        ]
        
        for pattern in strength_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            if matches:
                info['dosage_strength'] = matches[0]
                break
        
        # Enhanced medicine name extraction - EXPANDED
        common_meds = [
            'paracetamol', 'amoxicillin', 'ibuprofen', 'aspirin', 'metformin',
            'insulin', 'morphine', 'warfarin', 'diazepam', 'codeine', 'ciprofloxacin',
            'levothyroxine', 'atorvastatin', 'simvastatin', 'omeprazole', 'losartan',
            'glimepiride', 'gliclazide', 'aflibercept', 'moxifloxacin', 'perindopril',
            'ceftriaxone', 'azithromycin', 'clindamycin', 'doxycycline', 'gentamicin',
            'vancomycin', 'phenytoin', 'carbamazepine', 'valproate', 'furosemide',
            'spironolactone', 'digoxin', 'nitroglycerin', 'salbutamol', 'loratadine',
            'cetirizine', 'ranitidine', 'pantoprazole'
        ]
        
        for text in individual_texts:
            text_lower = text.lower()
            
            # Check for medicine names
            for med in common_meds:
                if med in text_lower and med.capitalize() not in info['active_ingredients']:
                    info['active_ingredients'].append(med.capitalize())
                    if not info['generic_name']:
                        info['generic_name'] = med.capitalize()
            
            # Extract potential brand names
            if (text.isupper() and 2 < len(text) < 25 and 
                not any(char.isdigit() for char in text) and
                text not in ['ML', 'MG', 'MCG', 'IU', 'BP', 'USP', 'FDA']):
                info['brand_name'] = text
            
            # Extract manufacturer information
            if any(keyword in text_lower for keyword in ['ltd', 'inc', 'corporation', 'pharma', 'laboratories', 'company', 'laboratoire']):
                info['manufacturer'] = text
        
        # If no generic name found but we have active ingredients, use the first one
        if not info['generic_name'] and info['active_ingredients']:
            info['generic_name'] = info['active_ingredients'][0]
        
        # Extract dosage form - EXPANDED
        dosage_forms = ['tablet', 'capsule', 'injection', 'solution', 'suspension', 
                       'cream', 'ointment', 'syrup', 'drops', 'spray', 'ampoule',
                       'vial', 'injectable', 'infusion', 'emulsion', 'gel', 'lotion',
                       'inhaler', 'patch', 'suppository', 'powder', 'liquid']
        for form in dosage_forms:
            if form in all_text.lower():
                info['dosage_form'] = form.capitalize()
                break
        
        # Collect other potentially relevant information
        for text in individual_texts:
            if (len(text) > 8 and 
                not any(keyword in text.lower() for keyword in ['www', 'http', '.com']) and
                text not in info.values() and text not in info['active_ingredients']):
                info['other_info'].append(text)
        
        return info