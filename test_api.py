# test_api.py
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print("✅ Health Check Response:")
        print(json.dumps(response.json(), indent=2))
        print("-" * 50)
    except Exception as e:
        print(f"❌ Health check failed: {e}")

def test_text_prediction():
    """Test text prediction endpoint"""
    try:
        data = {
            "generic_name": "Amoxicillin 500mg",
            "brand_name": "Amoxil",
            "dosage_form": "Capsules",
            "packaging_type": "Blister"
        }
        response = requests.post(f"{BASE_URL}/api/predict/text", data=data)
        print("✅ Text Prediction Response:")
        print(json.dumps(response.json(), indent=2))
        print("-" * 50)
    except Exception as e:
        print(f"❌ Text prediction failed: {e}")

def test_guidelines():
    """Test guidelines endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/api/guidelines")
        data = response.json()
        print("✅ Guidelines Response:")
        print(f"Total categories: {data['total_categories']}")
        print(f"Categories: {', '.join(data['categories'])}")
        print("-" * 50)
    except Exception as e:
        print(f"❌ Guidelines failed: {e}")

def test_batch_prediction():
    """Test batch prediction endpoint"""
    try:
        batch_data = [
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
        response = requests.post(f"{BASE_URL}/api/predict/batch", json=batch_data)
        print("✅ Batch Prediction Response:")
        print(json.dumps(response.json(), indent=2))
        print("-" * 50)
    except Exception as e:
        print(f"❌ Batch prediction failed: {e}")

def test_all_endpoints():
    """Test all API endpoints"""
    print("🚀 Testing UmutiSafe API Endpoints")
    print("=" * 60)
    
    # Make sure server is running first
    try:
        requests.get(f"{BASE_URL}/health", timeout=5)
    except:
        print("❌ API server is not running. Please start the server first.")
        print("   Run: python main.py")
        return
    
    test_health()
    test_text_prediction()
    test_guidelines()
    test_batch_prediction()
    
    print(" API Testing Complete!")
    print(f" Open Swagger UI at: {BASE_URL}/docs")

if __name__ == "__main__":
    test_all_endpoints()