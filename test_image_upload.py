import requests

def test_image_upload():
    url = "http://localhost:8000/api/predict/image"
    
    # You can test with a sample image file
    # Replace 'test_medicine.jpg' with an actual image file path
    try:
        with open('test_medicine.jpg', 'rb') as f:
            files = {'file': ('test_medicine.jpg', f, 'image/jpeg')}
            response = requests.post(url, files=files)
        
        print("✅ Image Upload Response:")
        print(response.json())
    except FileNotFoundError:
        print("❌ No test image found. Please use Swagger UI to upload an image.")
        print("🌐 Open: http://localhost:8000/docs")

if __name__ == "__main__":
    test_image_upload()