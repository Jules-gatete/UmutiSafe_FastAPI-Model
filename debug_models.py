import joblib
import os
import sklearn
import numpy

print(f"scikit-learn version: {sklearn.__version__}")
print(f"numpy version: {numpy.__version__}")
print(f"joblib version: {joblib.__version__}")

base_path = './models/'
models_to_test = [
    'best_category_model.pkl',
    'best_risk_model.pkl', 
    'le_category.pkl',
    'le_risk.pkl',
    'tfidf_vectorizer.pkl'
]

for model_file in models_to_test:
    model_path = os.path.join(base_path, model_file)
    if os.path.exists(model_path):
        print(f"\n Testing: {model_file}")
        try:
            model = joblib.load(model_path)
            print(f"   ✅ Loaded successfully")
            print(f"   📊 Type: {type(model)}")
        except Exception as e:
            print(f"   ❌ Failed to load: {e}")
    else:
        print(f"   ❌ File not found: {model_file}")