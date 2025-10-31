import joblib
import os
import sys
import traceback

print("=== Model load smoke test ===")
print("Python executable:", sys.executable)

files = [
    'best_category_model.pkl',
    'best_risk_model.pkl',
    'le_category.pkl',
    'le_risk.pkl',
    'tfidf_vectorizer.pkl'
]

for f in files:
    path = os.path.abspath(f)
    print('\n---')
    print('File:', path)
    exists = os.path.exists(path)
    print('Exists:', exists)
    if not exists:
        continue
    try:
        size = os.path.getsize(path)
        print('Size (bytes):', size)
        obj = joblib.load(path)
        print('Loaded OK -> type:', type(obj))
    except Exception as e:
        print('ERROR loading file:', e)
        traceback.print_exc()

print('\nSmoke test completed')
