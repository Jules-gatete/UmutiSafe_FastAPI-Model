# UmutiSafe — App Documentation

This document expands on the README with concrete commands, file mappings, testing notes, deployment steps, and troubleshooting guidance for the FastAPI backend.

Contents

- Architecture and file map
- How the models were trained & saved (summary)
- How the app loads models at startup
- How the app was tested (unit, smoke, integration)
- How to run locally (detailed)
- Deployment steps (Docker, Render) with notes
- Common errors & fixes encountered during deployment
- How to remove model files from git and alternatives (S3, Git LFS)
- Improvement suggestions and next steps

---

## Architecture and file map

- `main.py` — FastAPI entrypoint. Sets up CORS middleware, health endpoints, loads system components (models), and exposes prediction endpoints.
- `enhanced_ocr_processor.py` — OCR extraction using EasyOCR plus image preprocessing and helper utilities.
- `medicine_disposal_predictor.py` — Creates prediction-ready features and calls the ML models. Handles mapping of predictions back to labels and composing safety guidance.
- `disposal_guidelines_db.py` (in `data/processed/` or repo root) — structured guidance data by category.
- `requirements.txt` — Python package dependencies used for Docker build and venv installs.
- `Dockerfile` — builds the container image; includes system-level packages required by OpenCV.
- `test_api.py`, `test_image_upload.py`, `smoke_test_load_models.py` — tests and smoke checks.

---

## How the models were produced & saved (summary)

The models are sklearn-compatible estimators saved via `joblib`. The training pipeline (not included here) produced:

- `tfidf_vectorizer.pkl` — TF-IDF transformer fit on combined text fields.
- `best_category_model.pkl` — classifier for disposal categories.
- `best_risk_model.pkl` — classifier for risk levels.
- `le_category.pkl`, `le_risk.pkl` — LabelEncoders for mapping between encoded classes and strings.

If you need the training code, this repository's notebooks (in `notebooks/`) show experimentation; export the final artifacts with `joblib.dump()`.

---

## How the app loads models at startup

`main.py` -> `load_system_components()` does the following:

1. Calls `ensure_model_files_from_s3_if_needed()` which =
   - Checks for missing `.pkl` artifacts in `./`, `./models/`, `./app/models/`.
   - If missing and `MODELS_S3_BUCKET` is set, it attempts to download missing artifacts from S3 using `boto3`.
2. Searches for the five required model files in configured search paths.
3. Uses `joblib.load()` to load the models and returns a `components` dict.

If components are missing, startup will return degraded and the API will return 503 for prediction endpoints.

---

## How the app was tested

Included artifacts and how to use them:

- `smoke_test_load_models.py` — simple script to verify `joblib.load()` can read the `.pkl` artifacts in the environment (helps confirm file paths and Python env compatibility).
- `test_api.py` — uses `TestClient` or makes HTTP calls to the running server to validate endpoints (ensure server is running or tests start it via FastAPI's TestClient).
- `test_image_upload.py` — tests multipart image uploads to `/api/predict/image` using the sample `test_Images/medecine.jfif`.

Testing commands

```powershell
# Run smoke model load test
python smoke_test_load_models.py

# Run pytest suite
pytest -q
```

Manual checks

- Use curl or Postman to POST to `/api/predict/text` and `/api/predict/image` and inspect JSON outputs.

---

## Run locally — detailed

1) Create venv and install:

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Start server:

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

3) Health check:

```powershell
curl -i http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/api/health
```

4) Example prediction (text):

```powershell
curl -X POST "http://127.0.0.1:8000/api/predict/text" -F "generic_name=Amoxicillin 500mg" -F "brand_name=Amoxil" -F "dosage_form=Tablet"
```

5) Example prediction (image):

```powershell
curl -X POST "http://127.0.0.1:8000/api/predict/image" -F "file=@test_Images/medecine.jfif;type=image/jpeg"
```

---

## Deployment steps (Docker & Render) — practical notes

1. Ensure `requirements.txt` is clean (no code fences), and contains `boto3` if using S3 downloads.
2. Dockerfile must install system libs OpenCV requires (we added libgl and related packages).
3. If model files are committed, Docker `COPY . .` will include them. Otherwise, set S3 env vars in the target environment.
4. Recommended Render env vars:

- `DEBUG=False`
- `LOG_LEVEL=INFO`
- `MODELS_IN_REPO=yes` (if artifacts are included) or `no` if using S3
- `MODELS_S3_BUCKET`, `MODELS_S3_PREFIX` (if using S3)
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (mark secrets)

Health path: configure Render to use `/health` or `/api/health`.

---

## Common errors & fixes encountered during deployment

- ImportError: `libGL.so.1` not found when `import cv2` — fixed by installing system packages in Dockerfile: `libgl1`, `libglib2.0-0`, `libsm6`, `libxrender1`, `libxext6`.
- Pip errors during Docker build (invalid `requirements.txt`) — ensure the file contains only newline-separated package specifiers.
- Pylance "could not be resolved" for `boto3` — install dependencies in your venv: `pip install -r requirements.txt`.

---

## How to remove model files from git (if desired)

If you want to keep model binaries out of the repository and store them externally:

1. Add `*.pkl` and `models/` to `.gitignore` (already done).
2. Remove the files from git tracking (local):

```powershell
git rm --cached best_category_model.pkl best_risk_model.pkl le_category.pkl le_risk.pkl tfidf_vectorizer.pkl
git commit -m "Remove model artifacts from repo; use S3 for model storage"
git push origin main
```

3. Optionally, add the artifacts to S3 and set `MODELS_S3_BUCKET` in your deployment environment so the app downloads models at startup.

Note: If model files are large and already in git history, consider using `git filter-branch` or `git filter-repo` to rewrite history, or use Git LFS going forward.

---

## Improvements & next steps (concrete)

- Move models to S3 and remove from repo. Use `ensure_model_files_from_s3_if_needed()` already implemented in `main.py`.
- Add an automated CI pipeline: GitHub Actions to build Docker image and push to a registry on pushes to `main`.
- Add integration tests that run a container and run smoke tests against it (e.g., docker-compose or test matrix).

---

If you want, I can perform the optional cleanup (remove model files from git, add GitHub Actions, or create a `deploy.md` for Render UI steps). Tell me which item to do next.