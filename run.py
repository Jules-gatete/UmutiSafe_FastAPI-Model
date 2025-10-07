# run.py
import uvicorn
import os

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./data/processed', exist_ok=True)
    
    # Run the FastAPI application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )