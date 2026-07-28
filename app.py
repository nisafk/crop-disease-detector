# app.py - Hugging Face entry point
import uvicorn
from backend.main import app

# Hugging Face runs on port 7860 by default
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
