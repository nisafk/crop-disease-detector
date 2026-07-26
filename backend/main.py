"""
main.py - FastAPI Backend Server for Crop Disease Detection

WHAT THIS FILE DOES:
This is the "brain server" of our app.
- It runs on a server (like Render.com)
- It exposes ONE main endpoint: POST /predict
- The frontend sends a leaf image to this endpoint
- This file loads the AI model, runs the image through it, and returns the result

HOW A REQUEST FLOWS:
1. User uploads image on frontend (React)
2. React sends POST request to https://your-backend.onrender.com/predict
3. This file receives the image
4. Preprocesses it (resize to 224x224, normalize)
5. Runs it through the PyTorch model
6. Gets top prediction + confidence score
7. Looks up disease info from disease_info.json
8. Returns JSON response to frontend
9. Frontend displays the result

TO RUN LOCALLY:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
Then open: http://localhost:8000/docs (auto-generated API documentation!)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
# FastAPI  = the web framework
# File     = tells FastAPI to expect a file in the request
# UploadFile = the type for uploaded files
# HTTPException = for sending error responses

from fastapi.middleware.cors import CORSMiddleware
# CORS = Cross-Origin Resource Sharing
# Without this, your React frontend (on vercel.com) 
# cannot talk to your backend (on render.com)
# Browsers block requests between different domains by default

from fastapi.responses import JSONResponse
# JSONResponse = sends data back as JSON ({"key": "value"} format)

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image    # Pillow: opens and manipulates images
import io                # For handling file bytes in memory
import json
import os

# ============================================================
# STEP 1: CREATE THE FASTAPI APP
# ============================================================

app = FastAPI(
    title="Crop Disease Detector API",
    description="Upload a leaf image and get AI-powered disease detection",
    version="1.0.0"
)

# ============================================================
# STEP 2: CONFIGURE CORS
# This allows your frontend to call this backend
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allow ALL origins (any frontend can call this)
                               # In production, change to your specific Vercel URL
    allow_credentials=True,
    allow_methods=["*"],       # Allow GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],       # Allow all request headers
)

# ============================================================
# STEP 3: LOAD MODEL AND DATA AT STARTUP
# We load the model ONCE when the server starts
# (Not on every request - that would be too slow!)
# ============================================================

# Paths to our files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pth")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "model", "class_names.json")
DISEASE_INFO_PATH = os.path.join(BASE_DIR, "model", "disease_info.json")

NUM_CLASSES = 38
CONFIDENCE_THRESHOLD = 0.70   # If confidence < 70%, say "uncertain"

# Device: use GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Backend using device: {DEVICE}")

def load_model():
    """
    Loads the trained ResNet50 model from the .pth file.
    Returns the model ready for inference.
    """
    # Recreate the same architecture we used in training
    model = models.resnet50(weights=None)   # No pretrained weights this time
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(2048, NUM_CLASSES)
    )
    
    # Load our trained weights into this architecture
    # map_location=DEVICE handles loading on CPU even if trained on GPU
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()   # Set to evaluation mode (disables dropout)
    
    print("✅ Model loaded successfully!")
    return model

# Load class names {0: "Apple___Apple_scab", 1: "Apple___Black_rot", ...}
def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as f:
        raw = json.load(f)
    # JSON keys are strings, convert to int
    return {int(k): v for k, v in raw.items()}

# Load disease treatment information
def load_disease_info():
    with open(DISEASE_INFO_PATH, "r") as f:
        return json.load(f)

# Load everything at startup
print("Loading model and data...")
try:
    model = load_model()
    idx_to_class = load_class_names()
    disease_info = load_disease_info()
    print("All data loaded. Server ready!")
    MODEL_LOADED = True
except Exception as e:
    print(f"⚠️ Warning: Could not load model: {e}")
    print("Server starting without model. /predict will return demo data.")
    MODEL_LOADED = False
    model = None
    idx_to_class = {}
    disease_info = {}

# ============================================================
# STEP 4: IMAGE PREPROCESSING
# Prepare uploaded image for the model
# (Same transforms as validation in training)
# ============================================================

preprocess = transforms.Compose([
    transforms.Resize(256),          # Resize smallest dimension to 256px
    transforms.CenterCrop(224),      # Crop center 224x224
    transforms.ToTensor(),           # Convert to tensor (0-1 float values)
    transforms.Normalize(            # Normalize using ImageNet stats
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Takes raw image bytes (from upload) and returns a tensor ready for the model.
    
    Steps:
    1. Convert bytes to PIL Image object
    2. Convert to RGB (in case it's RGBA or grayscale)
    3. Apply transforms (resize, crop, normalize)
    4. Add batch dimension: [3, 224, 224] → [1, 3, 224, 224]
       (Model expects batches, even if batch size = 1)
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = preprocess(image)
    tensor = tensor.unsqueeze(0)   # Add batch dimension
    return tensor.to(DEVICE)

# ============================================================
# STEP 5: API ENDPOINTS
# ============================================================

@app.get("/")
def root():
    """
    Health check endpoint.
    When someone visits https://your-backend.onrender.com/ they see this.
    Useful to check if the server is alive.
    """
    return {
        "status": "running",
        "model_loaded": MODEL_LOADED,
        "message": "Crop Disease Detector API is live!"
    }

@app.get("/health")
def health_check():
    """Simple health check for deployment monitoring."""
    return {"status": "healthy", "model_loaded": MODEL_LOADED}

@app.get("/classes")
def get_classes():
    """Returns all 38 disease classes the model can detect."""
    return {
        "total_classes": NUM_CLASSES,
        "classes": list(idx_to_class.values()) if idx_to_class else []
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    MAIN ENDPOINT - Takes a leaf image, returns disease prediction.
    
    Input:  Multipart form with image file
    Output: JSON with disease name, confidence, and treatment info
    
    Example response:
    {
        "disease_key": "Tomato___Early_blight",
        "disease_name": "Tomato Early Blight",
        "confidence": 0.94,
        "confidence_percent": "94.2%",
        "is_healthy": false,
        "description": "Dark brown target-like spots...",
        "cause": "Fungus (Alternaria solani)",
        "treatment": "Apply chlorothalonil or mancozeb...",
        "prevention": "Mulch soil. Water at base..."
    }
    """
    
    # Validate file type - only accept images
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image. Got: {file.content_type}"
        )
    
    # If model not loaded, return demo response
    if not MODEL_LOADED:
        return JSONResponse(content={
            "disease_key": "Tomato___Early_blight",
            "disease_name": "Tomato Early Blight (Demo Mode)",
            "confidence": 0.87,
            "confidence_percent": "87.0%",
            "is_healthy": False,
            "description": "This is a demo response. Model not loaded.",
            "cause": "Fungus (Alternaria solani)",
            "treatment": "Apply chlorothalonil or mancozeb fungicides.",
            "prevention": "Mulch soil. Water at base. Rotate crops annually.",
            "note": "Demo mode - train and add best_model.pth to get real predictions"
        })
    
    try:
        # Read the uploaded file bytes
        image_bytes = await file.read()
        
        # Preprocess: convert bytes → tensor
        tensor = preprocess_image(image_bytes)
        
        # Run through model (no gradient needed for inference)
        with torch.no_grad():       # torch.no_grad() = faster, less memory
            outputs = model(tensor) # Shape: [1, 38] - score for each class
            
            # Convert raw scores to probabilities using Softmax
            # Softmax makes all values between 0-1 and sum to 1
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get top 3 predictions (most confident first)
            top3_prob, top3_idx = torch.topk(probabilities, 3)
        
        # Get the #1 prediction
        top_prob = top3_prob[0].item()        # Confidence score (0.0 to 1.0)
        top_idx = top3_idx[0].item()          # Index of predicted class (0-37)
        top_class_key = idx_to_class[top_idx] # e.g., "Tomato___Early_blight"
        
        # If confidence is too low, don't guess
        if top_prob < CONFIDENCE_THRESHOLD:
            return JSONResponse(content={
                "disease_key": "uncertain",
                "disease_name": "Cannot Determine",
                "confidence": top_prob,
                "confidence_percent": f"{top_prob*100:.1f}%",
                "is_healthy": None,
                "description": f"Model confidence is too low ({top_prob*100:.1f}%). The image may be unclear, not a leaf, or show a disease not in our database.",
                "cause": "N/A",
                "treatment": "Please consult a local agriculture officer.",
                "prevention": "Take a clear, close-up photo of the affected leaf in good lighting.",
                "top3": [
                    {"class": idx_to_class[top3_idx[i].item()], "confidence": f"{top3_prob[i].item()*100:.1f}%"}
                    for i in range(3)
                ]
            })
        
        # Get disease info from our JSON database
        info = disease_info.get(top_class_key, {})
        is_healthy = "healthy" in top_class_key.lower()
        
        # Build top-3 alternatives for display
        top3_results = [
            {
                "class": idx_to_class[top3_idx[i].item()],
                "display_name": disease_info.get(idx_to_class[top3_idx[i].item()], {}).get("display_name", idx_to_class[top3_idx[i].item()]),
                "confidence": f"{top3_prob[i].item()*100:.1f}%"
            }
            for i in range(3)
        ]
        
        return JSONResponse(content={
            "disease_key": top_class_key,
            "disease_name": info.get("display_name", top_class_key),
            "confidence": top_prob,
            "confidence_percent": f"{top_prob*100:.1f}%",
            "is_healthy": is_healthy,
            "description": info.get("description", ""),
            "cause": info.get("cause", ""),
            "treatment": info.get("treatment", ""),
            "prevention": info.get("prevention", ""),
            "top3_predictions": top3_results
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
