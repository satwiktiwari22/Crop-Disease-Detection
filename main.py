from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import json
import io
import torch
import torchvision.transforms as transforms
import logging
import time
import os
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("crop-disease-api")


# Load disease suggestions
try:
    with open("suggestions.json", "r", encoding="utf-8") as f:
        DISEASE_SUGGESTIONS = json.load(f)
    logger.info("Disease suggestions loaded successfully")
except Exception as e:
    logger.error(f"Failed to load suggestions: {str(e)}")
    DISEASE_SUGGESTIONS = {}

# Class mapping for model predictions
CLASS_MAPPING = {
    0: "American Bollworm on Cotton",
    1: "Anthracnose on Cotton",
    2: "Army worm",
    3: "Becterial Blight in Rice",
    4: "Brownspot",
    5: "Common_Rust",
    6: "Cotton Aphid",
    7: "Flag Smut",
    8: "Gray_Leaf_Spot",
    9: "Healthy Maize",
    10: "Healthy Wheat",
    11: "Healthy cotton",
    12: "Leaf Curl",
    13: "Leaf smut",
    14: "Mosaic sugarcane",
    15: "RedRot sugarcane",
    16: "RedRust sugarcane",
    17: "Rice Blast",
    18: "Sugarcane Healthy",
    19: "Tungro",
    20: "Wheat Brown leaf Rust",
    21: "Wheat Stem fly",
    22: "Wheat aphid",
    23: "Wheat black rust",
    24: "Wheat leaf blight",
    25: "Wheat mite",
    26: "Wheat powdery mildew",
    27: "Wheat scab",
    28: "Wheat___Yellow_Rust",
    29: "Wilt",
    30: "Yellow Rust Sugarcane",
    31: "bacterial_blight in Cotton",
    32: "bollrot on Cotton",
    33: "bollworm on Cotton",
    34: "cotton mealy bug",
    35: "cotton whitefly",
    36: "maize ear rot",
    37: "maize fall armyworm",
    38: "maize stem borer",
    39: "pink bollworm in cotton",
    40: "red cotton bug",
    41: "thirps on cotton",
}

# Load the TorchScript model with error handling
try:
    model_path = "model/efficientnet_b2_scripted.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = torch.jit.load(model_path)
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    model = None

# Define image preprocessing (must match training pipeline)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Define FastAPI app
app = FastAPI(
    title="Crop Disease Detection API",
    description="API for detecting diseases in crop images",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify allowed origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def home():
    """API health check endpoint"""
    return {"status": "online", "model_loaded": model is not None, "version": "1.0.0"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Predict crop disease from an uploaded image
    """
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    start_time = time.time()

    try:
        # Read the image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess the image
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()
            confidence = float(probabilities[predicted_class].item())

        # Get class name
        class_name = CLASS_MAPPING.get(predicted_class, f"Unknown ({predicted_class})")

        # Get suggestions for the predicted class
        suggestions = DISEASE_SUGGESTIONS.get(str(predicted_class), {})

        processing_time = (time.time() - start_time) * 1000  # ms

        return {
            "prediction": {
                "class_id": int(predicted_class),
                "class_name": class_name,
                "confidence": round(confidence * 100, 2),  # as percentage
            },
            "suggestions": suggestions,
            "processing_time_ms": round(processing_time, 2),
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch/")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Batch predict diseases for multiple crop images
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")

    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")

    results = []

    for file in files:
        try:
            # Process each file similar to the single prediction endpoint
            if not file.content_type.startswith("image/"):
                results.append(
                    {"filename": file.filename, "error": "Not an image file"}
                )
                continue

            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                predicted_class = torch.argmax(probabilities).item()
                confidence = float(probabilities[predicted_class].item())

            class_name = CLASS_MAPPING.get(
                predicted_class, f"Unknown ({predicted_class})"
            )

            # Get suggestions for the predicted class
            suggestions = DISEASE_SUGGESTIONS.get(str(predicted_class), {})

            results.append(
                {
                    "filename": file.filename,
                    "class_id": int(predicted_class),
                    "class_name": class_name,
                    "confidence": round(confidence * 100, 2),
                    "suggestions": suggestions,
                }
            )

        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

    return {"predictions": results}
