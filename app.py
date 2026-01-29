import os
import io
import joblib
import numpy as np
import requests
from io import BytesIO

# Optimization for Render's limited CPU resources
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine

# --- CONFIGURATION ---
MODEL_PATH = "resnet50_sports.pth"
ENCODER_PATH = "label_encoder.joblib"
# Replace with your actual Supabase Password
DB_URI = DB_URI = os.getenv("DATABASE_URL", "sqlite:///./test.db")

# --- INITIALIZATION ---
app = FastAPI(title="Sports Classifier API", version="1.0.1")
templates = Jinja2Templates(directory="templates")
device = torch.device("cpu")

# --- DATABASE ENGINE ---
# This is ready for when you want to query your image_metadata table
engine = create_engine(DB_URI)

# --- MODEL ARCHITECTURE ---
class SportsResNet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        # Initialize ResNet50 without pre-trained weights to save memory
        self.base_model = models.resnet50(weights=None)
        self.base_model.fc = nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

# --- LOAD ASSETS SAFELY ---
# 1. Load Label Encoder
if os.path.exists(ENCODER_PATH):
    try:
        le = joblib.load(ENCODER_PATH)
        # Dynamically set class count based on encoder
        num_classes = len(le.classes_)
    except Exception as e:
        print(f"CRITICAL ERROR loading encoder: {e}")
        le = None
        num_classes = 100 # Fallback
else:
    le = None
    num_classes = 100
    print("WARNING: label_encoder.joblib not found")

# 2. Load Model
model = SportsResNet(num_classes=num_classes)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Model weights loaded successfully.")
else:
    print(f"❌ WARNING: Model file not found at {MODEL_PATH}")

# Image Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "version": "1.0.1"})

@app.get("/health")
def health():
    return {"status": "ok", "db_connected": True if engine else False}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if le is None:
        raise HTTPException(status_code=500, detail="Label encoder not available.")
        
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Transformation
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)

    sport_name = le.inverse_transform([pred.item()])[0]
    
    return {
        "sport": sport_name,
        "confidence": "Prediction successful"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)