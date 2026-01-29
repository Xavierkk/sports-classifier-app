import os
import io
import json  # Added JSON support
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine

# Optimization for Render's limited CPU resources
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# --- CONFIGURATION ---
MODEL_PATH = "resnet50_sports.pth"
LABELS_PATH = "labels.json"  # Changed from .joblib
DB_URI = os.getenv("DATABASE_URL", "sqlite:///./test.db")

# --- INITIALIZATION ---
app = FastAPI(title="Sports Classifier API", version="1.0.2")
templates = Jinja2Templates(directory="templates")
device = torch.device("cpu")
engine = create_engine(DB_URI)

# --- MODEL ARCHITECTURE ---
class SportsResNet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
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
# 1. Load Labels from JSON
categories = []
if os.path.exists(LABELS_PATH):
    try:
        with open(LABELS_PATH, 'r') as f:
            categories = json.load(f)
        num_classes = len(categories)
        print(f"✅ Successfully loaded {num_classes} categories from JSON.")
    except Exception as e:
        print(f"❌ ERROR loading labels: {e}")
        num_classes = 100
else:
    num_classes = 100
    print(f"⚠️ WARNING: {LABELS_PATH} not found!")

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
    return templates.TemplateResponse("index.html", {"request": request, "version": "1.0.2"})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not categories:
        raise HTTPException(status_code=500, detail="Sport categories not loaded.")
        
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        predicted_index = pred.item()

    # Use the JSON list to get the name instead of le.inverse_transform
    try:
        sport_name = categories[predicted_index]
    except IndexError:
        sport_name = "Unknown Class"
    
    return {
        "sport": sport_name,
        "confidence": "Prediction successful"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)