import os
import io
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine

# --- CONFIGURATION ---
# Render's CPU optimization
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

MODEL_PATH = "resnet50_sports.pth"
LABELS_PATH = "labels.json"
DB_URI = os.getenv("DATABASE_URL", "sqlite:///./test.db")

# --- INITIALIZATION ---
app = FastAPI(title="Sports Classifier API", version="1.0.3")
templates = Jinja2Templates(directory="templates")
device = torch.device("cpu")
engine = create_engine(DB_URI)

# --- MODEL ARCHITECTURE ---
class SportsResNet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        # Load weights=None because we will load our custom .pth weights
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

# --- ASSET LOADING ---

# 1. Load Labels (Map Dictionary)
categories_map = {}
if os.path.exists(LABELS_PATH):
    try:
        with open(LABELS_PATH, 'r') as f:
            # Expected format: {"0": "air hockey", "1": "archery"...}
            categories_map = json.load(f)
        num_classes = len(categories_map)
        print(f"✅ Successfully loaded {num_classes} categories.")
    except Exception as e:
        print(f"❌ Error loading {LABELS_PATH}: {e}")
        num_classes = 100
else:
    print(f"⚠️ Warning: {LABELS_PATH} not found. Defaulting to 100 classes.")
    num_classes = 100

# 2. Load Model Weights
model = SportsResNet(num_classes=num_classes)
if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval() # Set to evaluation mode for inference
        print(f"✅ Model weights loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading model weights: {e}")
else:
    print(f"❌ Critical Error: Model file {MODEL_PATH} not found!")

# --- PREPROCESSING ---
# Must match the training transformation exactly
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # ResNet50 requirement: Normalization to ImageNet standards
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "version": "1.0.3",
        "status": "Ready" if categories_map else "Error: Labels Missing"
    })

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not categories_map:
        raise HTTPException(status_code=500, detail="Sport categories map not loaded.")
        
    try:
        # Read and open image
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Transform and add batch dimension
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        # Convert logits to probabilities
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, pred = torch.max(probabilities, 1)
        
        predicted_idx_str = str(pred.item())

    # Map the predicted index to the name using the JSON dictionary
    sport_name = categories_map.get(predicted_idx_str, "Unknown Category")
    confidence_score = f"{confidence.item() * 100:.2f}%"

    return {
        "sport": sport_name,
        "confidence": confidence_score,
        "status": "Success"
    }

if __name__ == "__main__":
    import uvicorn
    # Use the PORT provided by Render, or default to 10000 for local
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)