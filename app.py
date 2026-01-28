import os
import io
import joblib

# Must be set BEFORE importing torch to optimize CPU usage on Render
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# --- CONFIGURATION ---
MODEL_PATH = os.getenv("MODEL_PATH", "resnet50_sports.pth")
# APP_VERSION is passed from your Dockerfile/GitHub Actions
APP_VERSION = os.getenv("APP_VERSION", "v1.0.1") 

# --- INITIALIZATION ---
app = FastAPI(
    title="Sports Classifier API",
    version=APP_VERSION
)

# Set up templates folder
templates = Jinja2Templates(directory="templates")

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

device = torch.device("cpu")
model = SportsResNet(num_classes=100)

# Load model weights safely
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
else:
    print(f"WARNING: Model file not found at {MODEL_PATH}")

# Load Label Encoder
if os.path.exists("label_encoder.joblib"):
    le = joblib.load("label_encoder.joblib")
else:
    le = None
    print("WARNING: label_encoder.joblib not found")

# Image Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serves the interactive web interface."""
    return templates.TemplateResponse("index.html", {"request": request, "version": APP_VERSION})

@app.get("/health")
def health():
    """Health check for Render monitoring."""
    return {"status": "ok", "version": APP_VERSION}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Upload an image to predict the sport."""
    if le is None:
        raise HTTPException(status_code=500, detail="Label encoder not loaded.")
        
    try:
        data = await file.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return {"error": "Invalid image file. Please upload a valid JPG or PNG."}

    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)

    sport_name = le.inverse_transform([pred.item()])[0]
    return {
        "sport": sport_name,
        "version": APP_VERSION
    }

# --- PRODUCTION SERVER SETUP ---
if __name__ == "__main__":
    import uvicorn
    # Render assigns a port via the PORT environment variable
    port = int(os.environ.get("PORT", 10000))
    # Run the server on 0.0.0.0 to be accessible externally
    uvicorn.run(app, host="0.0.0.0", port=port)