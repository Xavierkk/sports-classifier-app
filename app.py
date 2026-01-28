import os
# Must be set BEFORE importing torch
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import joblib

# --- UPDATED: Read dynamic settings ---
MODEL_PATH = os.getenv("MODEL_PATH", "resnet50_sports.pth")
# This grabs the v1.0.x tag from Docker during build
APP_VERSION = os.getenv("APP_VERSION", "v1.0.1") 

class SportsResNet(nn.Module):
    # ... keep your class exactly as it is ...
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

app = FastAPI(title="Sports Classifier", version=APP_VERSION)

@app.get("/")
def home():
    return {
        "message": "Sports Classifier API is Live!",
        "usage": "Visit /docs to test the model.",
        "version": APP_VERSION  # UPDATED: Now uses the auto-incremented version
    }

# ... keep your model loading logic ...
device = torch.device("cpu")
model = SportsResNet(num_classes=100)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

le = joblib.load("label_encoder.joblib")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        data = await file.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return {"error": "Invalid image file"}

    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)

    sport_name = le.inverse_transform([pred.item()])[0]
    return {"sport": sport_name}

# --- NEW: This block is crucial for Render ---
if __name__ == "__main__":
    import uvicorn
    # Grab the port Render provides, or default to 10000
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
