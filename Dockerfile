# Stage 1: Builder
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies if needed (e.g., libgl1 for OpenCV, though not needed for PIL)
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 1. Install CPU-only Torch first to avoid massive CUDA binaries
# 2. Install the rest of the requirements
RUN pip install --no-cache-dir --prefix=/install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Final image
FROM python:3.10-slim

WORKDIR /app

# Copy only the necessary site-packages from builder
COPY --from=builder /install /usr/local

# Copy application code and model files
COPY app.py .
COPY resnet50_sports.pth .
COPY label_encoder.joblib .

ENV MODEL_PATH=resnet50_sports.pth
ENV PYTHONUNBUFFERED=1

# Use 1 worker to stay within RAM limits; increase timeout for model loading
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app:app", "--bind", "0.0.0.0:$PORT", "--workers", "1", "--timeout", "120"]
