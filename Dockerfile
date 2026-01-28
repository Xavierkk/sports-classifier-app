# Stage 1: Builder
FROM python:3.10-slim AS builder
WORKDIR /app

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CRITICAL: We install CPU-only torch FIRST and separately
# This prevents the '4.41GB' CUDA version from ever being downloaded
RUN pip install --no-cache-dir --prefix=/install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu

# Then install the rest
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Final image
FROM python:3.10-slim
WORKDIR /app

# Only take what we need
COPY --from=builder /install /usr/local
COPY app.py .
COPY resnet50_sports.pth .
COPY label_encoder.joblib .

ENV MODEL_PATH=resnet50_sports.pth
ENV PYTHONUNBUFFERED=1

# 1 worker is mandatory for 512MB RAM
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app:app", "--bind", "0.0.0.0:$PORT", "--workers", "1", "--timeout", "120"]
