# Stage 1: Builder (temporary, will install dependencies)
FROM python:3.10-slim AS builder

WORKDIR /app

# Copy only requirements first (faster rebuilds)
COPY requirements.txt .

# Install dependencies into a local folder to copy later
RUN python -m pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Final image
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code and model files
COPY app.py .
COPY resnet50_sports.pth .
COPY label_encoder.joblib .

# Set environment variable
ENV MODEL_PATH=resnet50_sports.pth

# Expose a port (optional, for clarity)
EXPOSE 80

# Use $PORT environment variable for Render
CMD gunicorn -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:$PORT --workers 2
