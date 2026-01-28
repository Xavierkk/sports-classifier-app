# Stage 1: Builder
FROM python:3.10-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Final image
FROM python:3.10-slim
WORKDIR /app

ARG APP_VERSION=v1.0.0
ENV APP_VERSION=$APP_VERSION


COPY --from=builder /install /usr/local
COPY app.py .
COPY resnet50_sports.pth .
COPY label_encoder.joblib .

ENV MODEL_PATH=resnet50_sports.pth
ENV PYTHONUNBUFFERED=1

# 1 worker is mandatory for 512MB RAM
CMD gunicorn -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
