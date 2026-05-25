# syntax=docker/dockerfile:1.6
FROM python:3.12-slim

# System libs required by:
#  - opencv-python                 → libgl1, libglib2.0-0
#  - kokoro / misaki G2P phonemes  → espeak-ng
#  - insightface wheel install     → build-essential, cmake (only used if pip
#                                    falls back to building from source)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake \
        libgl1 libglib2.0-0 \
        espeak-ng \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python deps ---
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Bake the small FER+ emotion ONNX (~34 MB) into the image at a path that the
# /app/models volume mount can't shadow, so emotion-driven triggers work the
# moment the container starts.
RUN mkdir -p /app/baked_models && \
    curl -fsSL -o /app/baked_models/emotion-ferplus-8.onnx \
      https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx
ENV EMOTION_MODEL_PATH=/app/baked_models/emotion-ferplus-8.onnx

# --- App code ---
COPY companion_ai/ ./companion_ai/
COPY web/ ./web/

# Larger models (Kokoro voices, InsightFace buffalo_l) auto-download on first
# use into /app/models, which is mounted as a volume so they persist across
# restarts and stay out of the image itself.
RUN mkdir -p /app/models

EXPOSE 8000
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000"]
