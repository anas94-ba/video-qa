FROM python:3.12-slim

# ffmpeg is needed for the no-captions fallback (audio download + transcription)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install dependencies first so Docker caches this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY static ./static

EXPOSE 8000

# 0.0.0.0 is required so the container is reachable from outside
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]