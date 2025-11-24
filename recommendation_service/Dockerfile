# ==========================================
# Dockerfile for Recommendation Service (Render)
# FIXED VERSION
# ==========================================

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code from this directory
# (since Dockerfile is already inside /recommendation_service)
COPY . .

# Ensure runtime directories exist
RUN mkdir -p models text_embeddings image_embeddings logs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 💥 IMPORTANT: use correct filename and module name!
# Your file is "recommendation_Service_enhenced.py"
CMD ["uvicorn", "recommendation_Service_enhenced:app", "--host", "0.0.0.0", "--port", "8000"]
