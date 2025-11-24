# ==========================================
# Dockerfile for Futurist Recommendation Service
# Production deployment for Render.com
# ==========================================

FROM python:3.11-slim

# -------------------------
# Environment
# -------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# -------------------------
# System dependencies
# -------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Working directory
# -------------------------
WORKDIR /app

# -------------------------
# Install Python dependencies
# -------------------------
COPY recommendation_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------
# Copy the full app directory
# -------------------------
COPY recommendation_service/ ./recommendation_service/

# -------------------------
# Create runtime directories
# -------------------------
RUN mkdir -p models text_embeddings image_embeddings logs

# -------------------------
# Expose port
# -------------------------
EXPOSE 8000

# -------------------------
# Health check
# -------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# -------------------------
# Run the FastAPI app
# Use full module path for imports
# -------------------------
CMD ["uvicorn", "recommendation_service.recommendation_service_enhanced:app", "--host", "0.0.0.0", "--port", "8000"]
