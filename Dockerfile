# ==========================================
# Multi-stage Dockerfile for Recommendation Service
# Optimized for production deployment
# ==========================================

# Stage 1: Base image with system dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Stage 2: Dependencies installation
FROM base as dependencies

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Stage 3: Application
FROM base as application

# Copy installed dependencies from previous stage
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application code (all Python files from recommendation_service directory)
COPY recommendation_service_enhanced.py .
COPY train_hybrid_enhanced.py .
COPY embedding_loader_enhanced.py .
COPY train_endpoint.py .
COPY product_model.py .
COPY config/ ./config/
COPY models/ ./models/
COPY utils/ ./utils/

# Create directories for models and embeddings if they don't exist
RUN mkdir -p models text_embeddings image_embeddings logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application - file is now in /app root
CMD ["uvicorn", "recommendation_service_enhanced:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
