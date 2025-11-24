# ==========================================
# Dockerfile for Recommendation Service
# Production deployment for Render.com
# ==========================================

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python modules (these are the actual files in recommendation_service/)
COPY recommendation_service_enhanced.py .
COPY product_model.py .
COPY train_endpoint.py .

# Copy package directories with all Python files
COPY config/ ./config/
COPY models/ ./models/
COPY utils/ ./utils/

# Create necessary directories for runtime data
RUN mkdir -p models text_embeddings image_embeddings logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
# The main FastAPI app is in recommendation_service_enhanced.py with app variable
CMD ["uvicorn", "recommendation_service_enhanced:app", "--host", "0.0.0.0", "--port", "8000"]
