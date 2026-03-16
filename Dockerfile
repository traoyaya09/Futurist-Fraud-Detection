
# ═══════════════════════════════════════════════════════════════════════════
# Fraud Detection API - Dockerfile
# ═══════════════════════════════════════════════════════════════════════════
# Production-ready Docker image for FastAPI fraud detection service
# Optimized for: Small size, fast builds, production reliability
# ═══════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────
# Stage 1: Builder (Dependencies Installation)
# ───────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

# Set working directory
WORKDIR /app

# DEBUG: List files in build context to verify requirements.txt is present
RUN echo "=== DEBUG: Listing build context files ===" && ls -la / || true

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
# Using explicit relative path to avoid any path resolution issues
COPY ./requirements.txt .

# DEBUG: Verify requirements.txt was copied
RUN echo "=== DEBUG: Verifying requirements.txt ===" && ls -la requirements.txt && head -5 requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# ───────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime (Production Image)
# ───────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs trained_models results/plots

# Set Python environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/trained_models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run uvicorn server
CMD ["uvicorn", "fraud_detection_service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

