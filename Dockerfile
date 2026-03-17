

# ═══════════════════════════════════════════════════════════════════════════
# Fraud Detection API - Dockerfile
# ═══════════════════════════════════════════════════════════════════════════
# Production-ready Docker image for FastAPI fraud detection service
# Optimized for: Small size, fast builds, production reliability
# 
# FIXED: Module import resolution - working directory set correctly
# ═══════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────
# Stage 1: Builder (Dependencies Installation)
# ───────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# ───────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime (Production Image)
# ───────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy ALL application code to /app
COPY . .

# Create necessary directories
RUN mkdir -p logs trained_models results/plots

# ───────────────────────────────────────────────────────────────────────────
# CRITICAL FIX: Add current directory to PYTHONPATH
# ───────────────────────────────────────────────────────────────────────────
# This allows "from config.settings import ..." to work correctly
# IMPORTANT: Ensure model files exist in trained_models/ before deploying
ENV PYTHONPATH=/app:$PYTHONPATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/trained_models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

# Make start script executable
RUN chmod +x start.sh

# Run with start script (better than direct uvicorn - sets PYTHONPATH correctly)
CMD ["./start.sh"]


