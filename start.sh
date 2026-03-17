
#!/bin/bash
# =============================================================================
# Fraud Detection API Startup Script
# Ensures proper Python path configuration for Render deployment
# =============================================================================

set -e  # Exit on any error

echo "═══════════════════════════════════════════════════════════════════════"
echo "🚀 FRAUD DETECTION API - STARTUP"
echo "═══════════════════════════════════════════════════════════════════════"

# Display environment info
echo "📁 Working Directory: $(pwd)"
echo "🐍 Python Version: $(python --version)"
echo "📦 Pip Version: $(pip --version)"
echo "🌍 PYTHONPATH: ${PYTHONPATH:-'(not set)'}"
echo "🔢 PORT: ${PORT:-8000}"
echo "👥 WEB_CONCURRENCY: ${WEB_CONCURRENCY:-1}"
echo ""

# Ensure we're in the correct directory
cd /app || {
    echo "❌ ERROR: /app directory not found!"
    exit 1
}

# Set PYTHONPATH to include /app
export PYTHONPATH="/app:${PYTHONPATH}"
echo "✅ PYTHONPATH set to: $PYTHONPATH"

# List directory contents for debugging
echo ""
echo "📂 Files in /app:"
ls -lah
echo ""

# Verify fraud_detection_service.py exists
if [ ! -f "fraud_detection_service.py" ]; then
    echo "❌ ERROR: fraud_detection_service.py not found in /app!"
    echo "   Available Python files:"
    find . -name "*.py" -type f
    exit 1
fi

echo "✅ fraud_detection_service.py found"

# Check if trained models exist
if [ -d "trained_models" ]; then
    echo "✅ trained_models/ directory exists"
    echo "   Model files:"
    ls -lah trained_models/
else
    echo "⚠️  WARNING: trained_models/ directory not found"
    echo "   Creating directory..."
    mkdir -p trained_models
fi

# Create necessary directories
mkdir -p logs results/plots
echo "✅ Directories created: logs, results/plots"

# Test import (quick smoke test)
echo ""
echo "🧪 Testing Python imports..."
python -c "import fraud_detection_service; print('✅ fraud_detection_service module imports successfully')" || {
    echo "❌ ERROR: Failed to import fraud_detection_service"
    echo "   Trying to diagnose..."
    python -c "import sys; print('Python path:', sys.path)"
    exit 1
}

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "🚀 STARTING UVICORN SERVER"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# Get port from Render's PORT env var (default 8000 for local)
API_PORT="${PORT:-8000}"
echo "🔌 Server will bind to port: $API_PORT"
echo ""

# Start Uvicorn with Python module syntax
python -m uvicorn fraud_detection_service:app \
    --host 0.0.0.0 \
    --port "$API_PORT" \
    --workers "${WEB_CONCURRENCY:-1}" \
    --log-level info \
    --access-log \
    --use-colors

