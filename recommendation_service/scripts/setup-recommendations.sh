#!/bin/bash

# setup-recommendations.sh
# Quick setup script for the recommendation system

set -e  # Exit on error

echo "========================================="
echo "  Recommendation System Setup"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}  $1${NC}"
}

print_error() {
    echo -e "${RED}  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}   $1${NC}"
}

print_info() {
    echo -e "ℹ️  $1"
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi

print_success "Python 3 found: $(python3 --version)"

# Check if MongoDB is running
if ! command -v mongod &> /dev/null; then
    print_warning "MongoDB command not found. Make sure MongoDB is installed and running."
else
    print_success "MongoDB found"
fi

# Navigate to recommendation service directory
cd recommendation_service

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_info "Creating .env file from env.example..."
    if [ -f "env.example" ]; then
        cp env.example .env
        print_success ".env file created"
    else
        print_error "env.example not found"
        exit 1
    fi
else
    print_success ".env file already exists"
fi

# Install Python dependencies
print_info "Installing Python dependencies..."
if pip3 install -r requirements.txt > /dev/null 2>&1; then
    print_success "Python dependencies installed"
else
    print_error "Failed to install Python dependencies"
    exit 1
fi

# Download spaCy model
print_info "Downloading spaCy English model..."
if python3 -m spacy download en_core_web_sm > /dev/null 2>&1; then
    print_success "spaCy model downloaded"
else
    print_warning "Failed to download spaCy model (may already exist)"
fi

# Create necessary directories
print_info "Creating model directories..."
mkdir -p models text_embeddings image_embeddings
print_success "Directories created"

# Run setup validation
print_info "Running setup validation..."
if python3 test_training_setup.py; then
    print_success "Setup validation passed"
else
    print_error "Setup validation failed"
    print_info "Please check the errors above and fix them before proceeding"
    exit 1
fi

# Ask if user wants to run training
echo ""
read -p "Do you want to run initial training? (y/N) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Starting initial training (this may take a few minutes)..."
    if python3 train_hybrid_enhanced.py --force-full; then
        print_success "Training completed successfully"
    else
        print_error "Training failed"
        print_info "Check training.log for details"
        exit 1
    fi
else
    print_warning "Skipping training. You can run it later with:"
    print_info "  cd recommendation_service"
    print_info "  python3 train_hybrid_enhanced.py --force-full"
fi

# Back to root directory
cd ..

# Update backend .env if it exists
if [ -f "backend/.env" ]; then
    if ! grep -q "RECOMMENDATION_SERVICE_URL" backend/.env; then
        print_info "Adding recommendation service config to backend/.env..."
        cat >> backend/.env << EOF

# Recommendation Service
RECOMMENDATION_SERVICE_URL=http://localhost:8000
RECOMMENDATION_TIMEOUT=10000
RECOMMENDATION_CACHE_TTL=300
EOF
        print_success "Backend .env updated"
    else
        print_success "Backend .env already configured"
    fi
else
    print_warning "Backend .env not found"
fi

echo ""
echo "========================================="
echo "  Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Start the recommendation service:"
echo "   cd recommendation_service"
echo "   python3 recommendation_service_enhanced.py"
echo ""
echo "2. In another terminal, start the backend:"
echo "   cd backend"
echo "   npm start"
echo ""
echo "3. Test the service:"
echo "   curl http://localhost:8000/health?deep=true"
echo ""
echo "4. View documentation:"
echo "   cat PHASE_2_COMPLETE.md"
echo ""
