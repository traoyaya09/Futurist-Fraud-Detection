#!/bin/bash

# ==========================================
# Render Deployment Helper Script
# ==========================================

set -e

echo "  Render Deployment Helper"
echo "============================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "recommendation_service_enhanced.py" ]; then
    echo -e "${RED}  Error: Must run from recommendation_service directory${NC}"
    echo "Run: cd recommendation_service && ./deploy-to-render.sh"
    exit 1
fi

echo -e "${BLUE}📦 Step 1: Testing Docker Build Locally${NC}"
echo "Building Docker image..."

if docker build -t futurist-rec-test . > /dev/null 2>&1; then
    echo -e "${GREEN}  Docker build successful!${NC}"
else
    echo -e "${RED}  Docker build failed. Check your Dockerfile.${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}🧪 Step 2: Testing Container${NC}"
echo "Starting test container..."

# Kill any existing test container
docker rm -f futurist-rec-test-container 2>/dev/null || true

# Start container in background
docker run -d --name futurist-rec-test-container -p 8001:8000 futurist-rec-test

# Wait for startup
echo "Waiting for service to start..."
sleep 5

# Test health endpoint
if curl -s http://localhost:8001/health > /dev/null 2>&1; then
    echo -e "${GREEN}  Container is running and healthy!${NC}"
    
    # Show health response
    echo ""
    echo "Health check response:"
    curl -s http://localhost:8001/health | python3 -m json.tool 2>/dev/null || echo "Service is running"
else
    echo -e "${RED}  Container health check failed${NC}"
    echo "Container logs:"
    docker logs futurist-rec-test-container
    docker rm -f futurist-rec-test-container
    exit 1
fi

# Cleanup
echo ""
echo "Cleaning up test container..."
docker rm -f futurist-rec-test-container > /dev/null 2>&1

echo ""
echo -e "${GREEN}  All local tests passed!${NC}"
echo ""

# Check if render.yaml exists
if [ ! -f "render.yaml" ]; then
    echo -e "${YELLOW}   render.yaml not found in current directory${NC}"
    echo "This is OK if you're deploying via Render dashboard"
fi

echo ""
echo -e "${BLUE}📝 Next Steps for Render Deployment:${NC}"
echo ""
echo "Option A - Deploy via Dashboard (Recommended):"
echo "  1. Go to: https://dashboard.render.com"
echo "  2. Select your service: futurist-recommendation-service"
echo "  3. Go to Settings → Build & Deploy"
echo "  4. Set Docker Dockerfile Path: ./recommendation_service/Dockerfile"
echo "  5. Set Docker Build Context Directory: ./recommendation_service"
echo "  6. Click 'Save Changes'"
echo "  7. Click 'Manual Deploy' → 'Deploy latest commit'"
echo ""
echo "Option B - Deploy via render.yaml:"
echo "  1. Commit changes: git add . && git commit -m 'Fix Render paths'"
echo "  2. Push to GitHub: git push origin main"
echo "  3. Render will auto-deploy (if enabled)"
echo ""

# Offer to commit and push
echo -e "${YELLOW}Do you want to commit and push changes now? (y/n)${NC}"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    cd ..
    
    echo "Committing changes..."
    git add recommendation_service/Dockerfile recommendation_service/render.yaml recommendation_service/RENDER_DEPLOYMENT_FIX.md
    git commit -m "fix: Update Render deployment configuration for recommendation service

- Fix Docker context path
- Update Dockerfile to properly copy application files
- Add deployment troubleshooting guide
- Ensure uvicorn can find recommendation_service_enhanced module"
    
    echo "Pushing to remote..."
    git push origin main
    
    echo -e "${GREEN}  Changes pushed to GitHub!${NC}"
    echo ""
    echo "Check Render dashboard for deployment progress:"
    echo "https://dashboard.render.com"
else
    echo ""
    echo "Skipping git push. Remember to commit and push when ready!"
fi

echo ""
echo -e "${GREEN}🎉 Deployment preparation complete!${NC}"
echo ""
echo "Monitor deployment at: https://dashboard.render.com"
