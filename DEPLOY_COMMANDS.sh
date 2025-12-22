#!/bin/bash
# ==========================================
# Quick Deploy Script for Fixed Dockerfile
# ==========================================

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║            Deploying Fixed Recommendation Service           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if we're in the right directory
if [ ! -f "recommendation_service/Dockerfile" ]; then
    echo "  Error: Please run this from the project root directory"
    echo "   Current directory: $(pwd)"
    echo "   Expected: Should see recommendation_service/ folder"
    exit 1
fi

echo "  Verified: In project root directory"
echo ""

# Show what changed
echo "📝 Changes to be deployed:"
echo "   - recommendation_service/Dockerfile (FIXED)"
echo ""

# Check git status
echo "🔍 Checking git status..."
git status --short recommendation_service/Dockerfile

if [ $? -ne 0 ]; then
    echo "  Error: Git repository not found"
    exit 1
fi

echo ""
echo "📦 Staging changes..."
git add recommendation_service/Dockerfile

echo ""
echo "💬 Creating commit..."
git commit -m "Fix: Dockerfile module import error - copy only existing files

- Removed references to non-existent train_hybrid_enhanced.py and embedding_loader_enhanced.py
- Added missing product_model.py to COPY list
- Verified all files exist in recommendation_service/ directory
- This resolves the 'Could not import module recommendation_service_enhanced' error"

if [ $? -ne 0 ]; then
    echo ""
    echo "   Commit failed. Possible reasons:"
    echo "    - No changes to commit (already committed?)"
    echo "    - Git user not configured"
    echo ""
    echo "If already committed, skip to push:"
    read -p "Push to GitHub anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "  Pushing to GitHub..."
git push origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                     PUSH SUCCESSFUL!                        ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "🎯 Next Steps:"
    echo ""
    echo "1. Render will automatically detect the new commit"
    echo "2. Build will start in ~30 seconds"
    echo "3. Expected build time: 3-4 minutes"
    echo ""
    echo "📊 Monitor deployment:"
    echo "   https://dashboard.render.com"
    echo ""
    echo "🧪 Test health endpoint after deployment:"
    echo "   curl https://futurist-recommendation-service.onrender.com/health"
    echo ""
    echo "  Expected result:"
    echo '   {"status":"healthy","timestamp":"...","version":"1.0.0"}'
    echo ""
    echo "🎉 Deployment initiated successfully!"
else
    echo ""
    echo "  Push failed. Please check:"
    echo "   - GitHub credentials configured"
    echo "   - Correct branch name (main vs master)"
    echo "   - Internet connection"
    echo ""
    echo "Manual push command:"
    echo "   git push origin main"
fi
