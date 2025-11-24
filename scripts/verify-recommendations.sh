#!/bin/bash

# verify-recommendations.sh
# Verify that the recommendation system is working correctly

set -e

echo "========================================="
echo "🔍 Recommendation System Verification"
echo "========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo "ℹ️  $1"
}

SERVICE_URL="http://localhost:8000"
BACKEND_URL="http://localhost:3000"

# Test counter
PASSED=0
FAILED=0

# Function to test endpoint
test_endpoint() {
    local name=$1
    local url=$2
    local method=${3:-GET}
    local data=${4:-""}
    
    echo ""
    print_info "Testing: $name"
    
    if [ "$method" = "POST" ]; then
        response=$(curl -s -w "\n%{http_code}" -X POST "$url" \
            -H "Content-Type: application/json" \
            -d "$data" 2>&1)
    else
        response=$(curl -s -w "\n%{http_code}" "$url" 2>&1)
    fi
    
    http_code=$(echo "$response" | tail -n 1)
    body=$(echo "$response" | head -n -1)
    
    if [ "$http_code" -ge 200 ] && [ "$http_code" -lt 300 ]; then
        print_success "$name - Status: $http_code"
        ((PASSED++))
        return 0
    else
        print_error "$name - Status: $http_code"
        echo "Response: $body"
        ((FAILED++))
        return 1
    fi
}

# Check if services are running
print_info "Checking if services are running..."
echo ""

# Check recommendation service
if curl -s "$SERVICE_URL/health" > /dev/null 2>&1; then
    print_success "Recommendation service is running"
else
    print_error "Recommendation service is NOT running on $SERVICE_URL"
    print_info "Start it with: cd recommendation_service && python recommendation_service_enhanced.py"
    exit 1
fi

# Check backend (optional)
if curl -s "$BACKEND_URL/api/recommendations/health" > /dev/null 2>&1; then
    print_success "Backend is running and integrated"
    BACKEND_AVAILABLE=true
else
    print_warning "Backend is NOT running on $BACKEND_URL (optional)"
    BACKEND_AVAILABLE=false
fi

echo ""
echo "========================================="
echo "Running Tests"
echo "========================================="

# Test 1: Basic health check
test_endpoint "Health Check (Basic)" "$SERVICE_URL/health"

# Test 2: Deep health check
test_endpoint "Health Check (Deep)" "$SERVICE_URL/health?deep=true"

# Test 3: Get recommendations (text query)
test_endpoint "Recommendations (Text)" \
    "$SERVICE_URL/recommendations" \
    "POST" \
    '{"query":"shoes","limit":5}'

# Test 4: Get recommendations (user-based)
test_endpoint "Recommendations (User)" \
    "$SERVICE_URL/recommendations" \
    "POST" \
    '{"userId":"test_user","limit":5}'

# Test 5: Log interaction
test_endpoint "Log Interaction" \
    "$SERVICE_URL/interactions" \
    "POST" \
    '{"userId":"test_user","productId":"test_product","action":"view"}'

# Test 6: Training history
test_endpoint "Training History" \
    "$SERVICE_URL/train/history"

# Test 7: Training jobs list
test_endpoint "Training Jobs" \
    "$SERVICE_URL/train/jobs"

# Test backend integration if available
if [ "$BACKEND_AVAILABLE" = true ]; then
    echo ""
    echo "========================================="
    echo "Testing Backend Integration"
    echo "========================================="
    
    test_endpoint "Backend Health Check" \
        "$BACKEND_URL/api/recommendations/health"
    
    test_endpoint "Backend Recommendations" \
        "$BACKEND_URL/api/recommendations" \
        "POST" \
        '{"query":"shoes","limit":5}'
fi

# Summary
echo ""
echo "========================================="
echo "Test Summary"
echo "========================================="
echo ""
echo "Total Tests: $((PASSED + FAILED))"
print_success "Passed: $PASSED"
if [ $FAILED -gt 0 ]; then
    print_error "Failed: $FAILED"
else
    print_success "Failed: 0"
fi
echo ""

# Check training status
echo "========================================="
echo "System Status"
echo "========================================="
echo ""

# Check if models are loaded
print_info "Checking model status..."
health_response=$(curl -s "$SERVICE_URL/health?deep=true")

if echo "$health_response" | grep -q '"status":"healthy"'; then
    print_success "System is healthy"
else
    print_warning "System may have issues"
fi

# Check training history
print_info "Checking training history..."
history_response=$(curl -s "$SERVICE_URL/train/history")

if echo "$history_response" | grep -q '"runs"'; then
    last_train=$(echo "$history_response" | grep -o '"last_full_train":"[^"]*"' | cut -d'"' -f4)
    
    if [ -n "$last_train" ] && [ "$last_train" != "null" ]; then
        print_success "Last training: $last_train"
    else
        print_warning "No training runs found"
        print_info "Run training with: cd recommendation_service && python train_hybrid_enhanced.py --force-full"
    fi
fi

echo ""

# Final verdict
if [ $FAILED -eq 0 ]; then
    echo "========================================="
    print_success "All Tests Passed! 🎉"
    echo "========================================="
    echo ""
    echo "Your recommendation system is working correctly!"
    echo ""
    echo "Next steps:"
    echo "1. Integrate with your frontend"
    echo "2. Monitor performance in production"
    echo "3. Schedule regular training updates"
    echo ""
    exit 0
else
    echo "========================================="
    print_error "Some Tests Failed"
    echo "========================================="
    echo ""
    echo "Please check the errors above and:"
    echo "1. Verify services are running"
    echo "2. Check logs for errors"
    echo "3. Run setup validation: cd recommendation_service && python test_training_setup.py"
    echo ""
    exit 1
fi
