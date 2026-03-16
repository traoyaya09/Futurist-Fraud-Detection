# 🤖 AI-Powered Recommendation Service

> **Production-ready recommendation engine with collaborative filtering, content-based filtering, and visual similarity search**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.118.0-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Configuration](#️-configuration)
- [Training Scripts](#-training-scripts)
- [API Endpoints](#-api-endpoints)
- [Deployment](#-deployment)
- [Monitoring](#-monitoring)
- [Troubleshooting](#-troubleshooting)
- [Performance](#-performance)
- [Contributing](#-contributing)

---

## 🎯 Overview

**Futurist E-commerce Recommendation Service** is an intelligent, ML-powered recommendation system that provides:

- **🤝 Collaborative Filtering**: Learn from user behavior patterns
- **📝 Content-Based Filtering**: Match products by text similarity
- **🖼️ Visual Similarity**: Find similar products by appearance
- **🎨 Hybrid Recommendations**: Combine multiple signals for better results
- **⚡ Real-time Processing**: Fast API responses (<100ms)
- **📊 Scalable Architecture**: Handle millions of products and users

### Use Cases

  **Personalized Product Recommendations** - "You may also like..."  
  **Visual Search** - "Find similar looking products"  
  **Content Discovery** - "Based on what you viewed..."  
  **Cart Abandonment Recovery** - Smart product suggestions  
  **Cold Start Solutions** - Recommendations for new users  

---

## ✨ Features

### 🤖 Machine Learning Models

| Model | Purpose | Technology | Output |
|-------|---------|------------|--------|
| **Text Embeddings** | Semantic product matching | SentenceTransformers | 384-dim vectors |
| **Collaborative Filtering** | User behavior patterns | SVD (Matrix Factorization) | User-product scores |
| **Image Embeddings** | Visual similarity | CLIP (OpenAI) | 512-dim vectors |

###   API Capabilities

-   **GET /recommendations/user/{user_id}** - Personalized recommendations
-   **GET /recommendations/product/{product_id}** - Similar products
-   **POST /recommendations/hybrid** - Multi-signal recommendations
-   **POST /recommendations/visual-search** - Image similarity search
-   **POST /interactions** - Track user interactions
-   **GET /health** - Service health check

### 💾 Data Processing

- **Interaction Tracking**: View, click, purchase, add-to-cart, wishlist
- **Interaction Weighting**: Prioritize high-intent actions
- **Cold Start Handling**: Fallback strategies for new users/products
- **Real-time Updates**: Track interactions instantly

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Applications                      │
│          (Web App, Mobile App, Admin Dashboard)             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ HTTP/REST
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Service                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Endpoints   │  │ Rate Limiting│  │   CORS       │      │
│  │  /recommend  │  │  Protection  │  │  Middleware  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Recommendation Engine                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Hybrid Model (Collaborative + Content + Visual)     │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Text Encoder │  │ Collaborative│  │ Image Encoder│    │
│  │  (384-dim)   │  │    Filter    │  │  (512-dim)   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                      MongoDB Database                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Products   │  │ Interactions │  │    Users     │      │
│  │  Collection  │  │  Collection  │  │  Collection  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

##   Quick Start

### Prerequisites

- **Python 3.10+** installed
- **MongoDB** instance (local or Atlas)
- **4GB+ RAM** (8GB+ recommended for training)
- **Git** for cloning

### 5-Minute Setup

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd recommendation_service

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file
cp .env.example .env
# Edit .env with your MongoDB credentials

# 5. Start the service (without models)
export LOAD_ML_MODELS=false
python recommendation_service_enhanced.py

# Service running at http://localhost:8000
# API docs at http://localhost:8000/docs
```

---

## 🔧 Installation

### Option 1: Production Installation (Full)

```bash
# Install all dependencies
pip install -r requirements.txt

# Download spaCy model (optional, for advanced NLP)
python -m spacy download en_core_web_sm

# Install CLIP model
pip install git+https://github.com/openai/CLIP.git
```

### Option 2: Minimal Installation (Service Only)

```bash
# Core service dependencies only
pip install fastapi uvicorn pydantic pydantic-settings
pip install pymongo motor numpy pandas scikit-learn
pip install sentence-transformers torch transformers
pip install requests httpx python-dotenv slowapi
```

### Option 3: Training Scripts Only

```bash
# For running training scripts locally
pip install torch torchvision sentence-transformers
pip install scikit-learn scipy joblib pandas
pip install Pillow requests tqdm pymongo python-dotenv
pip install git+https://github.com/openai/CLIP.git
```

### Option 4: Docker Installation

```bash
# Build Docker image
docker build -t recommendation-service .

# Run container
docker run -p 8000:8000 \
  -e MONGO_URI="your_mongo_uri" \
  -e LOAD_ML_MODELS=false \
  recommendation-service
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the `recommendation_service/` directory:

```bash
# ==========================================
# Database Configuration
# ==========================================
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/
MONGO_DB_NAME=futurist_ecommerce
MONGO_COLLECTION_PRODUCTS=products
MONGO_COLLECTION_INTERACTIONS=interaction_logs
MONGO_COLLECTION_USERS=users

# ==========================================
# Service Configuration
# ==========================================
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8000
ENVIRONMENT=production
DEBUG=false

# ==========================================
# ML Models Configuration
# ==========================================
LOAD_ML_MODELS=true  # Set to false for lightweight deployment
TEXT_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
IMAGE_MODEL_NAME=ViT-B/32
LATENT_FEATURES=50

# ==========================================
# Recommendation Settings
# ==========================================
DEFAULT_RECOMMENDATION_LIMIT=10
MIN_INTERACTIONS=3
SIMILARITY_THRESHOLD=0.5
MAX_RECOMMENDATIONS=50

# ==========================================
# CORS Settings
# ==========================================
CORS_ORIGINS=http://localhost:3000,http://localhost:5173,https://yourdomain.com

# ==========================================
# Rate Limiting
# ==========================================
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_ENABLED=true

# ==========================================
# Caching (Optional)
# ==========================================
REDIS_ENABLED=false
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
CACHE_TTL=3600

# ==========================================
# Training Configuration
# ==========================================
IMAGE_EMBEDDING_BATCH_SIZE=32
IMAGE_DOWNLOAD_WORKERS=4
IMAGE_DOWNLOAD_TIMEOUT=10
IMAGE_DOWNLOAD_RETRIES=3

# ==========================================
# Monitoring (Optional)
# ==========================================
PROMETHEUS_ENABLED=false
PROMETHEUS_PORT=9090
LOG_LEVEL=INFO
```

### Configuration Files

```
recommendation_service/
├── .env                          # Environment variables (create this)
├── .env.example                  # Example configuration
├── config/
│   └── settings.py              # Application settings (Pydantic)
└── requirements.txt             # Python dependencies
```

### Model Paths

Models are loaded from these directories:

```
recommendation_service/
├── text_embeddings/              # Text embeddings (.npy files)
│   ├── manifest.json
│   └── *.npy
├── models/                       # Collaborative filtering model
│   ├── hybrid_model.joblib
│   ├── svd_model.joblib
│   ├── user_item_matrix.npz
│   ├── id_mappings.json
│   └── training_report.json
└── image_embeddings/             # Image embeddings (.npy files)
    ├── manifest.json
    └── *.npy
```

---

## 🤖 Training Scripts

### Overview

Train all recommendation models using **3 training scripts**:

| Script | Purpose | Output | Duration |
|--------|---------|--------|----------|
| `generate_text_embeddings.py` | Text similarity | `text_embeddings/*.npy` | ~5-10s |
| `train_hybrid_model.py` | Collaborative filtering | `models/hybrid_model.joblib` | ~10-30s |
| `generate_image_embeddings.py` | Visual similarity | `image_embeddings/*.npy` | ~30-120s |

### 🎯 Master Training Script (Recommended)

**Run all 3 scripts with ONE command:**

```bash
# Full training pipeline with validation
python train_all_models.py --validate

# Dry run (check setup without training)
python train_all_models.py --dry-run

# Skip specific stages
python train_all_models.py --skip-images  # Skip image embeddings
python train_all_models.py --skip-text    # Skip text embeddings
```

**Expected Output:**

```
  MASTER TRAINING PIPELINE
Pre-flight checks:   PASSED
  ✓ Dependencies installed
  ✓ MongoDB connected (150 products, 5,432 interactions)
  ✓ Environment variables set

📝 Stage 1/3: Text Embeddings Generation
  → Loaded 150 products
  → Generated 150 embeddings
    SUCCESS (8.45 seconds)

🤝 Stage 2/3: Collaborative Model Training
  → Loaded 5,432 interactions
  → 87 users, 142 products
  → Trained SVD model (50 features)
    SUCCESS (15.23 seconds)

🖼️ Stage 3/3: Image Embeddings Generation
  → Downloaded 142/145 images
  → Generated 142 embeddings
    SUCCESS (45.67 seconds)

🎉 ALL STAGES COMPLETED SUCCESSFULLY!
Total duration: 69.35 seconds
```

### Individual Training Scripts

#### 1️⃣ Text Embeddings

```bash
# Generate text embeddings for all products
python generate_text_embeddings.py

# With validation
python generate_text_embeddings.py --validate

# Custom model
python generate_text_embeddings.py --model sentence-transformers/paraphrase-MiniLM-L3-v2
```

**Output:**
```
text_embeddings/
├── manifest.json
├── 507f1f77bcf86cd799439011.npy
├── 507f1f77bcf86cd799439012.npy
└── ...
```

#### 2️⃣ Collaborative Model

```bash
# Train collaborative filtering model
python train_hybrid_model.py

# Custom settings
python train_hybrid_model.py \
  --min-interactions 5 \
  --days-back 120 \
  --latent-features 100 \
  --validate
```

**Output:**
```
models/
├── hybrid_model.joblib           # Main model
├── svd_model.joblib              # SVD model
├── user_item_matrix.npz          # Sparse matrix
├── id_mappings.json              # ID mappings
└── training_report.json          # Metrics
```

#### 3️⃣ Image Embeddings

```bash
# Generate image embeddings
python generate_image_embeddings.py

# Custom settings
python generate_image_embeddings.py \
  --batch-size 64 \
  --max-workers 8 \
  --model ViT-B/16 \
  --validate
```

**Output:**
```
image_embeddings/
├── manifest.json
├── 507f1f77bcf86cd799439011.npy
├── 507f1f77bcf86cd799439012.npy
└── ...
```

### Training Requirements

**Minimum Requirements:**
- Python 3.10+
- 4GB RAM (8GB+ recommended)
- MongoDB with products and interactions
- Internet connection (for model downloads)

**Estimated Times:**
- Text embeddings: 5-10 seconds (150 products)
- Collaborative model: 10-30 seconds (5K interactions)
- Image embeddings: 30-120 seconds (150 products)
- **Total: ~1-2 minutes**

### Retraining Schedule

**Recommended retraining frequency:**

- **Text Embeddings**: When products change significantly (~weekly)
- **Collaborative Model**: As user interactions grow (~daily or weekly)
- **Image Embeddings**: When new products are added (~weekly)

**Automated retraining:**
```python
# Use APScheduler (included in requirements.txt)
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()
scheduler.add_job(retrain_models, 'cron', day_of_week='sun', hour=2)
scheduler.start()
```

---

## 🌐 API Endpoints

### Base URL

```
Local:       http://localhost:8000
Production:  https://your-service.onrender.com
```

### Interactive Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

---

### 📋 Endpoints Reference

#### 🏥 Health & Status

##### `GET /health`

Check service health and model status.

**Query Parameters:**
- `deep` (boolean, optional): Run deep health check including database

**Request:**
```bash
curl http://localhost:8000/health?deep=true
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-24T16:00:00.000000",
  "models_loaded": true,
  "models": {
    "text_embeddings": true,
    "collaborative_model": true,
    "image_embeddings": true
  },
  "database": {
    "connected": true,
    "products": 150,
    "interactions": 5432
  }
}
```

---

#### 🎯 Recommendations

##### `GET /recommendations/user/{user_id}`

Get personalized recommendations for a specific user.

**Path Parameters:**
- `user_id` (string, required): User ID

**Query Parameters:**
- `limit` (integer, optional, default=10): Number of recommendations
- `strategy` (string, optional): "collaborative" | "content" | "hybrid"

**Request:**
```bash
curl "http://localhost:8000/recommendations/user/user_123?limit=5&strategy=hybrid"
```

**Response:**
```json
{
  "user_id": "user_123",
  "recommendations": [
    {
      "product_id": "507f1f77bcf86cd799439011",
      "name": "Nike Air Max",
      "score": 0.892,
      "reason": "Based on your recent purchases"
    },
    {
      "product_id": "507f1f77bcf86cd799439012",
      "name": "Adidas Ultraboost",
      "score": 0.845,
      "reason": "Similar to items you liked"
    }
  ],
  "strategy_used": "hybrid",
  "timestamp": "2025-11-24T16:00:00.000000"
}
```

---

##### `GET /recommendations/product/{product_id}`

Get similar products (content-based recommendations).

**Path Parameters:**
- `product_id` (string, required): Product ID

**Query Parameters:**
- `limit` (integer, optional, default=10): Number of recommendations
- `method` (string, optional): "text" | "visual" | "both"

**Request:**
```bash
curl "http://localhost:8000/recommendations/product/507f1f77bcf86cd799439011?limit=5&method=both"
```

**Response:**
```json
{
  "product_id": "507f1f77bcf86cd799439011",
  "product_name": "Nike Air Max",
  "similar_products": [
    {
      "product_id": "507f1f77bcf86cd799439012",
      "name": "Nike Air Force 1",
      "similarity_score": 0.925,
      "method": "visual"
    },
    {
      "product_id": "507f1f77bcf86cd799439013",
      "name": "Nike React",
      "similarity_score": 0.887,
      "method": "text"
    }
  ],
  "timestamp": "2025-11-24T16:00:00.000000"
}
```

---

##### `POST /recommendations/hybrid`

Get hybrid recommendations combining multiple signals.

**Request Body:**
```json
{
  "user_id": "user_123",
  "context": {
    "current_product": "507f1f77bcf86cd799439011",
    "cart_items": ["507f1f77bcf86cd799439012"],
    "recent_views": ["507f1f77bcf86cd799439013"]
  },
  "weights": {
    "collaborative": 0.4,
    "content": 0.4,
    "visual": 0.2
  },
  "limit": 10
}
```

**Request:**
```bash
curl -X POST "http://localhost:8000/recommendations/hybrid" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "context": {
      "current_product": "507f1f77bcf86cd799439011"
    },
    "limit": 5
  }'
```

**Response:**
```json
{
  "user_id": "user_123",
  "recommendations": [
    {
      "product_id": "507f1f77bcf86cd799439014",
      "name": "Nike Blazer",
      "score": 0.912,
      "signals": {
        "collaborative": 0.85,
        "content": 0.92,
        "visual": 0.88
      },
      "reason": "Highly rated by similar users"
    }
  ],
  "total_score_breakdown": {
    "collaborative_weight": 0.4,
    "content_weight": 0.4,
    "visual_weight": 0.2
  },
  "timestamp": "2025-11-24T16:00:00.000000"
}
```

---

##### `POST /recommendations/visual-search`

Search for products by uploading an image.

**Request Body (multipart/form-data):**
- `image` (file, required): Image file (JPEG, PNG)
- `limit` (integer, optional): Number of results

**Request (curl):**
```bash
curl -X POST "http://localhost:8000/recommendations/visual-search" \
  -F "image=@/path/to/image.jpg" \
  -F "limit=5"
```

**Request (Python):**
```python
import requests

files = {'image': open('shoe.jpg', 'rb')}
data = {'limit': 5}
response = requests.post(
    'http://localhost:8000/recommendations/visual-search',
    files=files,
    data=data
)
```

**Response:**
```json
{
  "results": [
    {
      "product_id": "507f1f77bcf86cd799439015",
      "name": "Nike Air Jordan",
      "similarity": 0.945,
      "image_url": "https://example.com/image.jpg"
    },
    {
      "product_id": "507f1f77bcf86cd799439016",
      "name": "Nike Dunk Low",
      "similarity": 0.912,
      "image_url": "https://example.com/image2.jpg"
    }
  ],
  "search_time_ms": 45,
  "timestamp": "2025-11-24T16:00:00.000000"
}
```

---

#### 📊 Interactions

##### `POST /interactions`

Track user interaction with a product.

**Request Body:**
```json
{
  "user_id": "user_123",
  "product_id": "507f1f77bcf86cd799439011",
  "interaction_type": "view",
  "metadata": {
    "source": "product_page",
    "session_id": "sess_abc123"
  }
}
```

**Interaction Types:**
- `view` - User viewed product
- `click` - User clicked on product
- `add_to_cart` - Added to cart
- `purchase` - Completed purchase
- `wishlist` - Added to wishlist
- `like` - Liked product
- `share` - Shared product
- `review` - Left a review

**Request:**
```bash
curl -X POST "http://localhost:8000/interactions" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "product_id": "507f1f77bcf86cd799439011",
    "interaction_type": "add_to_cart"
  }'
```

**Response:**
```json
{
  "status": "success",
  "interaction_id": "int_abc123",
  "message": "Interaction recorded",
  "timestamp": "2025-11-24T16:00:00.000000"
}
```

---

#### 📈 Analytics (Optional)

##### `GET /analytics/popular`

Get most popular products.

**Query Parameters:**
- `limit` (integer, optional, default=10)
- `time_range` (string, optional): "day" | "week" | "month" | "all"

**Request:**
```bash
curl "http://localhost:8000/analytics/popular?limit=10&time_range=week"
```

**Response:**
```json
{
  "time_range": "week",
  "popular_products": [
    {
      "product_id": "507f1f77bcf86cd799439011",
      "name": "Nike Air Max",
      "view_count": 1234,
      "purchase_count": 89,
      "conversion_rate": 0.072
    }
  ]
}
```

---

### 🔐 Authentication (Optional)

If authentication is enabled:

```bash
# Get API token
curl -X POST "http://localhost:8000/auth/token" \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}'

# Use token in requests
curl "http://localhost:8000/recommendations/user/123" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

---

### 🚨 Error Responses

**400 Bad Request:**
```json
{
  "detail": "Invalid user_id format"
}
```

**404 Not Found:**
```json
{
  "detail": "Product not found"
}
```

**429 Too Many Requests:**
```json
{
  "detail": "Rate limit exceeded. Try again in 60 seconds."
}
```

**500 Internal Server Error:**
```json
{
  "detail": "Internal server error",
  "error_id": "err_abc123"
}
```

---

## 🔄 Deployment

### Option 1: Render (Recommended)

**Step 1: Prepare Files**

Create `render.yaml` in the repository root:

```yaml
services:
  - type: web
    name: recommendation-service
    env: python
    region: oregon
    plan: starter  # or 'free'
    buildCommand: |
      cd recommendation_service
      pip install -r requirements.txt
    startCommand: |
      cd recommendation_service
      python recommendation_service_enhanced.py
    envVars:
      - key: MONGO_URI
        sync: false
      - key: MONGO_DB_NAME
        value: futurist_ecommerce
      - key: LOAD_ML_MODELS
        value: false  # Set to true after uploading models
      - key: CORS_ORIGINS
        value: https://yourapp.com
      - key: ENVIRONMENT
        value: production
```

**Step 2: Deploy to Render**

```bash
# 1. Push code to GitHub
git add .
git commit -m "Deploy recommendation service"
git push origin main

# 2. Connect GitHub repo to Render
# - Go to https://dashboard.render.com
# - Click "New +" → "Web Service"
# - Connect your GitHub repository
# - Render will auto-detect render.yaml

# 3. Set environment variables in Render dashboard
# - MONGO_URI (your MongoDB connection string)
# - LOAD_ML_MODELS=false (initially)

# 4. Deploy!
```

**Step 3: Upload Trained Models (Optional)**

```bash
# After training locally, upload to Render via SSH or persistent disk
# Or use cloud storage (S3, GCS) and download on startup

# Enable models
# In Render dashboard: Set LOAD_ML_MODELS=true
# Restart service
```

---

### Option 2: Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Set environment variables
railway variables set MONGO_URI="your_mongo_uri"
railway variables set LOAD_ML_MODELS=false

# Deploy
railway up
```

---

### Option 3: Docker

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY recommendation_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY recommendation_service/ .

EXPOSE 8000

CMD ["python", "recommendation_service_enhanced.py"]
```

**Build & Run:**
```bash
# Build image
docker build -t recommendation-service .

# Run container
docker run -d -p 8000:8000 \
  -e MONGO_URI="your_mongo_uri" \
  -e LOAD_ML_MODELS=false \
  --name rec-service \
  recommendation-service

# Check logs
docker logs -f rec-service
```

**Docker Compose:**
```yaml
version: '3.8'
services:
  recommendation-service:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MONGO_URI=${MONGO_URI}
      - LOAD_ML_MODELS=false
      - ENVIRONMENT=production
    restart: unless-stopped
```

---

### Option 4: Cloud Platforms

#### AWS EC2
```bash
# Launch EC2 instance (Ubuntu 22.04)
# SSH into instance
ssh ubuntu@your-ec2-ip

# Install dependencies
sudo apt update
sudo apt install python3.10 python3-pip git -y

# Clone repo
git clone <your-repo>
cd recommendation_service

# Install packages
pip3 install -r requirements.txt

# Set environment variables
export MONGO_URI="your_mongo_uri"
export LOAD_ML_MODELS=false

# Run with screen or systemd
python3 recommendation_service_enhanced.py
```

#### Google Cloud Run
```bash
# Build container
gcloud builds submit --tag gcr.io/PROJECT_ID/recommendation-service

# Deploy
gcloud run deploy recommendation-service \
  --image gcr.io/PROJECT_ID/recommendation-service \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars MONGO_URI="your_mongo_uri",LOAD_ML_MODELS=false
```

#### Azure Container Instances
```bash
# Create resource group
az group create --name rec-service-rg --location eastus

# Deploy container
az container create \
  --resource-group rec-service-rg \
  --name recommendation-service \
  --image your-docker-image \
  --dns-name-label rec-service \
  --ports 8000 \
  --environment-variables \
    MONGO_URI="your_mongo_uri" \
    LOAD_ML_MODELS=false
```

---

### Deployment Checklist

- [ ]   MongoDB connection string configured
- [ ]   CORS origins set correctly
- [ ]   Environment variables configured
- [ ]   `LOAD_ML_MODELS=false` for initial deployment
- [ ]   Health endpoint accessible
- [ ]   Rate limiting enabled
- [ ]   Logs monitoring configured
- [ ]   SSL/HTTPS enabled (production)
- [ ]   Backup strategy for models
- [ ]   Auto-restart on failure configured

---

## 📊 Monitoring

### Health Checks

**Basic Health Check:**
```bash
curl http://localhost:8000/health
```

**Deep Health Check:**
```bash
curl http://localhost:8000/health?deep=true
```

### Logging

**Log Levels:**
```bash
# Set in .env
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

**Log Output:**
```
2025-11-24 16:00:00 [INFO] recommendation_service_enhanced: Service started
2025-11-24 16:00:01 [INFO] recommendation_service_enhanced: Models loaded successfully
2025-11-24 16:00:05 [INFO] recommendation_service_enhanced: Recommendation request for user_123
```

### Metrics (Prometheus)

**Enable Prometheus:**
```bash
# In .env
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
```

**Metrics Endpoint:**
```bash
curl http://localhost:9090/metrics
```

**Available Metrics:**
- `recommendation_requests_total` - Total recommendation requests
- `recommendation_duration_seconds` - Request duration
- `model_load_time_seconds` - Model loading time
- `database_query_duration_seconds` - DB query time

### Performance Monitoring

**Key Metrics to Track:**

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Response Time | <100ms | >500ms |
| Error Rate | <1% | >5% |
| Availability | >99.9% | <99% |
| Memory Usage | <2GB | >4GB |
| CPU Usage | <70% | >90% |

### Alerting

**Setup Alerts (Example with Render):**
```yaml
# render.yaml
services:
  - type: web
    name: recommendation-service
    alerts:
      - type: cpu
        threshold: 90
      - type: memory
        threshold: 90
      - type: restart
        threshold: 5
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Service Won't Start

**Error:** `ModuleNotFoundError: No module named 'fastapi'`

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

---

#### 2. MongoDB Connection Failed

**Error:** `ServerSelectionTimeoutError: connection refused`

**Solution:**
```bash
# Check MongoDB URI
echo $MONGO_URI

# Test connection
mongosh "your_mongo_uri"

# Update .env with correct URI
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/
```

---

#### 3. Models Not Loading

**Error:** `FileNotFoundError: [Errno 2] No such file or directory: 'models/hybrid_model.joblib'`

**Solution:**
```bash
# Option 1: Train models locally
python train_all_models.py

# Option 2: Deploy without models initially
export LOAD_ML_MODELS=false
python recommendation_service_enhanced.py

# Option 3: Check model paths
ls -la models/
ls -la text_embeddings/
ls -la image_embeddings/
```

---

#### 4. Out of Memory Error

**Error:** `Killed (OOM)`

**Solution:**
```bash
# Option 1: Disable ML models
export LOAD_ML_MODELS=false

# Option 2: Increase memory limit (Docker)
docker run --memory="4g" recommendation-service

# Option 3: Upgrade server plan
# Render: Switch to Starter or Pro plan
# Railway: Upgrade memory allocation
```

---

#### 5. Rate Limit Errors

**Error:** `429 Too Many Requests`

**Solution:**
```bash
# Increase rate limit in .env
RATE_LIMIT_PER_MINUTE=120

# Or disable rate limiting (not recommended)
RATE_LIMIT_ENABLED=false
```

---

#### 6. CORS Errors

**Error:** `Access to fetch at '...' from origin '...' has been blocked by CORS policy`

**Solution:**
```bash
# Add your frontend URL to CORS_ORIGINS in .env
CORS_ORIGINS=http://localhost:3000,https://yourapp.com

# Allow all origins (development only)
CORS_ORIGINS=*
```

---

#### 7. Slow Recommendations

**Issue:** Recommendations taking >1 second

**Solution:**
```bash
# 1. Check if models are loaded
curl http://localhost:8000/health?deep=true

# 2. Add database indexes
# In MongoDB, create indexes on:
# - products: _id, category, name
# - interactions: userId, productId, timestamp

# 3. Enable caching
REDIS_ENABLED=true
REDIS_HOST=localhost
CACHE_TTL=3600

# 4. Optimize batch size
DEFAULT_RECOMMENDATION_LIMIT=10  # Reduce from 50
```

---

#### 8. Training Scripts Fail

**Error:** `CLIP model download failed`

**Solution:**
```bash
# Install CLIP manually
pip install git+https://github.com/openai/CLIP.git

# Or use alternative
pip install clip-anytorch

# Check internet connection
ping github.com
```

---

### Debug Mode

**Enable debug logging:**
```bash
# In .env
DEBUG=true
LOG_LEVEL=DEBUG

# Run service
python recommendation_service_enhanced.py
```

**Check logs:**
```bash
# Tail logs
tail -f master_training.log
tail -f recommendation_service.log

# Search for errors
grep -i error *.log
```

---

### Performance Profiling

**Profile API endpoint:**
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run recommendation
result = get_recommendations(user_id="test")

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

---

### Getting Help

**Still stuck? Try these resources:**

1. **Check logs first:**
   ```bash
   tail -100 master_training.log
   ```

2. **Test individual components:**
   ```bash
   # Test MongoDB
   python -c "from pymongo import MongoClient; print(MongoClient('your_uri').admin.command('ping'))"
   
   # Test models
   python -c "import torch; print(torch.__version__)"
   ```

3. **Contact support:**
   - GitHub Issues: `<your-repo>/issues`
   - Email: support@yourcompany.com
   - Documentation: `<your-docs-url>`

---

## ⚡ Performance

### Benchmarks

**Hardware:** 4-core CPU, 8GB RAM

| Operation | Avg Time | 95th Percentile |
|-----------|----------|-----------------|
| User Recommendations | 45ms | 85ms |
| Similar Products | 30ms | 60ms |
| Visual Search | 120ms | 200ms |
| Hybrid Recommendations | 75ms | 150ms |
| Track Interaction | 15ms | 30ms |

### Optimization Tips

**1. Enable Caching:**
```bash
# Redis cache for frequently accessed recommendations
REDIS_ENABLED=true
CACHE_TTL=3600  # 1 hour
```

**2. Database Indexes:**
```javascript
// MongoDB
db.products.createIndex({ "_id": 1 })
db.products.createIndex({ "category": 1, "name": 1 })
db.interactions.createIndex({ "userId": 1, "timestamp": -1 })
db.interactions.createIndex({ "productId": 1 })
```

**3. Batch Processing:**
```python
# Process multiple recommendations at once
recommendations = await get_batch_recommendations(user_ids=[...])
```

**4. Model Loading:**
```bash
# Preload models on startup
LOAD_ML_MODELS=true

# Or lazy-load on first request (slower first call)
LAZY_LOAD_MODELS=true
```

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup

```bash
# 1. Fork the repository
# 2. Clone your fork
git clone https://github.com/your-username/recommendation-service.git
cd recommendation-service

# 3. Create virtual environment
python -m venv venv
source venv/bin/activate

# 4. Install dev dependencies
pip install -r requirements.txt
pip install black flake8 mypy pytest

# 5. Create feature branch
git checkout -b feature/your-feature-name
```

### Code Style

**Format code with Black:**
```bash
black recommendation_service/
```

**Lint with Flake8:**
```bash
flake8 recommendation_service/ --max-line-length=120
```

**Type check with MyPy:**
```bash
mypy recommendation_service/
```

### Testing

**Run tests:**
```bash
# All tests
pytest

# With coverage
pytest --cov=recommendation_service

# Specific test file
pytest tests/test_recommendations.py
```

### Pull Request Process

1.   Ensure all tests pass
2.   Update documentation
3.   Add tests for new features
4.   Follow code style guidelines
5.   Write clear commit messages
6.   Submit PR with description

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **FastAPI** - Modern Python web framework
- **Sentence Transformers** - State-of-the-art text embeddings
- **OpenAI CLIP** - Multimodal vision and language model
- **scikit-learn** - Machine learning library
- **MongoDB** - Flexible NoSQL database

---

## 📞 Support

- **Documentation**: `https://your-docs-url.com`
- **Issues**: `https://github.com/your-repo/issues`
- **Email**: support@yourcompany.com
- **Discord**: `https://discord.gg/your-server`

---

## 🗺️ Roadmap

###   Completed
- [x] Collaborative filtering
- [x] Content-based recommendations
- [x] Visual similarity search
- [x] Hybrid recommendations
- [x] Training scripts
- [x] API endpoints
- [x] Docker support

### 🚧 In Progress
- [ ] Real-time model updates
- [ ] A/B testing framework
- [ ] Advanced analytics dashboard

### 📋 Planned
- [ ] Multi-language support
- [ ] GraphQL API
- [ ] Mobile SDK
- [ ] Recommendation explanations (XAI)
- [ ] AutoML for hyperparameter tuning
- [ ] Federated learning support

---

## 📊 Statistics

```
📦 Total Lines of Code:     15,000+
🤖 ML Models:              3 (Text, Collaborative, Visual)
🌐 API Endpoints:          10+
📝 Documentation Pages:     25+
⭐ GitHub Stars:            Coming soon!
```

---

<div align="center">

**Built with ❤️ for E-commerce**

[⭐ Star us on GitHub](https://github.com/your-repo) • [📖 Read the Docs](https://docs.yoursite.com) • [🐛 Report Bug](https://github.com/your-repo/issues)

</div>
#   F u t u r i s t - F r a u d - D e t e c t i o n  
 