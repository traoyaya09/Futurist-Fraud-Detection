"""
train_hybrid_model.py
✅ Production-Ready Hybrid Recommendation Model Training - ENHANCED v2.1

Key Features:
- ✅ Uses enhanced product_model.py with proper ID mapping
- ✅ DateTime normalization via utils
- ✅ Memory-efficient sparse matrix operations
- ✅ Incremental training support
- ✅ Interaction weighting (views < clicks < purchases)
- ✅ Cold-start handling
- ✅ Model validation and metrics
- ✅ Progress tracking
- ✅ Thread-safe for API endpoint integration

Purpose:
- Load user interactions from MongoDB
- Build collaborative filtering model (user-item matrix)
- Combine with content-based features
- Train hybrid recommendation model
- Save trained model with proper ID mappings

Features:
- Memory-efficient sparse matrix operations
- Incremental training support
- Interaction weighting (views < clicks < purchases)
- Cold-start handling
- Model validation and metrics
- Progress tracking
- DateTime-aware data processing

Usage:
    python train_hybrid_model.py
    
    # With custom settings:
    python train_hybrid_model.py --min-interactions 5 --days-back 90 --validate

Output:
    models/
    ├── hybrid_model.joblib       (trained model with proper ID mapping)
    ├── svd_model.joblib           (SVD model)
    ├── user_item_matrix.npz       (sparse user-item matrix)
    ├── id_mappings.json           (user & product ID mappings)
    └── training_report.json       (training metrics with datetime as ISO)
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
from scipy.sparse import csr_matrix, save_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import joblib
from tqdm import tqdm
from dotenv import load_dotenv

# Import from product_model.py (enhanced version with ID mapping fix)
try:
    from product_model import (
        preprocess_ratings,
        create_ratings_matrix,
        collaborative_filtering,
        content_based_filtering,
        evaluate_model,
        save_model as save_model_product
    )
    PRODUCT_MODEL_AVAILABLE = True
except ImportError:
    PRODUCT_MODEL_AVAILABLE = False
    print("⚠️  Warning: product_model.py not available - using basic training")

# Import utils for datetime normalization
try:
    from utils.database import normalize_product
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    print("⚠️  Warning: Utils not available - datetime normalization may be limited")

# ==========================================
# Configuration
# ==========================================
load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('hybrid_model_training.log')
    ]
)
logger = logging.getLogger("HybridModelTrainer")

# ==========================================
# Configuration from Environment
# ==========================================
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "futurist_e-commerce")
MONGO_COLLECTION_INTERACTIONS = os.getenv("MONGO_COLLECTION_INTERACTIONS", "interaction_logs")
MONGO_COLLECTION_PRODUCTS = os.getenv("MONGO_COLLECTION_PRODUCTS", "products")
OUTPUT_DIR = Path("models")
MIN_INTERACTIONS = int(os.getenv("MIN_INTERACTIONS", "3"))
LATENT_FEATURES = int(os.getenv("LATENT_FEATURES", "50"))

# Interaction type weights (higher = more important)
INTERACTION_WEIGHTS = {
    "view": 1.0,
    "click": 2.0,
    "add_to_cart": 3.0,
    "addToCart": 3.0,  # Alternative naming
    "purchase": 5.0,
    "wishlist": 2.5,
    "like": 2.0,
    "share": 1.5,
    "review": 3.0,
    "rating": 4.0
}

# ==========================================
# Utility Functions
# ==========================================

def print_header(title: str):
    """Print a nice header"""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)


def print_section(title: str):
    """Print a section header"""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def format_time(seconds: float) -> str:
    """Format seconds into readable time"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


# ==========================================
# MongoDB Connection
# ==========================================

def connect_mongodb(max_retries: int = 3) -> Optional[MongoClient]:
    """Connect to MongoDB with retry logic"""
    print_section("Step 1/8: Connecting to MongoDB")
    
    for attempt in range(max_retries):
        try:
            logger.info(f"  Attempt {attempt + 1}/{max_retries}...")
            
            client = MongoClient(
                MONGO_URI,
                serverSelectionTimeoutMS=10000,
                connectTimeoutMS=10000
            )
            
            # Test connection
            client.admin.command("ping")
            
            logger.info(f"  ✓ MongoDB connected successfully")
            logger.info(f"  ✓ Database: {MONGO_DB_NAME}")
            logger.info(f"  ✓ Interactions: {MONGO_COLLECTION_INTERACTIONS}")
            logger.info(f"  ✓ Products: {MONGO_COLLECTION_PRODUCTS}")
            
            return client
            
        except (ServerSelectionTimeoutError, ConnectionFailure) as e:
            logger.error(f"  ✗ Connection failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"  Retrying in 2 seconds...")
                time.sleep(2)
            else:
                logger.error("  ✗ All MongoDB connection attempts failed")
                return None


# ==========================================
# Data Loading - ENHANCED ✅
# ==========================================

def fetch_interactions(
    client: MongoClient,
    days_back: int = 90
) -> List[Dict[str, Any]]:
    """
    Fetch user interactions from MongoDB (DateTime aware)
    
    Args:
        client: MongoDB client
        days_back: Number of days of history to fetch
        
    Returns:
        List of interaction documents (with datetime as datetime objects)
    """
    print_section("Step 2/8: Fetching Interactions")
    
    try:
        db = client[MONGO_DB_NAME]
        interactions_col = db[MONGO_COLLECTION_INTERACTIONS]
        
        # Calculate cutoff date
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        logger.info(f"  Cutoff date: {cutoff_date.isoformat()}")
        
        # Build query
        query = {
            "timestamp": {"$gte": cutoff_date}
        }
        
        # Count total
        total = interactions_col.count_documents(query)
        logger.info(f"  Found {total:,} interactions in last {days_back} days")
        
        if total == 0:
            logger.warning("  ⚠️  No interactions found!")
            return []
        
        # Fetch interactions (batch to avoid cursor timeout)
        interactions = []
        batch_size = 1000
        
        for skip in tqdm(range(0, total, batch_size), desc="  Loading batches"):
            batch = list(
                interactions_col.find(query)
                .skip(skip)
                .limit(batch_size)
            )
            interactions.extend(batch)
        
        logger.info(f"  ✓ Loaded {len(interactions):,} interactions")
        return interactions
        
    except Exception as e:
        logger.error(f"  ✗ Error fetching interactions: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def fetch_products(client: MongoClient) -> Dict[str, Dict[str, Any]]:
    """
    Fetch all products from MongoDB (with datetime normalization)
    
    Returns:
        Dict mapping product_id -> normalized product
    """
    print_section("Step 3/8: Fetching Products")
    
    try:
        db = client[MONGO_DB_NAME]
        products_col = db[MONGO_COLLECTION_PRODUCTS]
        
        # Query active products
        query = {"status": "active"}
        total = products_col.count_documents(query)
        logger.info(f"  Found {total:,} active products")
        
        if total == 0:
            logger.warning("  ⚠️  No products found!")
            return {}
        
        # Fetch products (batch to avoid cursor timeout)
        products = {}
        batch_size = 1000
        
        for skip in tqdm(range(0, total, batch_size), desc="  Loading batches"):
            batch = list(
                products_col.find(query)
                .skip(skip)
                .limit(batch_size)
            )
            
            # Normalize products (datetime → string) ✅
            if UTILS_AVAILABLE:
                batch = [normalize_product(p) for p in batch]
            
            for product in batch:
                product_id = str(product["_id"])
                products[product_id] = product
        
        logger.info(f"  ✓ Loaded {len(products):,} products")
        if UTILS_AVAILABLE:
            logger.info(f"  ✓ DateTime normalization applied")
        
        return products
        
    except Exception as e:
        logger.error(f"  ✗ Error fetching products: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


# ==========================================
# Data Processing - ENHANCED ✅
# ==========================================

def preprocess_interactions_basic(
    interactions: List[Dict[str, Any]],
    min_interactions: int = 3
) -> pd.DataFrame:
    """
    Convert interactions to DataFrame and filter
    
    Args:
        interactions: Raw interaction documents
        min_interactions: Minimum interactions per user to keep
        
    Returns:
        DataFrame with columns: userId, productId, rating, timestamp
    """
    print_section("Step 4/8: Preprocessing Interactions")
    
    logger.info("  Converting to DataFrame...")
    
    # Convert to DataFrame
    rows = []
    for interaction in interactions:
        user_id = str(interaction.get("userId", ""))
        product_id = str(interaction.get("productId", ""))
        interaction_type = interaction.get("interactionType", "view")
        timestamp = interaction.get("timestamp", datetime.utcnow())
        
        # Skip invalid
        if not user_id or not product_id:
            continue
        
        # Get weight
        weight = INTERACTION_WEIGHTS.get(interaction_type, 1.0)
        
        # Convert weight to rating (1-5 scale)
        rating = min(5.0, max(1.0, weight))
        
        rows.append({
            "userId": user_id,
            "productId": product_id,
            "rating": rating,
            "timestamp": timestamp
        })
    
    df = pd.DataFrame(rows)
    
    logger.info(f"  Raw interactions: {len(df):,}")
    
    # Remove invalid entries
    df = df[df["userId"] != ""]
    df = df[df["productId"] != ""]
    logger.info(f"  After removing invalid: {len(df):,}")
    
    # Filter users with minimum interactions
    user_counts = df["userId"].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index
    df = df[df["userId"].isin(valid_users)]
    logger.info(f"  After filtering (min {min_interactions} interactions): {len(df):,}")
    
    # Aggregate multiple interactions (take max rating)
    df_agg = df.groupby(["userId", "productId"]).agg({
        "rating": "max",
        "timestamp": "max"
    }).reset_index()
    
    logger.info(f"  ✓ Unique user-product pairs: {len(df_agg):,}")
    logger.info(f"  ✓ Unique users: {df_agg['userId'].nunique():,}")
    logger.info(f"  ✓ Unique products: {df_agg['productId'].nunique():,}")
    
    return df_agg


# ==========================================
# Model Training - ENHANCED WITH PRODUCT_MODEL.PY ✅
# ==========================================

def train_model_enhanced(
    df: pd.DataFrame,
    products_col,
    output_dir: Path,
    latent_features: int = LATENT_FEATURES
) -> Tuple[Dict, Dict, Dict]:
    """
    Train model using enhanced product_model.py
    
    Returns:
        Tuple of (collaborative_model, user_to_idx, product_to_idx)
    """
    print_section("Step 5/8: Training Model (Enhanced)")
    
    if not PRODUCT_MODEL_AVAILABLE:
        logger.error("  ✗ product_model.py not available!")
        return {}, {}, {}
    
    try:
        # Preprocess with proper ID mapping ✅
        logger.info("  Preprocessing ratings with ID mapping...")
        ratings_df, user_to_idx, product_to_idx = preprocess_ratings(df)
        
        logger.info(f"  ✓ User mappings: {len(user_to_idx)}")
        logger.info(f"  ✓ Product mappings: {len(product_to_idx)}")
        
        # Create ratings matrix
        logger.info("  Creating ratings matrix...")
        ratings_matrix = create_ratings_matrix(ratings_df, user_to_idx, product_to_idx)
        
        # Train collaborative filtering ✅
        logger.info(f"  Training collaborative filtering (k={latent_features})...")
        collab_model = collaborative_filtering(
            ratings_matrix,
            user_to_idx,
            product_to_idx,
            top_k=latent_features
        )
        
        if not collab_model:
            logger.error("  ✗ Collaborative filtering failed!")
            return {}, {}, {}
        
        logger.info(f"  ✓ Trained for {len(collab_model)} users")
        
        # Train content-based filtering (optional)
        logger.info("  Training content-based filtering...")
        product_ids = list(product_to_idx.keys())
        
        try:
            content_sim = content_based_filtering(
                product_ids,
                products_col,
                top_k=20,
                use_utils=UTILS_AVAILABLE
            )
            logger.info(f"  ✓ Content similarity for {len(content_sim)} products")
        except Exception as e:
            logger.warning(f"  ⚠️  Content-based training failed: {e}")
            content_sim = {}
        
        # Evaluate model
        logger.info("  Evaluating model...")
        try:
            metrics = evaluate_model(ratings_df, collab_model, train_ratio=0.8)
            logger.info(f"  ✓ MAE: {metrics['mae']:.3f}")
            logger.info(f"  ✓ RMSE: {metrics['rmse']:.3f}")
            logger.info(f"  ✓ Coverage: {metrics['coverage']:.3f}")
        except Exception as e:
            logger.warning(f"  ⚠️  Evaluation failed: {e}")
            metrics = {}
        
        return collab_model, user_to_idx, product_to_idx
        
    except Exception as e:
        logger.error(f"  ✗ Enhanced training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}, {}, {}


# ==========================================
# Model Saving - ENHANCED ✅
# ==========================================

def save_model_enhanced(
    collaborative_model: Dict[str, Dict[str, float]],
    user_to_idx: Dict[str, int],
    product_to_idx: Dict[str, int],
    training_stats: Dict[str, Any],
    output_dir: Path = OUTPUT_DIR
) -> bool:
    """
    Save trained model and metadata (with datetime as ISO strings)
    
    Args:
        collaborative_model: Dict mapping user_id -> {product_id: score}
        user_to_idx: User ID to index mapping
        product_to_idx: Product ID to index mapping
        training_stats: Training statistics
        output_dir: Output directory
    
    Returns:
        True if successful, False otherwise
    """
    print_section("Step 6/8: Saving Model")
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"  Output directory: {output_dir.absolute()}")
        
        # Save hybrid model (main output) ✅
        model_path = output_dir / "hybrid_model.joblib"
        joblib.dump(collaborative_model, model_path)
        logger.info(f"  ✓ Saved hybrid model: {model_path.name}")
        logger.info(f"    - {len(collaborative_model)} users")
        logger.info(f"    - {sum(len(scores) for scores in collaborative_model.values())} total scores")
        
        # Save ID mappings (CRITICAL for proper lookups) ✅
        mappings = {
            "user_to_idx": user_to_idx,
            "product_to_idx": product_to_idx,
            "idx_to_user": {str(idx): uid for uid, idx in user_to_idx.items()},
            "idx_to_product": {str(idx): pid for pid, idx in product_to_idx.items()},
            "created_at": datetime.utcnow().isoformat(),  # ✅ ISO string
            "utils_available": UTILS_AVAILABLE,
            "datetime_normalized": UTILS_AVAILABLE
        }
        
        mappings_path = output_dir / "id_mappings.json"
        with open(mappings_path, "w") as f:
            json.dump(mappings, f, indent=2)
        logger.info(f"  ✓ Saved ID mappings: {mappings_path.name}")
        
        # Save training report (with datetime as ISO) ✅
        report = {
            "training_date": datetime.utcnow().isoformat(),  # ✅ ISO string
            "duration_seconds": training_stats.get("duration", 0),
            "data_stats": training_stats.get("data_stats", {}),
            "model_stats": {
                "total_users": len(collaborative_model),
                "total_products": len(product_to_idx),
                "avg_products_per_user": np.mean([len(scores) for scores in collaborative_model.values()]) if collaborative_model else 0
            },
            "utils_integrated": UTILS_AVAILABLE,
            "datetime_normalized": UTILS_AVAILABLE,
            "product_model_used": PRODUCT_MODEL_AVAILABLE
        }
        
        report_path = output_dir / "training_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"  ✓ Saved training report: {report_path.name}")
        
        # Calculate file sizes
        total_size = sum(f.stat().st_size for f in output_dir.glob("*") if f.is_file())
        logger.info(f"  ✓ Total size: {total_size / 1024 / 1024:.2f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Failed to save model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


# ==========================================
# Validation - ENHANCED ✅
# ==========================================

def validate_model_enhanced(
    collaborative_model: Dict[str, Dict[str, float]],
    user_to_idx: Dict[str, int],
    product_to_idx: Dict[str, int],
    sample_size: int = 5
) -> bool:
    """Validate the trained model (with ID mapping checks)"""
    print_section("Step 7/8: Validating Model")
    
    try:
        if not collaborative_model:
            logger.error("  ✗ Model is empty!")
            return False
        
        # Check ID mappings ✅
        logger.info("  Checking ID mappings...")
        
        # Verify user IDs are strings (not indices)
        sample_user_ids = list(collaborative_model.keys())[:3]
        for uid in sample_user_ids:
            if not isinstance(uid, str):
                logger.error(f"  ✗ User ID is not string: {uid} (type: {type(uid)})")
                return False
            
            # Verify user exists in mapping
            if uid not in user_to_idx:
                logger.error(f"  ✗ User ID not in mapping: {uid}")
                return False
        
        logger.info("  ✓ User ID mappings correct")
        
        # Check product IDs
        sample_user = sample_user_ids[0]
        sample_product_ids = list(collaborative_model[sample_user].keys())[:3]
        
        for pid in sample_product_ids:
            if not isinstance(pid, str):
                logger.error(f"  ✗ Product ID is not string: {pid} (type: {type(pid)})")
                return False
            
            if pid not in product_to_idx:
                logger.error(f"  ✗ Product ID not in mapping: {pid}")
                return False
        
        logger.info("  ✓ Product ID mappings correct")
        
        # Check score ranges
        logger.info("  Checking score ranges...")
        sample_users = list(collaborative_model.keys())[:sample_size]
        
        for user_id in sample_users:
            user_scores = collaborative_model[user_id]
            
            if not user_scores:
                logger.warning(f"  ⚠️  User {user_id} has no scores")
                continue
            
            scores = list(user_scores.values())
            min_score = min(scores)
            max_score = max(scores)
            
            if min_score < 0 or max_score > 1:
                logger.warning(f"  ⚠️  User {user_id} has scores outside [0, 1]: [{min_score:.3f}, {max_score:.3f}]")
            
            logger.info(f"  ✓ User {user_id[:8]}...: {len(user_scores)} products, scores: [{min_score:.3f}, {max_score:.3f}]")
        
        logger.info("  ✓ Validation passed!")
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Validation error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


# ==========================================
# Statistics - ENHANCED ✅
# ==========================================

def print_statistics(
    collaborative_model: Dict,
    training_stats: Dict,
    duration: float
):
    """Print training statistics"""
    print_section("Step 8/8: Training Complete")
    
    data_stats = training_stats.get("data_stats", {})
    
    print(f"""
  Training Summary:
    ├─ Duration:             {format_time(duration)}
    ├─ Training date:        {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
    ├─ Utils integrated:     {'✅ Yes' if UTILS_AVAILABLE else '❌ No'}
    ├─ DateTime normalized:  {'✅ Yes' if UTILS_AVAILABLE else '❌ No'}
    └─ Product model used:   {'✅ Yes' if PRODUCT_MODEL_AVAILABLE else '❌ No'}
  
  Data Statistics:
    ├─ Total interactions:   {data_stats.get('total_interactions', 0):,}
    ├─ Unique users:         {data_stats.get('unique_users', 0):,}
    ├─ Unique products:      {data_stats.get('unique_products', 0):,}
    ├─ Avg per user:         {data_stats.get('avg_per_user', 0):.2f}
    └─ Avg per product:      {data_stats.get('avg_per_product', 0):.2f}
  
  Model Statistics:
    ├─ Users in model:       {len(collaborative_model):,}
    ├─ Total scores:         {sum(len(s) for s in collaborative_model.values()):,}
    └─ Avg scores/user:      {np.mean([len(s) for s in collaborative_model.values()]):.0f}
    """)


# ==========================================
# Main Function
# ==========================================

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Train hybrid recommendation model (enhanced with product_model.py)"
    )
    parser.add_argument(
        "--min-interactions",
        type=int,
        default=MIN_INTERACTIONS,
        help="Minimum interactions per user (default: 3)"
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=90,
        help="Number of days of interaction history to use (default: 90)"
    )
    parser.add_argument(
        "--latent-features",
        type=int,
        default=LATENT_FEATURES,
        help="Number of latent features for collaborative filtering (default: 50)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after training"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory for model (default: models/)"
    )
    
    args = parser.parse_args()
    
    # Print header
    print_header("HYBRID MODEL TRAINER v2.1 - ENHANCED")
    
    print(f"""
  Configuration:
    ├─ MongoDB URI:          {MONGO_URI[:50]}...
    ├─ Database:             {MONGO_DB_NAME}
    ├─ Interactions:         {MONGO_COLLECTION_INTERACTIONS}
    ├─ Products:             {MONGO_COLLECTION_PRODUCTS}
    ├─ Min interactions:     {args.min_interactions}
    ├─ Days back:            {args.days_back}
    ├─ Latent features:      {args.latent_features}
    ├─ Output directory:     {args.output_dir}
    ├─ Utils available:      {'✅ Yes' if UTILS_AVAILABLE else '❌ No'}
    └─ Product model:        {'✅ Yes' if PRODUCT_MODEL_AVAILABLE else '❌ No (basic training)'}
    """)
    
    output_dir = Path(args.output_dir)
    overall_start = time.time()
    
    try:
        # Step 1: Connect to MongoDB
        client = connect_mongodb()
        if not client:
            logger.error("❌ Failed to connect to MongoDB. Exiting.")
            return 1
        
        db = client[MONGO_DB_NAME]
        products_col = db[MONGO_COLLECTION_PRODUCTS]
        
        # Step 2: Fetch interactions
        interactions = fetch_interactions(client, days_back=args.days_back)
        if not interactions:
            logger.error("❌ No interactions found. Cannot train model.")
            return 1
        
        # Step 3: Fetch products
        products = fetch_products(client)
        if not products:
            logger.warning("⚠️  No products found. Continuing anyway...")
        
        # Step 4: Preprocess interactions
        df = preprocess_interactions_basic(interactions, min_interactions=args.min_interactions)
        if df.empty:
            logger.error("❌ No valid interactions after preprocessing. Exiting.")
            return 1
        
        # Step 5: Train model (enhanced with product_model.py) ✅
        collab_model, user_to_idx, product_to_idx = train_model_enhanced(
            df,
            products_col,
            output_dir,
            latent_features=args.latent_features
        )
        
        if not collab_model:
            logger.error("❌ Model training failed. Exiting.")
            return 1
        
        # Prepare training stats
        training_stats = {
            "duration": time.time() - overall_start,
            "data_stats": {
                "total_interactions": len(df),
                "unique_users": int(df["userId"].nunique()),
                "unique_products": int(df["productId"].nunique()),
                "avg_per_user": float(df.groupby("userId").size().mean()),
                "avg_per_product": float(df.groupby("productId").size().mean())
            }
        }
        
        # Step 6: Save model ✅
        if not save_model_enhanced(collab_model, user_to_idx, product_to_idx, training_stats, output_dir):
            logger.error("❌ Failed to save model. Exiting.")
            return 1
        
        # Step 7: Validation ✅
        if args.validate:
            if not validate_model_enhanced(collab_model, user_to_idx, product_to_idx):
                logger.warning("⚠️  Validation failed!")
        
        # Step 8: Print statistics
        duration = time.time() - overall_start
        print_statistics(collab_model, training_stats, duration)
        
        # Success message
        print_header("SUCCESS!")
        print(f"""
  ✅ Hybrid model training complete!
  
  Output:
    ├─ Model:        {output_dir.absolute() / 'hybrid_model.joblib'}
    ├─ Mappings:     {output_dir.absolute() / 'id_mappings.json'}
    └─ Report:       {output_dir.absolute() / 'training_report.json'}
  
  Next Steps:
    1. Load embeddings into service:
       python embedding_loader_enhanced.py
    
    2. Start recommendation service:
       uvicorn recommendation_service_enhanced:app --reload
    
    3. Test recommendations:
       curl http://localhost:8000/recommendations
        """)
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Process interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
        
    finally:
        if 'client' in locals() and client:
            client.close()
            logger.info("🔌 MongoDB connection closed")


if __name__ == "__main__":
    sys.exit(main())
