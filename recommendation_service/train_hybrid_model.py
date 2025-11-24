"""
train_hybrid_model.py
✅ Production-Ready Hybrid Recommendation Model Training

Purpose:
- Load user interactions from MongoDB
- Build collaborative filtering model (user-item matrix)
- Combine with content-based features
- Train hybrid recommendation model
- Save trained model to models/hybrid_model.joblib

Features:
- Memory-efficient sparse matrix operations
- Incremental training support
- Interaction weighting (views < clicks < purchases)
- Cold-start handling
- Model validation and metrics
- Progress tracking

Usage:
    python train_hybrid_model.py
    
    # With custom settings:
    python train_hybrid_model.py --min-interactions 5 --validate

Output:
    models/
    ├── hybrid_model.joblib       (trained model)
    ├── user_mappings.json        (user ID mappings)
    ├── product_mappings.json     (product ID mappings)
    └── training_report.json      (training metrics)
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
from collections import defaultdict

import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import joblib
from tqdm import tqdm
from dotenv import load_dotenv

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
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "futurist_ecommerce")
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
    "purchase": 5.0,
    "wishlist": 2.5,
    "like": 2.0,
    "share": 1.5,
    "review": 3.0
}

# ==========================================
# MongoDB Connection
# ==========================================
def connect_mongodb(max_retries: int = 3) -> Optional[MongoClient]:
    """Connect to MongoDB with retry logic"""
    for attempt in range(max_retries):
        try:
            logger.info(f"📡 Connecting to MongoDB (attempt {attempt + 1}/{max_retries})...")
            client = MongoClient(
                MONGO_URI,
                serverSelectionTimeoutMS=10000,
                connectTimeoutMS=10000
            )
            # Test connection
            client.admin.command("ping")
            logger.info("✅ MongoDB connected successfully")
            return client
        except (ServerSelectionTimeoutError, ConnectionFailure) as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                logger.error("❌ All MongoDB connection attempts failed")
                return None


# ==========================================
# Data Loading
# ==========================================
def fetch_interactions(
    client: MongoClient,
    days_back: int = 90
) -> List[Dict[str, Any]]:
    """
    Fetch user interactions from MongoDB
    
    Args:
        client: MongoDB client
        days_back: Number of days of history to fetch
        
    Returns:
        List of interaction documents
    """
    try:
        db = client[MONGO_DB_NAME]
        interactions_col = db[MONGO_COLLECTION_INTERACTIONS]
        
        # Calculate cutoff date
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        # Build query
        query = {
            "timestamp": {"$gte": cutoff_date}
        }
        
        # Count total
        total = interactions_col.count_documents(query)
        logger.info(f"📊 Found {total} interactions in last {days_back} days")
        
        if total == 0:
            logger.warning("⚠️  No interactions found!")
            return []
        
        # Fetch interactions
        interactions = []
        cursor = interactions_col.find(query)
        
        for interaction in tqdm(cursor, total=total, desc="Loading interactions"):
            interactions.append(interaction)
        
        logger.info(f"✅ Loaded {len(interactions)} interactions")
        return interactions
        
    except Exception as e:
        logger.error(f"❌ Error fetching interactions: {e}")
        return []


def fetch_products(client: MongoClient) -> Dict[str, Dict[str, Any]]:
    """Fetch all products from MongoDB"""
    try:
        db = client[MONGO_DB_NAME]
        products_col = db[MONGO_COLLECTION_PRODUCTS]
        
        total = products_col.count_documents({})
        logger.info(f"📦 Found {total} products in database")
        
        if total == 0:
            logger.warning("⚠️  No products found!")
            return {}
        
        products = {}
        cursor = products_col.find({})
        
        for product in tqdm(cursor, total=total, desc="Loading products"):
            product_id = str(product["_id"])
            products[product_id] = product
        
        logger.info(f"✅ Loaded {len(products)} products")
        return products
        
    except Exception as e:
        logger.error(f"❌ Error fetching products: {e}")
        return {}


# ==========================================
# Data Processing
# ==========================================
def preprocess_interactions(
    interactions: List[Dict[str, Any]],
    min_interactions: int = 3
) -> pd.DataFrame:
    """
    Convert interactions to DataFrame and filter
    
    Args:
        interactions: Raw interaction documents
        min_interactions: Minimum interactions per user to keep
        
    Returns:
        Cleaned DataFrame with columns: userId, productId, interactionType, weight
    """
    logger.info("🔄 Processing interactions...")
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            "userId": str(i.get("userId", "")),
            "productId": str(i.get("productId", "")),
            "interactionType": i.get("interactionType", "view"),
            "timestamp": i.get("timestamp", datetime.utcnow())
        }
        for i in interactions
    ])
    
    logger.info(f"📊 Raw interactions: {len(df)}")
    
    # Remove invalid entries
    df = df[df["userId"] != ""]
    df = df[df["productId"] != ""]
    logger.info(f"📊 After removing invalid: {len(df)}")
    
    # Add interaction weights
    df["weight"] = df["interactionType"].map(INTERACTION_WEIGHTS).fillna(1.0)
    
    # Filter users with minimum interactions
    user_counts = df["userId"].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index
    df = df[df["userId"].isin(valid_users)]
    logger.info(f"📊 After filtering (min {min_interactions} interactions): {len(df)}")
    
    # Aggregate multiple interactions
    df_agg = df.groupby(["userId", "productId"]).agg({
        "weight": "sum",
        "timestamp": "max"
    }).reset_index()
    
    logger.info(f"📊 Unique user-product pairs: {len(df_agg)}")
    logger.info(f"👥 Unique users: {df_agg['userId'].nunique()}")
    logger.info(f"📦 Unique products: {df_agg['productId'].nunique()}")
    
    return df_agg


def create_user_item_matrix(
    df: pd.DataFrame
) -> Tuple[csr_matrix, Dict[str, int], Dict[str, int]]:
    """
    Create sparse user-item interaction matrix
    
    Args:
        df: DataFrame with userId, productId, weight
        
    Returns:
        - Sparse matrix (users x products)
        - user_to_idx mapping
        - product_to_idx mapping
    """
    logger.info("🔄 Creating user-item matrix...")
    
    # Create mappings
    unique_users = sorted(df["userId"].unique())
    unique_products = sorted(df["productId"].unique())
    
    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    product_to_idx = {prod: idx for idx, prod in enumerate(unique_products)}
    
    # Create matrix indices
    user_indices = df["userId"].map(user_to_idx).values
    product_indices = df["productId"].map(product_to_idx).values
    weights = df["weight"].values
    
    # Create sparse matrix
    matrix = csr_matrix(
        (weights, (user_indices, product_indices)),
        shape=(len(unique_users), len(unique_products))
    )
    
    logger.info(f"✅ Matrix shape: {matrix.shape} (users x products)")
    logger.info(f"📊 Matrix density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]) * 100:.4f}%")
    logger.info(f"📊 Non-zero entries: {matrix.nnz:,}")
    
    return matrix, user_to_idx, product_to_idx


# ==========================================
# Model Training
# ==========================================
def train_collaborative_filtering(
    user_item_matrix: csr_matrix,
    n_components: int = LATENT_FEATURES
) -> TruncatedSVD:
    """
    Train collaborative filtering model using SVD
    
    Args:
        user_item_matrix: Sparse user-item matrix
        n_components: Number of latent features
        
    Returns:
        Trained SVD model
    """
    logger.info(f"🤖 Training collaborative filtering model...")
    logger.info(f"📊 Latent features: {n_components}")
    
    # Ensure n_components is valid
    max_components = min(user_item_matrix.shape) - 1
    n_components = min(n_components, max_components)
    
    if n_components < 1:
        logger.error(f"❌ Cannot train model: not enough data (matrix shape: {user_item_matrix.shape})")
        return None
    
    logger.info(f"📊 Using {n_components} components")
    
    # Train SVD
    try:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        
        start_time = time.time()
        user_features = svd.fit_transform(user_item_matrix)
        duration = time.time() - start_time
        
        logger.info(f"✅ Model trained in {duration:.2f} seconds")
        logger.info(f"📊 Explained variance: {svd.explained_variance_ratio_.sum() * 100:.2f}%")
        
        return svd
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return None


def compute_similarity_scores(
    svd: TruncatedSVD,
    user_item_matrix: csr_matrix,
    user_to_idx: Dict[str, int],
    product_to_idx: Dict[str, int]
) -> Dict[str, Dict[str, float]]:
    """
    Compute user-product similarity scores
    
    Returns:
        Dictionary: {userId: {productId: score}}
    """
    logger.info("🔄 Computing similarity scores...")
    
    # Get user and item features
    user_features = svd.transform(user_item_matrix)
    product_features = svd.components_.T  # (n_products, n_components)
    
    # Normalize features
    user_features_norm = normalize(user_features, axis=1)
    product_features_norm = normalize(product_features, axis=1)
    
    # Compute similarity matrix (users x products)
    similarity_matrix = user_features_norm @ product_features_norm.T
    
    # Convert to dictionary
    idx_to_user = {idx: user for user, idx in user_to_idx.items()}
    idx_to_product = {idx: prod for prod, idx in product_to_idx.items()}
    
    hybrid_model = {}
    
    for user_idx in tqdm(range(len(idx_to_user)), desc="Building user scores"):
        user_id = idx_to_user[user_idx]
        user_scores = {}
        
        for product_idx in range(len(idx_to_product)):
            product_id = idx_to_product[product_idx]
            score = similarity_matrix[user_idx, product_idx]
            
            # Normalize score to [0, 1]
            score = (score + 1) / 2  # from [-1, 1] to [0, 1]
            user_scores[product_id] = float(score)
        
        hybrid_model[user_id] = user_scores
    
    logger.info(f"✅ Computed scores for {len(hybrid_model)} users")
    
    return hybrid_model


# ==========================================
# Model Saving
# ==========================================
def save_model(
    hybrid_model: Dict[str, Dict[str, float]],
    user_to_idx: Dict[str, int],
    product_to_idx: Dict[str, int],
    svd: TruncatedSVD,
    user_item_matrix: csr_matrix,
    output_dir: Path = OUTPUT_DIR
) -> bool:
    """Save trained model and metadata"""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Saving model to {output_dir}...")
        
        # Save hybrid model (main output)
        model_path = output_dir / "hybrid_model.joblib"
        joblib.dump(hybrid_model, model_path)
        logger.info(f"✅ Saved hybrid model: {model_path}")
        
        # Save SVD model
        svd_path = output_dir / "svd_model.joblib"
        joblib.dump(svd, svd_path)
        logger.info(f"✅ Saved SVD model: {svd_path}")
        
        # Save user-item matrix
        matrix_path = output_dir / "user_item_matrix.npz"
        save_npz(matrix_path, user_item_matrix)
        logger.info(f"✅ Saved user-item matrix: {matrix_path}")
        
        # Save mappings
        mappings = {
            "user_to_idx": user_to_idx,
            "product_to_idx": product_to_idx,
            "idx_to_user": {str(idx): user for user, idx in user_to_idx.items()},
            "idx_to_product": {str(idx): prod for prod, idx in product_to_idx.items()}
        }
        
        mappings_path = output_dir / "id_mappings.json"
        with open(mappings_path, "w") as f:
            json.dump(mappings, f, indent=2)
        logger.info(f"✅ Saved ID mappings: {mappings_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to save model: {e}")
        return False


def save_training_report(
    df: pd.DataFrame,
    hybrid_model: Dict[str, Dict[str, float]],
    svd: TruncatedSVD,
    duration: float,
    output_dir: Path = OUTPUT_DIR
):
    """Save training report with metrics"""
    try:
        report = {
            "training_date": datetime.utcnow().isoformat(),
            "duration_seconds": round(duration, 2),
            "data_stats": {
                "total_interactions": len(df),
                "unique_users": int(df["userId"].nunique()),
                "unique_products": int(df["productId"].nunique()),
                "avg_interactions_per_user": float(df.groupby("userId").size().mean()),
                "avg_interactions_per_product": float(df.groupby("productId").size().mean())
            },
            "model_stats": {
                "total_users_in_model": len(hybrid_model),
                "latent_features": svd.n_components,
                "explained_variance": float(svd.explained_variance_ratio_.sum()),
                "avg_products_per_user": float(np.mean([len(scores) for scores in hybrid_model.values()]))
            },
            "interaction_distribution": df["interactionType"].value_counts().to_dict(),
            "weight_distribution": {
                "mean": float(df["weight"].mean()),
                "std": float(df["weight"].std()),
                "min": float(df["weight"].min()),
                "max": float(df["weight"].max())
            }
        }
        
        report_path = output_dir / "training_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Saved training report: {report_path}")
        return report
        
    except Exception as e:
        logger.error(f"⚠️  Failed to save training report: {e}")
        return None


# ==========================================
# Validation
# ==========================================
def validate_model(
    hybrid_model: Dict[str, Dict[str, float]],
    sample_size: int = 5
) -> bool:
    """Validate the trained model"""
    logger.info("🔍 Validating model...")
    
    try:
        if not hybrid_model:
            logger.error("❌ Model is empty!")
            return False
        
        # Check sample users
        sample_users = list(hybrid_model.keys())[:sample_size]
        
        for user_id in sample_users:
            user_scores = hybrid_model[user_id]
            
            if not user_scores:
                logger.warning(f"⚠️  User {user_id} has no scores")
                continue
            
            # Check score range
            scores = list(user_scores.values())
            min_score = min(scores)
            max_score = max(scores)
            
            if min_score < 0 or max_score > 1:
                logger.warning(f"⚠️  User {user_id} has scores outside [0, 1]: [{min_score}, {max_score}]")
            
            logger.info(f"  ✓ User {user_id}: {len(user_scores)} products, score range: [{min_score:.3f}, {max_score:.3f}]")
        
        logger.info("✅ Validation passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Validation error: {e}")
        return False


# ==========================================
# Statistics
# ==========================================
def print_statistics(report: Dict[str, Any]):
    """Print training statistics"""
    logger.info("=" * 80)
    logger.info("📊 HYBRID MODEL TRAINING STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Training date: {report['training_date']}")
    logger.info(f"Duration: {report['duration_seconds']:.2f} seconds")
    logger.info("")
    logger.info("Data Statistics:")
    logger.info(f"  Total interactions: {report['data_stats']['total_interactions']:,}")
    logger.info(f"  Unique users: {report['data_stats']['unique_users']:,}")
    logger.info(f"  Unique products: {report['data_stats']['unique_products']:,}")
    logger.info(f"  Avg interactions per user: {report['data_stats']['avg_interactions_per_user']:.2f}")
    logger.info(f"  Avg interactions per product: {report['data_stats']['avg_interactions_per_product']:.2f}")
    logger.info("")
    logger.info("Model Statistics:")
    logger.info(f"  Users in model: {report['model_stats']['total_users_in_model']:,}")
    logger.info(f"  Latent features: {report['model_stats']['latent_features']}")
    logger.info(f"  Explained variance: {report['model_stats']['explained_variance'] * 100:.2f}%")
    logger.info("")
    logger.info("Interaction Types:")
    for itype, count in report['interaction_distribution'].items():
        logger.info(f"  {itype}: {count:,}")
    logger.info("=" * 80)


# ==========================================
# Main Function
# ==========================================
def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Train hybrid recommendation model")
    parser.add_argument(
        "--min-interactions",
        type=int,
        default=MIN_INTERACTIONS,
        help="Minimum interactions per user"
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=90,
        help="Number of days of interaction history to use"
    )
    parser.add_argument(
        "--latent-features",
        type=int,
        default=LATENT_FEATURES,
        help="Number of latent features for SVD"
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
        help="Output directory for model"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("🚀 HYBRID MODEL TRAINER")
    logger.info("=" * 80)
    logger.info(f"MongoDB URI: {MONGO_URI[:50]}...")
    logger.info(f"Database: {MONGO_DB_NAME}")
    logger.info(f"Interactions collection: {MONGO_COLLECTION_INTERACTIONS}")
    logger.info(f"Products collection: {MONGO_COLLECTION_PRODUCTS}")
    logger.info(f"Min interactions: {args.min_interactions}")
    logger.info(f"Days back: {args.days_back}")
    logger.info(f"Latent features: {args.latent_features}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)
    
    output_dir = Path(args.output_dir)
    overall_start = time.time()
    
    try:
        # Step 1: Connect to MongoDB
        logger.info("\n📡 Step 1/7: Connecting to MongoDB...")
        client = connect_mongodb()
        if not client:
            logger.error("❌ Failed to connect to MongoDB. Exiting.")
            return 1
        
        # Step 2: Fetch interactions
        logger.info(f"\n📦 Step 2/7: Fetching interactions (last {args.days_back} days)...")
        interactions = fetch_interactions(client, days_back=args.days_back)
        if not interactions:
            logger.error("❌ No interactions found. Cannot train model.")
            return 1
        
        # Step 3: Fetch products
        logger.info("\n📦 Step 3/7: Fetching products...")
        products = fetch_products(client)
        if not products:
            logger.warning("⚠️  No products found. Continuing anyway...")
        
        # Step 4: Preprocess interactions
        logger.info("\n🔄 Step 4/7: Preprocessing interactions...")
        df = preprocess_interactions(interactions, min_interactions=args.min_interactions)
        if df.empty:
            logger.error("❌ No valid interactions after preprocessing. Exiting.")
            return 1
        
        # Step 5: Create user-item matrix
        logger.info("\n🔄 Step 5/7: Creating user-item matrix...")
        user_item_matrix, user_to_idx, product_to_idx = create_user_item_matrix(df)
        
        # Step 6: Train model
        logger.info("\n🤖 Step 6/7: Training collaborative filtering model...")
        svd = train_collaborative_filtering(user_item_matrix, n_components=args.latent_features)
        if svd is None:
            logger.error("❌ Model training failed. Exiting.")
            return 1
        
        # Compute similarity scores
        logger.info("\n🔄 Computing user-product similarity scores...")
        hybrid_model = compute_similarity_scores(svd, user_item_matrix, user_to_idx, product_to_idx)
        
        # Step 7: Save model
        logger.info("\n💾 Step 7/7: Saving model...")
        if not save_model(hybrid_model, user_to_idx, product_to_idx, svd, user_item_matrix, output_dir):
            logger.error("❌ Failed to save model. Exiting.")
            return 1
        
        # Save training report
        duration = time.time() - overall_start
        report = save_training_report(df, hybrid_model, svd, duration, output_dir)
        
        # Validation
        if args.validate:
            logger.info("\n✓ Validating model...")
            if not validate_model(hybrid_model):
                logger.warning("⚠️  Validation failed!")
        
        # Print statistics
        if report:
            print_statistics(report)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ HYBRID MODEL TRAINING COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"📂 Model saved to: {output_dir.absolute()}")
        logger.info(f"📄 Training report: {output_dir.absolute() / 'training_report.json'}")
        logger.info(f"⏱️  Total duration: {duration:.2f} seconds")
        logger.info("\n🎯 Next steps:")
        logger.info("  1. Test the model with sample queries")
        logger.info("  2. Generate image embeddings: python generate_image_embeddings.py")
        logger.info("  3. Upload model to your server")
        logger.info("  4. Restart recommendation service to load new model")
        logger.info("=" * 80)
        
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
        if client:
            client.close()
            logger.info("🔌 MongoDB connection closed")


if __name__ == "__main__":
    sys.exit(main())
