"""
generate_text_embeddings.py
✅ Production-Ready Text Embeddings Generator

Purpose:
- Load products from MongoDB
- Generate text embeddings using SentenceTransformer
- Save embeddings to text_embeddings/ folder
- Memory-efficient batch processing
- Progress tracking and error handling

Usage:
    python generate_text_embeddings.py
    
    # Or with custom settings:
    python generate_text_embeddings.py --batch-size 50 --model all-MiniLM-L6-v2

Output:
    text_embeddings/
    ├── product_id_1.npy
    ├── product_id_2.npy
    └── ...
    
    text_embeddings/manifest.json (metadata)
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import numpy as np
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
from sentence_transformers import SentenceTransformer
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
        logging.FileHandler('text_embeddings_training.log')
    ]
)
logger = logging.getLogger("TextEmbeddingsGenerator")

# ==========================================
# Configuration from Environment
# ==========================================
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "futurist_ecommerce")
MONGO_COLLECTION_PRODUCTS = os.getenv("MONGO_COLLECTION_PRODUCTS", "products")
TEXT_MODEL_NAME = os.getenv("TEXT_MODEL_NAME", "all-MiniLM-L6-v2")
OUTPUT_DIR = Path("text_embeddings")
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

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
# Product Data Loading
# ==========================================
def fetch_products(client: MongoClient, batch_size: int = 1000) -> List[Dict[str, Any]]:
    """Fetch all products from MongoDB"""
    try:
        db = client[MONGO_DB_NAME]
        products_col = db[MONGO_COLLECTION_PRODUCTS]
        
        # Count total
        total = products_col.count_documents({})
        logger.info(f"📊 Found {total} products in database")
        
        if total == 0:
            logger.warning("⚠️  No products found in database!")
            return []
        
        # Fetch products
        products = []
        cursor = products_col.find({})
        
        for product in tqdm(cursor, total=total, desc="Loading products"):
            products.append(product)
        
        logger.info(f"✅ Loaded {len(products)} products")
        return products
        
    except Exception as e:
        logger.error(f"❌ Error fetching products: {e}")
        return []


# ==========================================
# Text Processing
# ==========================================
def create_product_text(product: Dict[str, Any]) -> str:
    """
    Create a rich text representation of a product for embedding
    
    Combines: name, description, category, brand, tags
    """
    text_parts = []
    
    # Product name (most important)
    if product.get("name"):
        text_parts.append(product["name"])
    
    # Description
    if product.get("description"):
        text_parts.append(product["description"])
    
    # Category and subcategory
    if product.get("category"):
        text_parts.append(f"Category: {product['category']}")
    if product.get("subCategory"):
        text_parts.append(f"Subcategory: {product['subCategory']}")
    
    # Brand
    if product.get("brand"):
        text_parts.append(f"Brand: {product['brand']}")
    
    # Tags
    if product.get("tags") and isinstance(product["tags"], list):
        tags_str = ", ".join(product["tags"])
        text_parts.append(f"Tags: {tags_str}")
    
    # Features (if available)
    if product.get("features") and isinstance(product["features"], list):
        features_str = ", ".join(product["features"])
        text_parts.append(f"Features: {features_str}")
    
    return " | ".join(text_parts)


# ==========================================
# Embedding Generation
# ==========================================
def generate_embeddings(
    products: List[Dict[str, Any]],
    model: SentenceTransformer,
    batch_size: int = 32,
    output_dir: Path = OUTPUT_DIR
) -> Dict[str, Any]:
    """
    Generate text embeddings for all products
    
    Returns:
        manifest: Dictionary with metadata about generated embeddings
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🔄 Generating embeddings for {len(products)} products...")
    logger.info(f"📦 Batch size: {batch_size}")
    logger.info(f"💾 Output directory: {output_dir}")
    
    manifest = {
        "generated_at": datetime.utcnow().isoformat(),
        "model_name": model.get_sentence_embedding_dimension(),
        "embedding_dimension": model.get_sentence_embedding_dimension(),
        "total_products": len(products),
        "embeddings": {}
    }
    
    successful = 0
    failed = 0
    
    # Process in batches
    for i in tqdm(range(0, len(products), batch_size), desc="Generating embeddings"):
        batch = products[i:i + batch_size]
        
        try:
            # Create text representations
            texts = [create_product_text(p) for p in batch]
            
            # Generate embeddings
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Save individual embeddings
            for product, embedding in zip(batch, embeddings):
                try:
                    product_id = str(product["_id"])
                    
                    # Save embedding as .npy file
                    embedding_path = output_dir / f"{product_id}.npy"
                    np.save(embedding_path, embedding)
                    
                    # Add to manifest
                    manifest["embeddings"][product_id] = {
                        "name": product.get("name", "Unknown"),
                        "category": product.get("category", "Unknown"),
                        "file": str(embedding_path.name),
                        "shape": list(embedding.shape),
                        "text_length": len(create_product_text(product))
                    }
                    
                    successful += 1
                    
                except Exception as e:
                    logger.warning(f"⚠️  Failed to save embedding for product {product.get('_id')}: {e}")
                    failed += 1
        
        except Exception as e:
            logger.error(f"❌ Batch processing error: {e}")
            failed += len(batch)
    
    # Update manifest
    manifest["successful"] = successful
    manifest["failed"] = failed
    manifest["success_rate"] = successful / len(products) if len(products) > 0 else 0
    
    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"✅ Embeddings generated: {successful} successful, {failed} failed")
    logger.info(f"📄 Manifest saved to: {manifest_path}")
    
    return manifest


# ==========================================
# Validation
# ==========================================
def validate_embeddings(output_dir: Path = OUTPUT_DIR) -> bool:
    """Validate that embeddings were generated correctly"""
    try:
        manifest_path = output_dir / "manifest.json"
        
        if not manifest_path.exists():
            logger.error("❌ Manifest file not found")
            return False
        
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        total_expected = manifest.get("successful", 0)
        embedding_files = list(output_dir.glob("*.npy"))
        total_found = len(embedding_files)
        
        logger.info(f"📊 Validation: Expected {total_expected}, Found {total_found} embedding files")
        
        if total_found != total_expected:
            logger.warning(f"⚠️  Mismatch: Expected {total_expected} but found {total_found}")
            return False
        
        # Sample check: Load a few embeddings
        sample_size = min(5, total_found)
        logger.info(f"🔍 Checking {sample_size} sample embeddings...")
        
        for i, emb_file in enumerate(embedding_files[:sample_size]):
            try:
                embedding = np.load(emb_file)
                logger.info(f"  ✓ {emb_file.name}: shape={embedding.shape}, dtype={embedding.dtype}")
            except Exception as e:
                logger.error(f"  ✗ {emb_file.name}: Failed to load - {e}")
                return False
        
        logger.info("✅ Validation passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Validation error: {e}")
        return False


# ==========================================
# Statistics
# ==========================================
def print_statistics(manifest: Dict[str, Any]):
    """Print statistics about generated embeddings"""
    logger.info("=" * 80)
    logger.info("📊 EMBEDDING GENERATION STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Generated at: {manifest.get('generated_at', 'Unknown')}")
    logger.info(f"Model: {TEXT_MODEL_NAME}")
    logger.info(f"Embedding dimension: {manifest.get('embedding_dimension', 'Unknown')}")
    logger.info(f"Total products: {manifest.get('total_products', 0)}")
    logger.info(f"Successful: {manifest.get('successful', 0)}")
    logger.info(f"Failed: {manifest.get('failed', 0)}")
    logger.info(f"Success rate: {manifest.get('success_rate', 0) * 100:.2f}%")
    logger.info("=" * 80)


# ==========================================
# Main Function
# ==========================================
def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Generate text embeddings for products")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for embedding generation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=TEXT_MODEL_NAME,
        help="SentenceTransformer model name"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory for embeddings"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after generation"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("🚀 TEXT EMBEDDINGS GENERATOR")
    logger.info("=" * 80)
    logger.info(f"MongoDB URI: {MONGO_URI[:50]}...")
    logger.info(f"Database: {MONGO_DB_NAME}")
    logger.info(f"Collection: {MONGO_COLLECTION_PRODUCTS}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)
    
    output_dir = Path(args.output_dir)
    
    try:
        # Step 1: Connect to MongoDB
        logger.info("\n📡 Step 1/5: Connecting to MongoDB...")
        client = connect_mongodb()
        if not client:
            logger.error("❌ Failed to connect to MongoDB. Exiting.")
            return 1
        
        # Step 2: Fetch products
        logger.info("\n📦 Step 2/5: Fetching products...")
        products = fetch_products(client, batch_size=1000)
        if not products:
            logger.error("❌ No products found. Exiting.")
            return 1
        
        # Step 3: Load model
        logger.info(f"\n🤖 Step 3/5: Loading SentenceTransformer model '{args.model}'...")
        try:
            model = SentenceTransformer(args.model)
            embedding_dim = model.get_sentence_embedding_dimension()
            logger.info(f"✅ Model loaded: dimension={embedding_dim}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return 1
        
        # Step 4: Generate embeddings
        logger.info("\n🔄 Step 4/5: Generating embeddings...")
        start_time = time.time()
        manifest = generate_embeddings(
            products=products,
            model=model,
            batch_size=args.batch_size,
            output_dir=output_dir
        )
        duration = time.time() - start_time
        logger.info(f"⏱️  Generation completed in {duration:.2f} seconds")
        
        # Step 5: Validation
        if args.validate:
            logger.info("\n✓ Step 5/5: Validating embeddings...")
            if not validate_embeddings(output_dir):
                logger.warning("⚠️  Validation failed!")
        else:
            logger.info("\n⏭️  Step 5/5: Validation skipped (use --validate to enable)")
        
        # Print statistics
        print_statistics(manifest)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ TEXT EMBEDDINGS GENERATION COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"📂 Embeddings saved to: {output_dir.absolute()}")
        logger.info(f"📄 Manifest: {output_dir.absolute() / 'manifest.json'}")
        logger.info("\n🎯 Next steps:")
        logger.info("  1. Run collaborative model training: python train_hybrid_enhanced.py")
        logger.info("  2. Generate image embeddings: python generate_image_embeddings.py")
        logger.info("  3. Upload embeddings to your server")
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
