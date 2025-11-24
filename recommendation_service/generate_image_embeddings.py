"""
generate_image_embeddings.py
✅ Production-Ready Image Embeddings Generator

Purpose:
- Load products with image URLs from MongoDB
- Download and process product images
- Generate image embeddings using CLIP
- Save embeddings to image_embeddings/ folder
- Memory-efficient batch processing
- Error handling for missing/invalid images

Features:
- Multi-threaded image downloading
- Automatic retry for failed downloads
- Image preprocessing (resize, normalize)
- Progress tracking with tqdm
- Validation and statistics
- Fallback for products without images

Usage:
    python generate_image_embeddings.py
    
    # Or with custom settings:
    python generate_image_embeddings.py --batch-size 32 --max-workers 4

Output:
    image_embeddings/
    ├── product_id_1.npy
    ├── product_id_2.npy
    └── ...
    
    image_embeddings/manifest.json (metadata)
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import numpy as np
import torch
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
from PIL import Image
import clip
import requests
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
        logging.FileHandler('image_embeddings_training.log')
    ]
)
logger = logging.getLogger("ImageEmbeddingsGenerator")

# ==========================================
# Configuration from Environment
# ==========================================
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "futurist_ecommerce")
MONGO_COLLECTION_PRODUCTS = os.getenv("MONGO_COLLECTION_PRODUCTS", "products")
IMAGE_MODEL_NAME = os.getenv("IMAGE_MODEL_NAME", "ViT-B/32")
OUTPUT_DIR = Path("image_embeddings")
BATCH_SIZE = int(os.getenv("IMAGE_EMBEDDING_BATCH_SIZE", "32"))
MAX_WORKERS = int(os.getenv("IMAGE_DOWNLOAD_WORKERS", "4"))
IMAGE_TIMEOUT = int(os.getenv("IMAGE_DOWNLOAD_TIMEOUT", "10"))
MAX_RETRIES = int(os.getenv("IMAGE_DOWNLOAD_RETRIES", "3"))

# CLIP device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
def fetch_products_with_images(client: MongoClient) -> List[Dict[str, Any]]:
    """Fetch all products that have image URLs"""
    try:
        db = client[MONGO_DB_NAME]
        products_col = db[MONGO_COLLECTION_PRODUCTS]
        
        # Query for products with imageUrl
        query = {
            "$or": [
                {"imageUrl": {"$exists": True, "$ne": None, "$ne": ""}},
                {"images": {"$exists": True, "$ne": [], "$ne": None}}
            ]
        }
        
        total = products_col.count_documents(query)
        logger.info(f"📊 Found {total} products with images")
        
        if total == 0:
            logger.warning("⚠️  No products with images found!")
            return []
        
        # Fetch products
        products = []
        cursor = products_col.find(query)
        
        for product in tqdm(cursor, total=total, desc="Loading products"):
            # Normalize image URL
            image_url = None
            
            if product.get("imageUrl"):
                image_url = product["imageUrl"]
            elif product.get("images") and isinstance(product["images"], list) and len(product["images"]) > 0:
                # Use first image from images array
                first_image = product["images"][0]
                if isinstance(first_image, str):
                    image_url = first_image
                elif isinstance(first_image, dict) and "url" in first_image:
                    image_url = first_image["url"]
            
            if image_url:
                product["_normalized_image_url"] = image_url
                products.append(product)
        
        logger.info(f"✅ Loaded {len(products)} products with valid image URLs")
        return products
        
    except Exception as e:
        logger.error(f"❌ Error fetching products: {e}")
        return []


# ==========================================
# Image Downloading
# ==========================================
def download_image(url: str, product_id: str, timeout: int = IMAGE_TIMEOUT, retries: int = MAX_RETRIES) -> Optional[Image.Image]:
    """
    Download image from URL with retry logic
    
    Args:
        url: Image URL
        product_id: Product ID (for logging)
        timeout: Download timeout in seconds
        retries: Number of retry attempts
        
    Returns:
        PIL Image or None if failed
    """
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            response.raise_for_status()
            
            # Open image
            image = Image.open(BytesIO(response.content))
            
            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            return image
            
        except requests.exceptions.Timeout:
            logger.warning(f"⏱️  Timeout downloading image for {product_id} (attempt {attempt + 1}/{retries})")
        except requests.exceptions.RequestException as e:
            logger.warning(f"🌐 Network error for {product_id}: {e} (attempt {attempt + 1}/{retries})")
        except Exception as e:
            logger.warning(f"🖼️  Image processing error for {product_id}: {e} (attempt {attempt + 1}/{retries})")
        
        if attempt < retries - 1:
            time.sleep(1)  # Wait before retry
    
    logger.error(f"❌ Failed to download image for {product_id} after {retries} attempts")
    return None


def download_images_batch(
    products: List[Dict[str, Any]],
    max_workers: int = MAX_WORKERS
) -> Dict[str, Image.Image]:
    """
    Download images in parallel
    
    Args:
        products: List of product documents
        max_workers: Number of parallel download threads
        
    Returns:
        Dictionary mapping product_id to PIL Image
    """
    logger.info(f"🔄 Downloading {len(products)} images with {max_workers} workers...")
    
    images = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit download tasks
        future_to_product = {
            executor.submit(
                download_image,
                product["_normalized_image_url"],
                str(product["_id"])
            ): product
            for product in products
        }
        
        # Collect results with progress bar
        for future in tqdm(
            as_completed(future_to_product),
            total=len(products),
            desc="Downloading images"
        ):
            product = future_to_product[future]
            product_id = str(product["_id"])
            
            try:
                image = future.result()
                if image:
                    images[product_id] = image
            except Exception as e:
                logger.error(f"❌ Unexpected error processing {product_id}: {e}")
    
    logger.info(f"✅ Successfully downloaded {len(images)}/{len(products)} images")
    return images


# ==========================================
# Embedding Generation
# ==========================================
def generate_embeddings(
    products: List[Dict[str, Any]],
    images: Dict[str, Image.Image],
    model: torch.nn.Module,
    preprocess: Any,
    batch_size: int = BATCH_SIZE,
    output_dir: Path = OUTPUT_DIR
) -> Dict[str, Any]:
    """
    Generate image embeddings for all products
    
    Returns:
        manifest: Dictionary with metadata about generated embeddings
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"🔄 Generating image embeddings...")
    logger.info(f"📦 Batch size: {batch_size}")
    logger.info(f"💾 Output directory: {output_dir}")
    logger.info(f"🖥️  Device: {DEVICE}")
    
    manifest = {
        "generated_at": datetime.utcnow().isoformat(),
        "model_name": IMAGE_MODEL_NAME,
        "device": DEVICE,
        "total_products": len(products),
        "embeddings": {}
    }
    
    successful = 0
    failed = 0
    skipped = 0
    
    # Filter products with downloaded images
    products_with_images = [p for p in products if str(p["_id"]) in images]
    
    logger.info(f"📊 Processing {len(products_with_images)} products with images")
    
    # Process in batches
    for i in tqdm(range(0, len(products_with_images), batch_size), desc="Generating embeddings"):
        batch = products_with_images[i:i + batch_size]
        
        try:
            # Prepare images for batch processing
            batch_images = []
            batch_product_ids = []
            
            for product in batch:
                product_id = str(product["_id"])
                image = images.get(product_id)
                
                if image:
                    batch_images.append(preprocess(image))
                    batch_product_ids.append(product_id)
            
            if not batch_images:
                continue
            
            # Stack images into batch tensor
            image_tensor = torch.stack(batch_images).to(DEVICE)
            
            # Generate embeddings
            with torch.no_grad():
                embeddings = model.encode_image(image_tensor)
                embeddings = embeddings.cpu().numpy()
            
            # Save individual embeddings
            for product_id, embedding in zip(batch_product_ids, embeddings):
                try:
                    # Find product data
                    product = next(p for p in batch if str(p["_id"]) == product_id)
                    
                    # Save embedding as .npy file
                    embedding_path = output_dir / f"{product_id}.npy"
                    np.save(embedding_path, embedding)
                    
                    # Add to manifest
                    manifest["embeddings"][product_id] = {
                        "name": product.get("name", "Unknown"),
                        "category": product.get("category", "Unknown"),
                        "image_url": product.get("_normalized_image_url", ""),
                        "file": str(embedding_path.name),
                        "shape": list(embedding.shape)
                    }
                    
                    successful += 1
                    
                except Exception as e:
                    logger.warning(f"⚠️  Failed to save embedding for product {product_id}: {e}")
                    failed += 1
        
        except Exception as e:
            logger.error(f"❌ Batch processing error: {e}")
            failed += len(batch)
    
    # Count skipped products (no image downloaded)
    skipped = len(products) - len(products_with_images)
    
    # Update manifest
    manifest["successful"] = successful
    manifest["failed"] = failed
    manifest["skipped"] = skipped
    manifest["success_rate"] = successful / len(products) if len(products) > 0 else 0
    
    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"✅ Embeddings generated: {successful} successful, {failed} failed, {skipped} skipped")
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
    logger.info("📊 IMAGE EMBEDDING GENERATION STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Generated at: {manifest.get('generated_at', 'Unknown')}")
    logger.info(f"Model: {manifest.get('model_name', 'Unknown')}")
    logger.info(f"Device: {manifest.get('device', 'Unknown')}")
    logger.info(f"Total products: {manifest.get('total_products', 0)}")
    logger.info(f"Successful: {manifest.get('successful', 0)}")
    logger.info(f"Failed: {manifest.get('failed', 0)}")
    logger.info(f"Skipped (no image): {manifest.get('skipped', 0)}")
    logger.info(f"Success rate: {manifest.get('success_rate', 0) * 100:.2f}%")
    logger.info("=" * 80)


# ==========================================
# Main Function
# ==========================================
def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Generate image embeddings for products")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for embedding generation"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS,
        help="Number of parallel download workers"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=IMAGE_MODEL_NAME,
        help="CLIP model name"
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
    logger.info("🚀 IMAGE EMBEDDINGS GENERATOR")
    logger.info("=" * 80)
    logger.info(f"MongoDB URI: {MONGO_URI[:50]}...")
    logger.info(f"Database: {MONGO_DB_NAME}")
    logger.info(f"Collection: {MONGO_COLLECTION_PRODUCTS}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Max workers: {args.max_workers}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)
    
    output_dir = Path(args.output_dir)
    
    try:
        # Step 1: Connect to MongoDB
        logger.info("\n📡 Step 1/6: Connecting to MongoDB...")
        client = connect_mongodb()
        if not client:
            logger.error("❌ Failed to connect to MongoDB. Exiting.")
            return 1
        
        # Step 2: Fetch products with images
        logger.info("\n📦 Step 2/6: Fetching products with images...")
        products = fetch_products_with_images(client)
        if not products:
            logger.error("❌ No products with images found. Exiting.")
            return 1
        
        # Step 3: Load CLIP model
        logger.info(f"\n🤖 Step 3/6: Loading CLIP model '{args.model}'...")
        try:
            model, preprocess = clip.load(args.model, device=DEVICE)
            logger.info(f"✅ Model loaded on {DEVICE}")
        except Exception as e:
            logger.error(f"❌ Failed to load CLIP model: {e}")
            return 1
        
        # Step 4: Download images
        logger.info("\n🌐 Step 4/6: Downloading product images...")
        start_time = time.time()
        images = download_images_batch(products, max_workers=args.max_workers)
        duration = time.time() - start_time
        logger.info(f"⏱️  Download completed in {duration:.2f} seconds")
        
        if not images:
            logger.error("❌ No images downloaded successfully. Exiting.")
            return 1
        
        # Step 5: Generate embeddings
        logger.info("\n🔄 Step 5/6: Generating image embeddings...")
        start_time = time.time()
        manifest = generate_embeddings(
            products=products,
            images=images,
            model=model,
            preprocess=preprocess,
            batch_size=args.batch_size,
            output_dir=output_dir
        )
        duration = time.time() - start_time
        logger.info(f"⏱️  Generation completed in {duration:.2f} seconds")
        
        # Step 6: Validation
        if args.validate:
            logger.info("\n✓ Step 6/6: Validating embeddings...")
            if not validate_embeddings(output_dir):
                logger.warning("⚠️  Validation failed!")
        else:
            logger.info("\n⏭️  Step 6/6: Validation skipped (use --validate to enable)")
        
        # Print statistics
        print_statistics(manifest)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ IMAGE EMBEDDINGS GENERATION COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"📂 Embeddings saved to: {output_dir.absolute()}")
        logger.info(f"📄 Manifest: {output_dir.absolute() / 'manifest.json'}")
        logger.info("\n🎯 Next steps:")
        logger.info("  1. Verify embeddings are loaded correctly")
        logger.info("  2. Upload embeddings to your server")
        logger.info("  3. Restart recommendation service to load new embeddings")
        logger.info("  4. Test image similarity search")
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
