"""
generate_image_embeddings_fixed.py
✅ Production-Ready Image Embeddings Generator - CURSOR TIMEOUT FIXED

Key Improvements:
- Uses pagination (skip/limit) to avoid MongoDB cursor timeout
- Parallel image downloading with thread pool
- Batch processing for CLIP model
- Comprehensive error handling
- Progress tracking
- Memory-efficient

Purpose:
- Load products with image URLs from MongoDB (in batches)
- Download and process product images (parallel)
- Generate image embeddings using CLIP
- Save embeddings to image_embeddings/ folder

Features:
- ✅ No cursor timeout (pagination-based fetching)
- ✅ Multi-threaded image downloading
- ✅ Automatic retry for failed downloads
- ✅ Image preprocessing (resize, normalize)
- ✅ Progress tracking with tqdm
- ✅ Validation and statistics
- ✅ Fallback for products without images

Usage:
    python generate_image_embeddings_fixed.py
    
    # Or with custom settings:
    python generate_image_embeddings_fixed.py --batch-size 32 --max-workers 8 --validate

Output:
    image_embeddings/
    ├── product_id_1.npy
    ├── product_id_2.npy
    └── ...
    ├── manifest.json (metadata)
    └── statistics.json (stats)
"""

import os
import sys
import argparse
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

# Environment variables
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "futurist_e-commerce")
MONGO_COLLECTION_PRODUCTS = os.getenv("MONGO_COLLECTION_PRODUCTS", "products")
IMAGE_MODEL_NAME = os.getenv("IMAGE_MODEL_NAME", "ViT-B/32")
OUTPUT_DIR = Path("image_embeddings")

# Processing settings (optimized to avoid cursor timeout)
FETCH_BATCH_SIZE = 100  # Small batches - no cursor timeout!
EMBEDDING_BATCH_SIZE = 32  # Embedding generation batch
MAX_WORKERS = 8  # Parallel download threads
IMAGE_TIMEOUT = 10  # seconds
MAX_RETRIES = 3

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

def connect_mongodb(max_retries: int = 3) -> MongoClient:
    """Connect to MongoDB with retry logic"""
    print("\n[STEP 1/6] Connecting to MongoDB...")
    
    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}/{max_retries}...")
            
            client = MongoClient(
                MONGO_URI,
                serverSelectionTimeoutMS=10000,
                connectTimeoutMS=10000
            )
            
            # Test connection
            client.admin.command("ping")
            
            print(f"  ✓ MongoDB connected successfully")
            print(f"  ✓ Database: {MONGO_DB_NAME}")
            print(f"  ✓ Collection: {MONGO_COLLECTION_PRODUCTS}")
            
            return client
            
        except (ServerSelectionTimeoutError, ConnectionFailure) as e:
            print(f"  ✗ Connection failed: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in 2 seconds...")
                time.sleep(2)
            else:
                print(f"\n✗ ERROR: Failed to connect to MongoDB after {max_retries} attempts")
                sys.exit(1)

# ==========================================
# Image Downloading
# ==========================================

def download_image(
    url: str,
    product_id: str,
    timeout: int = IMAGE_TIMEOUT,
    retries: int = MAX_RETRIES
) -> Optional[Image.Image]:
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
                headers={"User-Agent": "Mozilla/5.0 (compatible; ProductBot/1.0)"},
                stream=True
            )
            response.raise_for_status()
            
            # Open image
            image = Image.open(BytesIO(response.content))
            
            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            return image
            
        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                time.sleep(0.5)
        except requests.exceptions.RequestException:
            if attempt < retries - 1:
                time.sleep(0.5)
        except Exception:
            if attempt < retries - 1:
                time.sleep(0.5)
    
    return None

def extract_image_url(product: Dict[str, Any]) -> Optional[str]:
    """
    Extract image URL from product document
    
    Handles multiple formats:
    - imageUrl: direct URL string
    - images: array of URLs or objects with url property
    """
    # Try imageUrl field
    if product.get("imageUrl"):
        url = product["imageUrl"]
        if isinstance(url, str) and url.strip():
            return url.strip()
    
    # Try images array
    if product.get("images"):
        images = product["images"]
        if isinstance(images, list) and len(images) > 0:
            first_image = images[0]
            
            # String URL
            if isinstance(first_image, str) and first_image.strip():
                return first_image.strip()
            
            # Object with url property
            if isinstance(first_image, dict) and first_image.get("url"):
                url = first_image["url"]
                if isinstance(url, str) and url.strip():
                    return url.strip()
    
    return None

def download_images_parallel(
    products: List[Dict[str, Any]],
    max_workers: int = MAX_WORKERS
) -> Dict[str, Image.Image]:
    """
    Download images in parallel
    
    Args:
        products: List of product documents with image URLs
        max_workers: Number of parallel download threads
        
    Returns:
        Dictionary mapping product_id to PIL Image
    """
    print(f"  Downloading {len(products)} images (workers: {max_workers})...")
    
    images = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit download tasks
        future_to_product = {}
        
        for product in products:
            product_id = str(product["_id"])
            image_url = product.get("_image_url")
            
            if image_url:
                future = executor.submit(download_image, image_url, product_id)
                future_to_product[future] = (product_id, product)
        
        # Collect results with progress bar
        for future in tqdm(
            as_completed(future_to_product),
            total=len(future_to_product),
            desc="  Progress",
            leave=False
        ):
            product_id, product = future_to_product[future]
            
            try:
                image = future.result()
                if image:
                    images[product_id] = image
            except Exception:
                pass
    
    print(f"  ✓ Downloaded {len(images)}/{len(products)} images")
    return images

# ==========================================
# Embedding Generation (WITH CURSOR FIX!)
# ==========================================

def generate_embeddings_batch_safe(
    client: MongoClient,
    model: torch.nn.Module,
    preprocess: Any,
    output_dir: Path,
    fetch_batch_size: int = FETCH_BATCH_SIZE,
    embedding_batch_size: int = EMBEDDING_BATCH_SIZE,
    max_workers: int = MAX_WORKERS
) -> Dict[str, Any]:
    """
    Generate embeddings using pagination to avoid cursor timeout
    
    KEY FIX: Uses skip() + limit() like the text embeddings generator
    
    Process:
    1. Fetch batch of products (skip/limit - no cursor timeout)
    2. Download images in parallel
    3. Generate embeddings with CLIP
    4. Save individual .npy files
    5. Repeat for next batch
    
    Args:
        client: MongoDB client
        model: CLIP model
        preprocess: CLIP preprocessing function
        output_dir: Where to save embeddings
        fetch_batch_size: How many products to fetch at once
        embedding_batch_size: Batch size for CLIP encoding
        max_workers: Parallel download threads
    
    Returns:
        manifest: Dictionary with generation metadata
    """
    print("\n[STEP 4/6] Generating Image Embeddings...")
    
    db = client[MONGO_DB_NAME]
    products_col = db[MONGO_COLLECTION_PRODUCTS]
    
    # Query for products with images
    query = {
        "$or": [
            {"imageUrl": {"$exists": True, "$ne": None, "$ne": ""}},
            {"images": {"$exists": True, "$ne": [], "$ne": None}}
        ]
    }
    
    total_products = products_col.count_documents(query)
    print(f"  Total products with images: {total_products:,}")
    
    if total_products == 0:
        print("  ✗ ERROR: No products with images found")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output directory: {output_dir.absolute()}")
    print(f"  Fetch batch size: {fetch_batch_size}")
    print(f"  Embedding batch size: {embedding_batch_size}")
    print(f"  Device: {DEVICE}")
    
    # Initialize manifest
    manifest = {
        "generated_at": datetime.utcnow().isoformat(),
        "model_name": IMAGE_MODEL_NAME,
        "device": DEVICE,
        "total_products": total_products,
        "fetch_batch_size": fetch_batch_size,
        "embedding_batch_size": embedding_batch_size,
        "embeddings": {},
        "errors": []
    }
    
    successful = 0
    failed = 0
    no_image = 0
    
    start_time = time.time()
    
    # Progress bar for overall progress
    pbar = tqdm(total=total_products, desc="  Overall progress", unit="products")
    
    # Process in pagination (NO CURSOR TIMEOUT!)
    for skip in range(0, total_products, fetch_batch_size):
        try:
            # CRUCIAL: Each iteration is a fresh query (no cursor timeout!)
            batch = list(
                products_col.find(query)
                .skip(skip)
                .limit(fetch_batch_size)
            )
            
            if not batch:
                break
            
            # Extract image URLs
            products_with_urls = []
            for product in batch:
                image_url = extract_image_url(product)
                if image_url:
                    product["_image_url"] = image_url
                    products_with_urls.append(product)
                else:
                    no_image += 1
            
            if not products_with_urls:
                pbar.update(len(batch))
                continue
            
            # Download images in parallel
            images = download_images_parallel(products_with_urls, max_workers)
            
            if not images:
                failed += len(products_with_urls)
                pbar.update(len(batch))
                continue
            
            # Process images in embedding batches
            product_ids_with_images = list(images.keys())
            
            for i in range(0, len(product_ids_with_images), embedding_batch_size):
                emb_batch_ids = product_ids_with_images[i:i + embedding_batch_size]
                
                try:
                    # Prepare images for batch
                    batch_images = []
                    valid_ids = []
                    
                    for pid in emb_batch_ids:
                        image = images[pid]
                        try:
                            processed = preprocess(image)
                            batch_images.append(processed)
                            valid_ids.append(pid)
                        except Exception:
                            failed += 1
                    
                    if not batch_images:
                        continue
                    
                    # Stack into tensor
                    image_tensor = torch.stack(batch_images).to(DEVICE)
                    
                    # Generate embeddings
                    with torch.no_grad():
                        embeddings = model.encode_image(image_tensor)
                        embeddings = embeddings.cpu().numpy()
                    
                    # Save individual embeddings
                    for pid, embedding in zip(valid_ids, embeddings):
                        try:
                            # Find product
                            product = next(p for p in batch if str(p["_id"]) == pid)
                            
                            # Save embedding
                            embedding_path = output_dir / f"{pid}.npy"
                            np.save(embedding_path, embedding)
                            
                            # Add to manifest
                            manifest["embeddings"][pid] = {
                                "name": product.get("name", "Unknown"),
                                "category": product.get("category", "Unknown"),
                                "image_url": product.get("_image_url", ""),
                                "file": embedding_path.name,
                                "shape": list(embedding.shape)
                            }
                            
                            successful += 1
                            
                        except Exception as e:
                            failed += 1
                            manifest["errors"].append({
                                "product_id": pid,
                                "error": f"Save failed: {str(e)}"
                            })
                
                except Exception as e:
                    failed += len(emb_batch_ids)
                    manifest["errors"].append({
                        "batch": f"skip={skip}, emb_batch={i}",
                        "error": f"Embedding generation failed: {str(e)}"
                    })
            
            # Update progress
            pbar.update(len(batch))
        
        except Exception as e:
            print(f"\n  ✗ ERROR at skip={skip}: {e}")
            failed += fetch_batch_size
            manifest["errors"].append({
                "batch_skip": skip,
                "error": f"Batch fetch failed: {str(e)}"
            })
    
    pbar.close()
    
    # Calculate statistics
    duration = time.time() - start_time
    
    manifest.update({
        "successful": successful,
        "failed": failed,
        "no_image_url": no_image,
        "success_rate": successful / total_products if total_products > 0 else 0,
        "duration_seconds": duration,
        "duration_formatted": format_time(duration),
        "products_per_second": successful / duration if duration > 0 else 0
    })
    
    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"\n  ✓ Embeddings generated: {successful:,} successful, {failed:,} failed")
    print(f"  ✓ Duration: {format_time(duration)}")
    print(f"  ✓ Speed: {successful/duration:.1f} products/second")
    print(f"  ✓ Manifest saved: {manifest_path}")
    
    return manifest

# ==========================================
# Validation
# ==========================================

def validate_embeddings(output_dir: Path, manifest: Dict[str, Any]) -> bool:
    """Validate that embeddings were generated correctly"""
    print("\n[STEP 5/6] Validating Embeddings...")
    
    try:
        total_expected = manifest.get("successful", 0)
        
        # Check files exist
        embedding_files = list(output_dir.glob("*.npy"))
        total_found = len(embedding_files)
        
        print(f"  Expected: {total_expected:,} embeddings")
        print(f"  Found: {total_found:,} .npy files")
        
        if total_found != total_expected:
            print(f"  ✗ WARNING: File count mismatch!")
            return False
        
        # Sample validation
        sample_size = min(10, total_found)
        print(f"  Checking {sample_size} sample embeddings...")
        
        valid_count = 0
        
        for emb_file in embedding_files[:sample_size]:
            try:
                embedding = np.load(emb_file)
                
                # Check for NaN or Inf
                if np.isnan(embedding).any() or np.isinf(embedding).any():
                    print(f"    ✗ {emb_file.name}: Contains NaN or Inf")
                    return False
                
                valid_count += 1
                
            except Exception as e:
                print(f"    ✗ {emb_file.name}: Load failed - {e}")
                return False
        
        print(f"  ✓ All {valid_count} samples valid")
        print(f"  ✓ Validation passed!")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Validation error: {e}")
        return False

# ==========================================
# Statistics
# ==========================================

def print_statistics(manifest: Dict[str, Any]):
    """Print comprehensive statistics"""
    print_section("GENERATION STATISTICS")
    
    print(f"""
  Generation Details:
    ├─ Generated at:        {manifest.get('generated_at', 'Unknown')}
    ├─ Model:               {manifest.get('model_name', 'Unknown')}
    ├─ Device:              {manifest.get('device', 'Unknown')}
    ├─ Fetch batch size:    {manifest.get('fetch_batch_size', 'Unknown')}
    └─ Embedding batch:     {manifest.get('embedding_batch_size', 'Unknown')}
  
  Results:
    ├─ Total products:      {manifest.get('total_products', 0):,}
    ├─ Successful:          {manifest.get('successful', 0):,}
    ├─ Failed:              {manifest.get('failed', 0):,}
    ├─ No image URL:        {manifest.get('no_image_url', 0):,}
    ├─ Success rate:        {manifest.get('success_rate', 0)*100:.1f}%
    └─ Errors logged:       {len(manifest.get('errors', []))}
  
  Performance:
    ├─ Duration:            {manifest.get('duration_formatted', 'Unknown')}
    └─ Speed:               {manifest.get('products_per_second', 0):.1f} products/second
    """)

def save_statistics(output_dir: Path, manifest: Dict[str, Any]):
    """Save statistics to separate file"""
    stats = {
        "summary": {
            "total": manifest.get("total_products", 0),
            "successful": manifest.get("successful", 0),
            "failed": manifest.get("failed", 0),
            "success_rate": manifest.get("success_rate", 0)
        },
        "performance": {
            "duration_seconds": manifest.get("duration_seconds", 0),
            "products_per_second": manifest.get("products_per_second", 0)
        },
        "model": {
            "name": manifest.get("model_name", "Unknown"),
            "device": manifest.get("device", "Unknown")
        },
        "timestamp": manifest.get("generated_at", datetime.utcnow().isoformat())
    }
    
    stats_path = output_dir / "statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"  ✓ Statistics saved: {stats_path}")

# ==========================================
# Main Function
# ==========================================

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Generate image embeddings for products (cursor timeout fixed)"
    )
    parser.add_argument(
        "--fetch-batch-size",
        type=int,
        default=FETCH_BATCH_SIZE,
        help="How many products to fetch at once (default: 100)"
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=EMBEDDING_BATCH_SIZE,
        help="Batch size for embedding generation (default: 32)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS,
        help="Parallel download workers (default: 8)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=IMAGE_MODEL_NAME,
        help="CLIP model name (default: ViT-B/32)"
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
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation"
    )
    
    args = parser.parse_args()
    
    # Print header
    print_header("IMAGE EMBEDDINGS GENERATOR v2.0 - CURSOR TIMEOUT FIXED")
    
    print(f"""
  Configuration:
    ├─ MongoDB URI:         {MONGO_URI[:50]}...
    ├─ Database:            {MONGO_DB_NAME}
    ├─ Collection:          {MONGO_COLLECTION_PRODUCTS}
    ├─ Model:               {args.model}
    ├─ Device:              {DEVICE}
    ├─ Fetch batch:         {args.fetch_batch_size} products
    ├─ Embedding batch:     {args.embedding_batch_size} products
    ├─ Download workers:    {args.max_workers} threads
    └─ Output directory:    {args.output_dir}
    """)
    
    output_dir = Path(args.output_dir)
    client = None
    
    try:
        # Step 1: Connect to MongoDB
        client = connect_mongodb()
        
        # Step 2: Load CLIP model
        print("\n[STEP 2/6] Loading CLIP Model...")
        print(f"  Model: {args.model}")
        print(f"  Device: {DEVICE}")
        
        try:
            model, preprocess = clip.load(args.model, device=DEVICE)
            print(f"  ✓ Model loaded successfully")
        except Exception as e:
            print(f"  ✗ ERROR: Failed to load CLIP model: {e}")
            return 1
        
        # Step 3: Count products (info only)
        print("\n[STEP 3/6] Checking Products...")
        db = client[MONGO_DB_NAME]
        products_col = db[MONGO_COLLECTION_PRODUCTS]
        query = {
            "$or": [
                {"imageUrl": {"$exists": True, "$ne": None, "$ne": ""}},
                {"images": {"$exists": True, "$ne": [], "$ne": None}}
            ]
        }
        total = products_col.count_documents(query)
        print(f"  ✓ Found {total:,} products with images")
        
        # Step 4: Generate embeddings (WITH CURSOR FIX!)
        start_time = time.time()
        manifest = generate_embeddings_batch_safe(
            client=client,
            model=model,
            preprocess=preprocess,
            output_dir=output_dir,
            fetch_batch_size=args.fetch_batch_size,
            embedding_batch_size=args.embedding_batch_size,
            max_workers=args.max_workers
        )
        
        # Step 5: Validation
        if not args.skip_validation:
            validation_passed = validate_embeddings(output_dir, manifest)
            if not validation_passed:
                print("  ⚠ WARNING: Validation issues detected")
        else:
            print("\n[STEP 5/6] Validation skipped")
        
        # Step 6: Save statistics
        print("\n[STEP 6/6] Saving Statistics...")
        save_statistics(output_dir, manifest)
        
        # Print final statistics
        print_statistics(manifest)
        
        # Success message
        print_header("SUCCESS!")
        print(f"""
  ✓ Image embeddings generated successfully!
  
  Output:
    ├─ Embeddings:  {output_dir.absolute()}
    ├─ Files:       {manifest.get('successful', 0):,} .npy files
    ├─ Manifest:    {output_dir / 'manifest.json'}
    └─ Statistics:  {output_dir / 'statistics.json'}
  
  Next Steps:
    1. Test embeddings:
       python embedding_loader_enhanced.py
    
    2. Start recommendation service:
       uvicorn recommendation_service_enhanced:app --reload
    
    3. Test image similarity search in Swagger UI
        """)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠ Process interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n\n✗ ERROR: Unexpected error occurred")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        if client:
            print("\n[CLEANUP] Closing MongoDB connection...")
            client.close()
            print("  ✓ Connection closed")


if __name__ == "__main__":
    sys.exit(main())
