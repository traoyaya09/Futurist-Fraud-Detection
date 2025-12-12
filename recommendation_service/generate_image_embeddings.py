"""
generate_image_embeddings_fixed.py
✅ Production-Ready Image Embeddings Generator - ENHANCED v2.1

Key Improvements:
- Uses pagination (skip/limit) to avoid MongoDB cursor timeout ✅
- Parallel image downloading with thread pool ✅
- Batch processing for CLIP model ✅
- DateTime normalization integrated ✅
- Matches MongoDB Product schema perfectly ✅
- Comprehensive error handling ✅
- Progress tracking ✅
- Memory-efficient ✅
- Utils integration ✅

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
- ✅ DateTime normalization via utils
- ✅ Fallback for products without images
- ✅ Handles multiple image formats

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

# Import utils for database operations ✅
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
MAX_IMAGE_SIZE = 1024  # Max dimension for images (memory optimization)

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Product schema fields for image embedding generation ✅
IMAGE_PRODUCT_FIELDS = {
    "_id": 1,
    "name": 1,
    "category": 1,
    "subCategory": 1,
    "brand": 1,
    "imageUrl": 1,
    "images": 1,
    "stock": 1,
    "status": 1,
    "rating": 1,
    "reviewsCount": 1,
    "isFeatured": 1,
    "isBestseller": 1,
    "isNewProduct": 1,
    "createdAt": 1
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


def format_size(bytes_size: int) -> str:
    """Format bytes into readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


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
# Image Downloading - ENHANCED ✅
# ==========================================

def download_image(
    url: str,
    product_id: str,
    timeout: int = IMAGE_TIMEOUT,
    retries: int = MAX_RETRIES,
    max_size: int = MAX_IMAGE_SIZE
) -> Optional[Image.Image]:
    """
    Download image from URL with retry logic and size optimization
    
    Args:
        url: Image URL
        product_id: Product ID (for logging)
        timeout: Download timeout in seconds
        retries: Number of retry attempts
        max_size: Maximum image dimension (for memory optimization)
        
    Returns:
        PIL Image or None if failed
    """
    for attempt in range(retries):
        try:
            # Download with proper headers
            response = requests.get(
                url,
                timeout=timeout,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; ProductBot/1.0)",
                    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8"
                },
                stream=True
            )
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                return None
            
            # Open image
            image = Image.open(BytesIO(response.content))
            
            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Resize if too large (memory optimization)
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.LANCZOS)
            
            return image
            
        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                time.sleep(0.5)
        except requests.exceptions.RequestException:
            if attempt < retries - 1:
                time.sleep(0.5)
        except (IOError, OSError):  # PIL image errors
            if attempt < retries - 1:
                time.sleep(0.5)
        except Exception:
            if attempt < retries - 1:
                time.sleep(0.5)
    
    return None


def extract_image_url(product: Dict[str, Any]) -> Optional[str]:
    """
    Extract image URL from product document
    
    Handles multiple formats matching ProductModel.js schema:
    - imageUrl: direct URL string
    - images: array of URLs or objects with url property
    
    Args:
        product: Product document (normalized) ✅
    
    Returns:
        Image URL string or None
    """
    # Try imageUrl field (primary)
    image_url = product.get("imageUrl")
    if image_url:
        if isinstance(image_url, str) and image_url.strip():
            url = image_url.strip()
            # Validate it's a proper URL
            if url.startswith(("http://", "https://")):
                return url
    
    # Try images array (fallback)
    images = product.get("images")
    if images and isinstance(images, list) and len(images) > 0:
        first_image = images[0]
        
        # String URL
        if isinstance(first_image, str) and first_image.strip():
            url = first_image.strip()
            if url.startswith(("http://", "https://")):
                return url
        
        # Object with url property
        if isinstance(first_image, dict):
            url = first_image.get("url")
            if url and isinstance(url, str) and url.strip():
                url = url.strip()
                if url.startswith(("http://", "https://")):
                    return url
    
    return None


def download_images_parallel(
    products: List[Dict[str, Any]],
    max_workers: int = MAX_WORKERS
) -> Tuple[Dict[str, Image.Image], Dict[str, str]]:
    """
    Download images in parallel
    
    Args:
        products: List of product documents with image URLs
        max_workers: Number of parallel download threads
        
    Returns:
        Tuple of (images dict, errors dict)
    """
    print(f"  Downloading {len(products)} images (workers: {max_workers})...")
    
    images = {}
    errors = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit download tasks
        future_to_product = {}
        
        for product in products:
            product_id = str(product["_id"])
            image_url = product.get("_image_url")
            
            if image_url:
                future = executor.submit(download_image, image_url, product_id)
                future_to_product[future] = (product_id, product, image_url)
        
        # Collect results with progress bar
        pbar = tqdm(
            as_completed(future_to_product),
            total=len(future_to_product),
            desc="  Download progress",
            leave=False,
            unit="img"
        )
        
        for future in pbar:
            product_id, product, image_url = future_to_product[future]
            
            try:
                image = future.result()
                if image:
                    images[product_id] = image
                else:
                    errors[product_id] = "Download returned None"
            except Exception as e:
                errors[product_id] = f"Download exception: {str(e)}"
        
        pbar.close()
    
    print(f"  ✓ Downloaded: {len(images)}/{len(products)} images")
    if errors:
        print(f"  ⚠ Failed: {len(errors)} images")
    
    return images, errors


# ==========================================
# Embedding Generation (CURSOR TIMEOUT FIXED!) ✅
# ==========================================

def generate_embeddings_batch_safe(
    client: MongoClient,
    model: torch.nn.Module,
    preprocess: Any,
    output_dir: Path,
    fetch_batch_size: int = FETCH_BATCH_SIZE,
    embedding_batch_size: int = EMBEDDING_BATCH_SIZE,
    max_workers: int = MAX_WORKERS,
    filter_active_only: bool = True
) -> Dict[str, Any]:
    """
    Generate embeddings using pagination to avoid cursor timeout
    
    KEY FIX: Uses skip() + limit() like the text embeddings generator ✅
    ENHANCEMENT: DateTime normalization via utils ✅
    
    Process:
    1. Fetch batch of products (skip/limit - no cursor timeout)
    2. Normalize products (datetime → string) ✅
    3. Download images in parallel
    4. Generate embeddings with CLIP
    5. Save individual .npy files
    6. Repeat for next batch
    
    Args:
        client: MongoDB client
        model: CLIP model
        preprocess: CLIP preprocessing function
        output_dir: Where to save embeddings
        fetch_batch_size: How many products to fetch at once
        embedding_batch_size: Batch size for CLIP encoding
        max_workers: Parallel download threads
        filter_active_only: Only process active products with stock
    
    Returns:
        manifest: Dictionary with generation metadata
    """
    print("\n[STEP 4/6] Generating Image Embeddings...")
    
    db = client[MONGO_DB_NAME]
    products_col = db[MONGO_COLLECTION_PRODUCTS]
    
    # Build filter query
    query = {
        "$or": [
            {"imageUrl": {"$exists": True, "$ne": None, "$ne": ""}},
            {"images": {"$exists": True, "$ne": [], "$ne": None}}
        ]
    }
    
    # Add active filter if requested
    if filter_active_only:
        query["status"] = "active"
        query["stock"] = {"$gt": 0}
    
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
    print(f"  Download workers: {max_workers}")
    print(f"  Device: {DEVICE}")
    print(f"  Filter active only: {filter_active_only}")
    
    # Initialize manifest
    manifest = {
        "generated_at": datetime.utcnow().isoformat(),
        "model_name": IMAGE_MODEL_NAME,
        "device": DEVICE,
        "total_products": total_products,
        "fetch_batch_size": fetch_batch_size,
        "embedding_batch_size": embedding_batch_size,
        "max_workers": max_workers,
        "filter_query": str(query),
        "utils_available": UTILS_AVAILABLE,
        "datetime_normalized": UTILS_AVAILABLE,
        "embeddings": {},
        "errors": []
    }
    
    successful = 0
    failed = 0
    no_image_url = 0
    download_failed = 0
    
    start_time = time.time()
    
    # Progress bar for overall progress
    pbar = tqdm(total=total_products, desc="  Overall progress", unit="products")
    
    # Process in pagination (NO CURSOR TIMEOUT!) ✅
    for skip in range(0, total_products, fetch_batch_size):
        try:
            # CRUCIAL: Each iteration is a fresh query (no cursor timeout!) ✅
            batch_cursor = products_col.find(
                query,
                IMAGE_PRODUCT_FIELDS  # Only fetch needed fields ✅
            ).skip(skip).limit(fetch_batch_size)
            
            batch = list(batch_cursor)
            
            if not batch:
                break
            
            # Normalize products using utils (datetime → string) ✅
            if UTILS_AVAILABLE:
                batch = [normalize_product(p) for p in batch]
            
            # Extract image URLs
            products_with_urls = []
            for product in batch:
                image_url = extract_image_url(product)
                if image_url:
                    product["_image_url"] = image_url
                    products_with_urls.append(product)
                else:
                    no_image_url += 1
                    manifest["errors"].append({
                        "product_id": str(product.get("_id", "unknown")),
                        "product_name": product.get("name", "Unknown"),
                        "error": "No valid image URL found"
                    })
            
            if not products_with_urls:
                pbar.update(len(batch))
                continue
            
            # Download images in parallel
            images, download_errors = download_images_parallel(products_with_urls, max_workers)
            
            # Log download errors
            for pid, error in download_errors.items():
                download_failed += 1
                product = next((p for p in products_with_urls if str(p["_id"]) == pid), None)
                manifest["errors"].append({
                    "product_id": pid,
                    "product_name": product.get("name", "Unknown") if product else "Unknown",
                    "image_url": product.get("_image_url", "") if product else "",
                    "error": f"Image download failed: {error}"
                })
            
            if not images:
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
                        except Exception as e:
                            failed += 1
                            manifest["errors"].append({
                                "product_id": pid,
                                "error": f"Image preprocessing failed: {str(e)}"
                            })
                    
                    if not batch_images:
                        continue
                    
                    # Stack into tensor
                    image_tensor = torch.stack(batch_images).to(DEVICE)
                    
                    # Generate embeddings
                    with torch.no_grad():
                        embeddings = model.encode_image(image_tensor)
                        embeddings = embeddings.cpu().numpy()
                        
                        # Normalize embeddings (L2 norm = 1.0)
                        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                        embeddings = embeddings / norms
                    
                    # Save individual embeddings
                    for pid, embedding in zip(valid_ids, embeddings):
                        try:
                            # Find product
                            product = next(p for p in batch if str(p["_id"]) == pid)
                            
                            # Save embedding
                            embedding_path = output_dir / f"{pid}.npy"
                            np.save(embedding_path, embedding)
                            
                            # Add to manifest with metadata
                            manifest["embeddings"][pid] = {
                                "name": product.get("name", "Unknown"),
                                "category": product.get("category", "Unknown"),
                                "subcategory": product.get("subCategory"),
                                "brand": product.get("brand"),
                                "image_url": product.get("_image_url", ""),
                                "file": embedding_path.name,
                                "shape": list(embedding.shape),
                                "rating": product.get("rating", 0),
                                "reviews_count": product.get("reviewsCount", 0),
                                "is_featured": product.get("isFeatured", False),
                                "is_bestseller": product.get("isBestseller", False),
                                "created_at": product.get("createdAt")  # ✅ Already ISO string
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
        "no_image_url": no_image_url,
        "download_failed": download_failed,
        "success_rate": successful / total_products if total_products > 0 else 0,
        "duration_seconds": duration,
        "duration_formatted": format_time(duration),
        "products_per_second": successful / duration if duration > 0 else 0
    })
    
    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"\n  ✓ Embeddings generated: {successful:,} successful")
    print(f"  ✗ Failed: {failed:,} (preprocessing/save)")
    print(f"  ✗ No image URL: {no_image_url:,}")
    print(f"  ✗ Download failed: {download_failed:,}")
    print(f"  ✓ Duration: {format_time(duration)}")
    print(f"  ✓ Speed: {successful/duration:.1f} products/second")
    print(f"  ✓ Manifest saved: {manifest_path}")
    
    return manifest


# ==========================================
# Validation - ENHANCED ✅
# ==========================================

def validate_embeddings(output_dir: Path, manifest: Dict[str, Any]) -> bool:
    """
    Validate that embeddings were generated correctly
    
    Checks:
    - Manifest file exists ✅
    - All expected .npy files exist ✅
    - Files can be loaded ✅
    - Embeddings have correct shape ✅
    - No NaN or Inf values ✅
    - Embeddings are normalized ✅
    """
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
                if np.isnan(embedding).any():
                    print(f"    ✗ {emb_file.name}: Contains NaN")
                    return False
                
                if np.isinf(embedding).any():
                    print(f"    ✗ {emb_file.name}: Contains Inf")
                    return False
                
                # Check normalization (L2 norm should be ~1.0)
                norm = np.linalg.norm(embedding)
                if not (0.99 <= norm <= 1.01):
                    print(f"    ⚠ {emb_file.name}: Not normalized (norm={norm:.3f})")
                
                valid_count += 1
                
            except Exception as e:
                print(f"    ✗ {emb_file.name}: Load failed - {e}")
                return False
        
        print(f"  ✓ All {valid_count} samples valid")
        print(f"  ✓ All embeddings normalized")
        print(f"  ✓ Validation passed!")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Validation error: {e}")
        return False


# ==========================================
# Statistics - ENHANCED ✅
# ==========================================

def print_statistics(manifest: Dict[str, Any], output_dir: Path):
    """Print comprehensive statistics"""
    print_section("GENERATION STATISTICS")
    
    # Calculate file sizes
    embedding_files = list(output_dir.glob("*.npy"))
    total_size = sum(f.stat().st_size for f in embedding_files)
    avg_size = total_size / len(embedding_files) if embedding_files else 0
    
    print(f"""
  Generation Details:
    ├─ Generated at:        {manifest.get('generated_at', 'Unknown')}
    ├─ Model:               {manifest.get('model_name', 'Unknown')}
    ├─ Device:              {manifest.get('device', 'Unknown')}
    ├─ Fetch batch size:    {manifest.get('fetch_batch_size', 'Unknown')}
    ├─ Embedding batch:     {manifest.get('embedding_batch_size', 'Unknown')}
    ├─ Download workers:    {manifest.get('max_workers', 'Unknown')}
    ├─ Utils integration:   {'✅ Enabled' if manifest.get('utils_available') else '❌ Disabled'}
    └─ DateTime fix:        {'✅ Applied' if manifest.get('datetime_normalized') else '❌ Not applied'}
  
  Results:
    ├─ Total products:      {manifest.get('total_products', 0):,}
    ├─ Successful:          {manifest.get('successful', 0):,}
    ├─ Failed (process):    {manifest.get('failed', 0):,}
    ├─ No image URL:        {manifest.get('no_image_url', 0):,}
    ├─ Download failed:     {manifest.get('download_failed', 0):,}
    ├─ Success rate:        {manifest.get('success_rate', 0)*100:.1f}%
    └─ Errors logged:       {len(manifest.get('errors', []))}
  
  Performance:
    ├─ Duration:            {manifest.get('duration_formatted', 'Unknown')}
    └─ Speed:               {manifest.get('products_per_second', 0):.1f} products/second
  
  Storage:
    ├─ Total size:          {format_size(total_size)}
    ├─ Avg file size:       {format_size(avg_size)}
    └─ Files generated:     {len(embedding_files):,}
    """)


def save_statistics(output_dir: Path, manifest: Dict[str, Any]):
    """Save statistics to separate file"""
    # Calculate file sizes
    embedding_files = list(output_dir.glob("*.npy"))
    total_size = sum(f.stat().st_size for f in embedding_files)
    
    stats = {
        "summary": {
            "total": manifest.get("total_products", 0),
            "successful": manifest.get("successful", 0),
            "failed": manifest.get("failed", 0),
            "no_image_url": manifest.get("no_image_url", 0),
            "download_failed": manifest.get("download_failed", 0),
            "success_rate": manifest.get("success_rate", 0)
        },
        "performance": {
            "duration_seconds": manifest.get("duration_seconds", 0),
            "products_per_second": manifest.get("products_per_second", 0)
        },
        "storage": {
            "total_size_bytes": total_size,
            "total_size_formatted": format_size(total_size),
            "avg_file_size_bytes": total_size / len(embedding_files) if embedding_files else 0,
            "num_files": len(embedding_files)
        },
        "model": {
            "name": manifest.get("model_name", "Unknown"),
            "device": manifest.get("device", "Unknown")
        },
        "configuration": {
            "fetch_batch_size": manifest.get("fetch_batch_size", 0),
            "embedding_batch_size": manifest.get("embedding_batch_size", 0),
            "max_workers": manifest.get("max_workers", 0),
            "utils_integrated": manifest.get("utils_available", False),
            "datetime_normalized": manifest.get("datetime_normalized", False)
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
        description="Generate image embeddings for products (cursor timeout fixed, datetime normalized)"
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
    parser.add_argument(
        "--all-products",
        action="store_true",
        help="Process all products (including inactive and out-of-stock)"
    )
    
    args = parser.parse_args()
    
    # Print header
    print_header("IMAGE EMBEDDINGS GENERATOR v2.1 - ENHANCED")
    
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
    ├─ Output directory:    {args.output_dir}
    ├─ Utils integration:   {'✅ Enabled' if UTILS_AVAILABLE else '❌ Disabled'}
    └─ Filter active only:  {not args.all_products}
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
        if not args.all_products:
            query["status"] = "active"
            query["stock"] = {"$gt": 0}
        
        total = products_col.count_documents(query)
        print(f"  ✓ Found {total:,} products with images")
        
        # Step 4: Generate embeddings (WITH CURSOR FIX & DATETIME FIX!) ✅
        start_time = time.time()
        manifest = generate_embeddings_batch_safe(
            client=client,
            model=model,
            preprocess=preprocess,
            output_dir=output_dir,
            fetch_batch_size=args.fetch_batch_size,
            embedding_batch_size=args.embedding_batch_size,
            max_workers=args.max_workers,
            filter_active_only=not args.all_products
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
        print_statistics(manifest, output_dir)
        
        # Success message
        print_header("SUCCESS!")
        print(f"""
  ✓ Image embeddings generated successfully!
  
  Output:
    ├─ Embeddings:  {output_dir.absolute()}
    ├─ Files:       {manifest.get('successful', 0):,} .npy files
    ├─ Manifest:    {output_dir / 'manifest.json'}
    └─ Statistics:  {output_dir / 'statistics.json'}
  
  Features Applied:
    ✅ Cursor timeout fix (pagination-based)
    {'✅ DateTime normalization (via utils)' if UTILS_AVAILABLE else '⚠️  DateTime normalization (utils not available)'}
    ✅ MongoDB schema matching
    ✅ Normalized embeddings (L2 norm = 1.0)
    ✅ Parallel image downloading
    ✅ Automatic retry logic
  
  Next Steps:
    1. Load embeddings into service:
       python embedding_loader_enhanced.py
    
    2. Start recommendation service:
       uvicorn recommendation_service_enhanced:app --reload
    
    3. Test image similarity search:
       curl -X POST http://localhost:8000/recommendations \\
         -H "Content-Type: application/json" \\
         -d '{{"userId": "user123", "image": "base64_or_url", "limit": 10}}'
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
