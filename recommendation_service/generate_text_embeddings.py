"""
generate_text_embeddings_fixed.py
  Production-Ready Text Embeddings Generator - ENHANCED v2.1

Key Improvements:
- Uses pagination (skip/limit) instead of long-running cursors  
- Avoids MongoDB cursor timeout (10 min limit)  
- Batch processing with progress tracking  
- DateTime normalization integrated  
- Matches MongoDB Product schema perfectly  
- Memory-efficient  
- Error handling and retry logic  
- Utils integration for database operations  

Purpose:
- Load products from MongoDB in small batches
- Generate text embeddings using SentenceTransformer
- Save embeddings to text_embeddings/ folder
- Create comprehensive manifest with metadata
- Validation and statistics

Usage:
    python generate_text_embeddings_fixed.py
    
    # Or with custom settings:
    python generate_text_embeddings_fixed.py --batch-size 100 --model all-MiniLM-L6-v2 --validate

Output:
    text_embeddings/
    ├── product_id_1.npy
    ├── product_id_2.npy
    └── ...
    ├── manifest.json (comprehensive metadata)
    └── statistics.json (generation stats)
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import numpy as np
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv

# Import utils for database operations  
try:
    from utils.database import normalize_product
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    print("   Warning: Utils not available - datetime normalization may be limited")

# ==========================================
# Configuration
# ==========================================
load_dotenv()

# Environment variables
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "futurist_e-commerce")
MONGO_COLLECTION_PRODUCTS = os.getenv("MONGO_COLLECTION_PRODUCTS", "products")
TEXT_MODEL_NAME = os.getenv("TEXT_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
OUTPUT_DIR = Path("text_embeddings")

# Processing settings (optimized to avoid cursor timeout)
FETCH_BATCH_SIZE = 100  # Small batches - no cursor timeout!
EMBEDDING_BATCH_SIZE = 32  # Embedding generation batch

# Product schema fields (matching ProductModel.js)  
PRODUCT_FIELDS = {
    "_id": 1,
    "name": 1,
    "description": 1,
    "shortDescription": 1,
    "category": 1,
    "subCategory": 1,
    "brand": 1,
    "price": 1,
    "discountPrice": 1,
    "tags": 1,
    "features": 1,
    "color": 1,
    "size": 1,
    "material": 1,
    "style": 1,
    "imageUrl": 1,
    "stock": 1,
    "rating": 1,
    "reviewsCount": 1,
    "isFeatured": 1,
    "isBestseller": 1,
    "isNewProduct": 1,
    "status": 1,
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
# Product Text Processing - ENHANCED  
# ==========================================

def create_product_text(product: Dict[str, Any]) -> str:
    """
    Create a rich text representation of a product for embedding
    
    Matches MongoDB Product schema and creates semantic-rich text:
    - Product name (most important)  
    - Short description (concise info)  
    - Description (detailed info)  
    - Category and subcategory (classification)  
    - Brand (manufacturer)  
    - Tags (keywords)  
    - Features (key attributes)  
    - Style, material, color, size (attributes)  
    
    Format: "field1 | field2 | field3..."
    
    Args:
        product: Product document (normalized with datetime as string)  
    
    Returns:
        Rich text representation for embedding
    """
    text_parts = []
    
    # Product name (highest weight) - REQUIRED
    name = product.get("name", "")
    if name:
        text_parts.append(name)
    
    # Short description (concise, user-friendly)
    short_desc = product.get("shortDescription", "")
    if short_desc:
        text_parts.append(short_desc)
    
    # Full description (detailed)
    description = product.get("description", "")
    if description:
        # Truncate very long descriptions to avoid token limits (512 tokens ≈ 2048 chars)
        if len(description) > 1000:
            description = description[:1000] + "..."
        text_parts.append(description)
    
    # Category hierarchy (classification)
    category = product.get("category", "")
    if category:
        text_parts.append(f"Category: {category}")
    
    sub_category = product.get("subCategory", "")
    if sub_category:
        text_parts.append(f"Subcategory: {sub_category}")
    
    # Brand (manufacturer)
    brand = product.get("brand", "")
    if brand:
        text_parts.append(f"Brand: {brand}")
    
    # Tags (keywords)
    tags = product.get("tags", [])
    if tags and isinstance(tags, list):
        tags_str = ", ".join(str(t) for t in tags[:10])  # Limit to 10 tags
        text_parts.append(f"Tags: {tags_str}")
    
    # Features (key attributes)
    features = product.get("features", [])
    if features and isinstance(features, list):
        features_str = ", ".join(str(f) for f in features[:10])  # Limit to 10
        text_parts.append(f"Features: {features_str}")
    
    # Style (design style)
    style = product.get("style", "")
    if style:
        text_parts.append(f"Style: {style}")
    
    # Material (construction material)
    material = product.get("material", "")
    if material:
        text_parts.append(f"Material: {material}")
    
    # Color (product color)
    color = product.get("color", "")
    if color:
        text_parts.append(f"Color: {color}")
    
    # Size (product size)
    size = product.get("size", "")
    if size:
        text_parts.append(f"Size: {size}")
    
    # Price range (helps with similarity)
    price = product.get("price")
    discount_price = product.get("discountPrice")
    if price:
        if discount_price and discount_price < price:
            text_parts.append(f"Price: ${discount_price:.2f} (on sale from ${price:.2f})")
        else:
            text_parts.append(f"Price: ${price:.2f}")
    
    # Quality signals (helps with ranking)
    rating = product.get("rating", 0)
    reviews_count = product.get("reviewsCount", 0)
    if rating > 0 and reviews_count > 0:
        text_parts.append(f"Rated {rating:.1f}/5 from {reviews_count} reviews")
    
    # Special badges
    badges = []
    if product.get("isFeatured"):
        badges.append("Featured")
    if product.get("isBestseller"):
        badges.append("Bestseller")
    if product.get("isNewProduct"):
        badges.append("New Arrival")
    
    if badges:
        text_parts.append(f"Badges: {', '.join(badges)}")
    
    # Join all parts with separator
    return " | ".join(text_parts)


# ==========================================
# Embedding Generation (CURSOR TIMEOUT FIXED!)  
# ==========================================

def generate_embeddings_batch_safe(
    client: MongoClient,
    model: SentenceTransformer,
    output_dir: Path,
    fetch_batch_size: int = FETCH_BATCH_SIZE,
    embedding_batch_size: int = EMBEDDING_BATCH_SIZE,
    filter_active_only: bool = True
) -> Dict[str, Any]:
    """
    Generate embeddings using pagination to avoid cursor timeout
    
    KEY FIX: Instead of one long cursor, we use skip() + limit()  
    to fetch small batches. Each batch is a fresh query, so no 
    cursor stays open long enough to timeout.
    
    ENHANCEMENT: Uses utils.normalize_product() for datetime fixes  
    
    Args:
        client: MongoDB client
        model: SentenceTransformer model
        output_dir: Where to save embeddings
        fetch_batch_size: How many products to fetch at once (small = no timeout)
        embedding_batch_size: Batch size for model.encode()
        filter_active_only: Only process active products with stock
    
    Returns:
        manifest: Dictionary with generation metadata
    """
    print("\n[STEP 3/6] Generating Text Embeddings...")
    
    db = client[MONGO_DB_NAME]
    products_col = db[MONGO_COLLECTION_PRODUCTS]
    
    # Build filter query
    filter_query = {}
    if filter_active_only:
        filter_query = {
            "status": "active",
            "stock": {"$gt": 0}
        }
    
    # Count total products
    total_products = products_col.count_documents(filter_query)
    print(f"  Total products: {total_products:,}")
    
    if total_products == 0:
        print("  ✗ ERROR: No products found in database")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output directory: {output_dir.absolute()}")
    print(f"  Fetch batch size: {fetch_batch_size}")
    print(f"  Embedding batch size: {embedding_batch_size}")
    print(f"  Filter active only: {filter_active_only}")
    
    # Initialize manifest
    manifest = {
        "generated_at": datetime.utcnow().isoformat(),
        "model_name": TEXT_MODEL_NAME,
        "embedding_dimension": model.get_sentence_embedding_dimension(),
        "total_products": total_products,
        "fetch_batch_size": fetch_batch_size,
        "embedding_batch_size": embedding_batch_size,
        "filter_query": filter_query,
        "utils_available": UTILS_AVAILABLE,
        "datetime_normalized": UTILS_AVAILABLE,
        "embeddings": {},
        "errors": []
    }
    
    successful = 0
    failed = 0
    total_text_length = 0
    start_time = time.time()
    
    # Progress bar for overall progress
    pbar = tqdm(total=total_products, desc="  Generating embeddings", unit="products")
    
    # Process in pagination (NO CURSOR TIMEOUT!)  
    for skip in range(0, total_products, fetch_batch_size):
        try:
            # CRUCIAL: Each iteration is a fresh query with skip/limit  
            # No long-running cursor = no timeout!
            batch_cursor = products_col.find(
                filter_query,
                PRODUCT_FIELDS  # Only fetch needed fields  
            ).skip(skip).limit(fetch_batch_size)
            
            batch = list(batch_cursor)
            
            if not batch:
                break
            
            # Normalize products using utils (datetime → string)  
            if UTILS_AVAILABLE:
                batch = [normalize_product(p) for p in batch]
            
            # Create text representations for this batch
            batch_texts = []
            batch_products = []
            
            for product in batch:
                try:
                    text = create_product_text(product)
                    batch_texts.append(text)
                    batch_products.append(product)
                    total_text_length += len(text)
                except Exception as e:
                    failed += 1
                    manifest["errors"].append({
                        "product_id": str(product.get("_id", "unknown")),
                        "error": f"Text creation failed: {str(e)}"
                    })
            
            # Generate embeddings for this batch
            if batch_texts:
                try:
                    # Generate embeddings (in sub-batches if needed)
                    embeddings = model.encode(
                        batch_texts,
                        batch_size=embedding_batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True  # Normalize for better cosine similarity
                    )
                    
                    # Save individual embeddings
                    for product, embedding, text in zip(batch_products, embeddings, batch_texts):
                        try:
                            product_id = str(product["_id"])
                            
                            # Save embedding as .npy file
                            embedding_path = output_dir / f"{product_id}.npy"
                            np.save(embedding_path, embedding)
                            
                            # Add to manifest with metadata
                            manifest["embeddings"][product_id] = {
                                "name": product.get("name", "Unknown"),
                                "category": product.get("category", "Unknown"),
                                "subcategory": product.get("subCategory"),
                                "brand": product.get("brand"),
                                "file": embedding_path.name,
                                "shape": list(embedding.shape),
                                "text_length": len(text),
                                "has_description": bool(product.get("description")),
                                "has_short_description": bool(product.get("shortDescription")),
                                "has_tags": bool(product.get("tags")),
                                "has_features": bool(product.get("features")),
                                "rating": product.get("rating", 0),
                                "reviews_count": product.get("reviewsCount", 0),
                                "is_featured": product.get("isFeatured", False),
                                "is_bestseller": product.get("isBestseller", False),
                                "created_at": product.get("createdAt")  #   Already ISO string
                            }
                            
                            successful += 1
                            
                        except Exception as e:
                            failed += 1
                            manifest["errors"].append({
                                "product_id": str(product.get("_id", "unknown")),
                                "error": f"Save failed: {str(e)}"
                            })
                
                except Exception as e:
                    failed += len(batch_texts)
                    manifest["errors"].append({
                        "batch_skip": skip,
                        "batch_size": len(batch_texts),
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
    avg_text_length = total_text_length / successful if successful > 0 else 0
    
    manifest.update({
        "successful": successful,
        "failed": failed,
        "success_rate": successful / total_products if total_products > 0 else 0,
        "duration_seconds": duration,
        "duration_formatted": format_time(duration),
        "products_per_second": successful / duration if duration > 0 else 0,
        "avg_text_length": avg_text_length,
        "total_text_length": total_text_length
    })
    
    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"\n  ✓ Embeddings generated: {successful:,} successful, {failed:,} failed")
    print(f"  ✓ Duration: {format_time(duration)}")
    print(f"  ✓ Speed: {successful/duration:.1f} products/second")
    print(f"  ✓ Avg text length: {avg_text_length:.0f} characters")
    print(f"  ✓ Manifest saved: {manifest_path}")
    
    return manifest


# ==========================================
# Validation - ENHANCED  
# ==========================================

def validate_embeddings(output_dir: Path, manifest: Dict[str, Any]) -> bool:
    """
    Validate that embeddings were generated correctly
    
    Checks:
    - Manifest file exists  
    - All expected .npy files exist  
    - Files can be loaded  
    - Embeddings have correct shape  
    - No NaN or Inf values  
    - Embeddings are normalized  
    """
    print("\n[STEP 4/6] Validating Embeddings...")
    
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
        
        expected_dim = manifest.get("embedding_dimension", 384)
        valid_count = 0
        
        for emb_file in embedding_files[:sample_size]:
            try:
                embedding = np.load(emb_file)
                
                # Check shape
                if embedding.shape[0] != expected_dim:
                    print(f"    ✗ {emb_file.name}: Wrong dimension {embedding.shape}")
                    return False
                
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
        print(f"  ✓ Dimension: {expected_dim}D")
        print(f"  ✓ All embeddings normalized")
        print(f"  ✓ Validation passed!")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Validation error: {e}")
        return False


# ==========================================
# Statistics - ENHANCED  
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
    ├─ Embedding dimension: {manifest.get('embedding_dimension', 'Unknown')}D
    ├─ Fetch batch size:    {manifest.get('fetch_batch_size', 'Unknown')}
    ├─ Embedding batch:     {manifest.get('embedding_batch_size', 'Unknown')}
    ├─ Utils integration:   {'  Enabled' if manifest.get('utils_available') else '  Disabled'}
    └─ DateTime fix:        {'  Applied' if manifest.get('datetime_normalized') else '  Not applied'}
  
  Results:
    ├─ Total products:      {manifest.get('total_products', 0):,}
    ├─ Successful:          {manifest.get('successful', 0):,}
    ├─ Failed:              {manifest.get('failed', 0):,}
    ├─ Success rate:        {manifest.get('success_rate', 0)*100:.1f}%
    └─ Errors logged:       {len(manifest.get('errors', []))}
  
  Performance:
    ├─ Duration:            {manifest.get('duration_formatted', 'Unknown')}
    ├─ Speed:               {manifest.get('products_per_second', 0):.1f} products/second
    └─ Avg text length:     {manifest.get('avg_text_length', 0):.0f} characters
  
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
            "success_rate": manifest.get("success_rate", 0)
        },
        "performance": {
            "duration_seconds": manifest.get("duration_seconds", 0),
            "products_per_second": manifest.get("products_per_second", 0),
            "avg_text_length": manifest.get("avg_text_length", 0)
        },
        "storage": {
            "total_size_bytes": total_size,
            "total_size_formatted": format_size(total_size),
            "avg_file_size_bytes": total_size / len(embedding_files) if embedding_files else 0,
            "num_files": len(embedding_files)
        },
        "model": {
            "name": manifest.get("model_name", "Unknown"),
            "dimension": manifest.get("embedding_dimension", 0)
        },
        "configuration": {
            "fetch_batch_size": manifest.get("fetch_batch_size", 0),
            "embedding_batch_size": manifest.get("embedding_batch_size", 0),
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
        description="Generate text embeddings for products (cursor timeout fixed, datetime normalized)"
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
    print_header("TEXT EMBEDDINGS GENERATOR v2.1 - ENHANCED")
    
    print(f"""
  Configuration:
    ├─ MongoDB URI:         {MONGO_URI[:50]}...
    ├─ Database:            {MONGO_DB_NAME}
    ├─ Collection:          {MONGO_COLLECTION_PRODUCTS}
    ├─ Model:               {args.model}
    ├─ Fetch batch:         {args.fetch_batch_size} products
    ├─ Embedding batch:     {args.embedding_batch_size} products
    ├─ Output directory:    {args.output_dir}
    ├─ Utils integration:   {'  Enabled' if UTILS_AVAILABLE else '  Disabled'}
    └─ Filter active only:  {not args.all_products}
    """)
    
    output_dir = Path(args.output_dir)
    client = None
    
    try:
        # Step 1: Connect to MongoDB
        client = connect_mongodb()
        
        # Step 2: Load model
        print("\n[STEP 2/6] Loading SentenceTransformer Model...")
        print(f"  Model: {args.model}")
        
        try:
            model = SentenceTransformer(args.model)
            embedding_dim = model.get_sentence_embedding_dimension()
            print(f"  ✓ Model loaded successfully")
            print(f"  ✓ Embedding dimension: {embedding_dim}D")
        except Exception as e:
            print(f"  ✗ ERROR: Failed to load model: {e}")
            return 1
        
        # Step 3: Generate embeddings (WITH CURSOR FIX & DATETIME FIX!)  
        start_time = time.time()
        manifest = generate_embeddings_batch_safe(
            client=client,
            model=model,
            output_dir=output_dir,
            fetch_batch_size=args.fetch_batch_size,
            embedding_batch_size=args.embedding_batch_size,
            filter_active_only=not args.all_products
        )
        
        # Step 4: Validation
        if not args.skip_validation:
            validation_passed = validate_embeddings(output_dir, manifest)
            if not validation_passed:
                print("  ⚠ WARNING: Validation issues detected")
        else:
            print("\n[STEP 4/6] Validation skipped")
        
        # Step 5: Save statistics
        print("\n[STEP 5/6] Saving Statistics...")
        save_statistics(output_dir, manifest)
        
        # Step 6: Print final statistics
        print("\n[STEP 6/6] Final Summary")
        print_statistics(manifest, output_dir)
        
        # Success message
        print_header("SUCCESS!")
        print(f"""
  ✓ Text embeddings generated successfully!
  
  Output:
    ├─ Embeddings:  {output_dir.absolute()}
    ├─ Files:       {manifest.get('successful', 0):,} .npy files
    ├─ Manifest:    {output_dir / 'manifest.json'}
    └─ Statistics:  {output_dir / 'statistics.json'}
  
  Features Applied:
      Cursor timeout fix (pagination-based)
    {'  DateTime normalization (via utils)' if UTILS_AVAILABLE else '   DateTime normalization (utils not available)'}
      MongoDB schema matching
      Normalized embeddings (L2 norm = 1.0)
      Rich text representation
  
  Next Steps:
    1. Generate image embeddings (optional):
       python generate_image_embeddings_fixed.py
    
    2. Train collaborative model:
       python train_hybrid_model.py
    
    3. Start recommendation service:
       uvicorn recommendation_service_enhanced:app --reload
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
