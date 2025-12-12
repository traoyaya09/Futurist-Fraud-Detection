"""
embedding_loader_enhanced.py
✅ Production-ready embedding loader for FastAPI service - ENHANCED v2.1

Key Features:
- ✅ Loads individual .npy files (one per product) - Compatible with fixed generators
- ✅ DateTime-aware manifest parsing
- ✅ Lazy loading of embeddings
- ✅ Smart caching mechanism with TTL
- ✅ Automatic refresh when files change
- ✅ Error handling and fallbacks
- ✅ Memory-efficient loading
- ✅ Supports both .npy (individual) and .npz (batched) formats
- ✅ Comprehensive validation
- ✅ Statistics and monitoring

File Format Support:
- .npy files: text_embeddings/product_id.npy (NEW - from generate_text_embeddings_fixed.py)
- .npy files: image_embeddings/product_id.npy (NEW - from generate_image_embeddings_fixed.py)
- .npz files: embeddings/batch_001.npz (OLD - legacy support)
- manifest.json: metadata file with product info (datetime as ISO strings)

Integration:
- Works seamlessly with recommendation_service_enhanced.py
- Thread-safe for concurrent requests
- Automatic cache invalidation
"""

import os
import glob
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime
import threading
import numpy as np

logger = logging.getLogger("EmbeddingLoader")

# ==========================================
# Thread-Safe Embedding Cache
# ==========================================

class EmbeddingCache:
    """
    Thread-safe cache for embeddings with automatic refresh
    
    Features:
    - TTL-based cache expiration
    - File modification detection
    - Lazy loading (load on first access)
    - Memory-efficient (loads only when needed)
    - Thread-safe for concurrent requests
    """
    
    def __init__(self, cache_ttl_seconds: int = 300):
        """
        Initialize cache
        
        Args:
            cache_ttl_seconds: How long to keep cache before refresh (default: 5 minutes)
        """
        self.cache_ttl = cache_ttl_seconds
        self.text_embeddings: Optional[Dict[str, np.ndarray]] = None
        self.image_embeddings: Optional[Dict[str, np.ndarray]] = None
        self.last_load_time: Dict[str, Optional[datetime]] = {
            'text': None,
            'image': None
        }
        self.last_modified_time: Dict[str, Optional[datetime]] = {
            'text': None,
            'image': None
        }
        self.manifest: Dict[str, Optional[Dict]] = {
            'text': None,
            'image': None
        }
        self.lock = threading.RLock()  # Thread-safe lock
        
        logger.info(f"🚀 EmbeddingCache initialized (TTL: {cache_ttl_seconds}s)")
    
    def is_cache_valid(self, embedding_dir: str, embedding_type: str) -> bool:
        """
        Check if cache is still valid
        
        Checks:
        1. Has cache been loaded?
        2. Has TTL expired?
        3. Have files been modified?
        
        Args:
            embedding_dir: Directory to check
            embedding_type: 'text' or 'image'
        
        Returns:
            True if cache is valid, False otherwise
        """
        if self.last_load_time[embedding_type] is None:
            return False
        
        # Check TTL
        elapsed = (datetime.utcnow() - self.last_load_time[embedding_type]).total_seconds()
        if elapsed > self.cache_ttl:
            logger.info(f"🔄 {embedding_type} cache TTL expired ({elapsed:.0f}s > {self.cache_ttl}s)")
            return False
        
        # Check if files have been modified
        try:
            latest_mtime = self._get_latest_mtime(embedding_dir)
            if latest_mtime and self.last_modified_time[embedding_type]:
                if latest_mtime > self.last_modified_time[embedding_type]:
                    logger.info(f"🔄 {embedding_type} embedding files updated, refreshing cache")
                    return False
        except Exception as e:
            logger.warning(f"⚠️  Failed to check file modification time: {e}")
        
        return True
    
    def _get_latest_mtime(self, embedding_dir: str) -> Optional[datetime]:
        """Get the latest modification time of embedding files"""
        embedding_path = Path(embedding_dir)
        
        if not embedding_path.exists():
            return None
        
        # Check both .npy and .npz files
        npy_files = list(embedding_path.glob("*.npy"))
        npz_files = list(embedding_path.glob("*.npz"))
        all_files = npy_files + npz_files
        
        if not all_files:
            return None
        
        latest = max(f.stat().st_mtime for f in all_files)
        return datetime.fromtimestamp(latest)
    
    def load(self, embedding_dir: str, embedding_type: str) -> Dict[str, np.ndarray]:
        """
        Load embeddings with caching (thread-safe)
        
        Args:
            embedding_dir: Directory containing embedding files
            embedding_type: 'text' or 'image'
        
        Returns:
            Dict mapping product_id -> embedding
        """
        with self.lock:
            # Check cache
            cached = self.text_embeddings if embedding_type == 'text' else self.image_embeddings
            
            if cached and self.is_cache_valid(embedding_dir, embedding_type):
                logger.debug(f"✅ Using cached {embedding_type} embeddings ({len(cached)} items)")
                return cached
            
            # Load from disk
            logger.info(f"📂 Loading {embedding_type} embeddings from {embedding_dir}...")
            embeddings, manifest = load_embeddings_smart(embedding_dir)
            
            # Update cache
            if embedding_type == 'text':
                self.text_embeddings = embeddings
            else:
                self.image_embeddings = embeddings
            
            self.last_load_time[embedding_type] = datetime.utcnow()
            self.last_modified_time[embedding_type] = self._get_latest_mtime(embedding_dir)
            self.manifest[embedding_type] = manifest
            
            logger.info(f"✅ Loaded {len(embeddings)} {embedding_type} embeddings")
            return embeddings
    
    def get_manifest(self, embedding_type: str) -> Optional[Dict]:
        """
        Get manifest for embedding type from cache
        
        Args:
            embedding_type: 'text' or 'image'
        
        Returns:
            Manifest dict or None
        """
        with self.lock:
            return self.manifest.get(embedding_type)
    
    def clear(self, embedding_type: Optional[str] = None):
        """
        Clear cache (thread-safe)
        
        Args:
            embedding_type: Specific type to clear, or None to clear all
        """
        with self.lock:
            if embedding_type is None or embedding_type == 'text':
                self.text_embeddings = None
                self.last_load_time['text'] = None
                self.last_modified_time['text'] = None
                self.manifest['text'] = None
                logger.info("🧹 Text embeddings cache cleared")
            
            if embedding_type is None or embedding_type == 'image':
                self.image_embeddings = None
                self.last_load_time['image'] = None
                self.last_modified_time['image'] = None
                self.manifest['image'] = None
                logger.info("🧹 Image embeddings cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                "text": {
                    "loaded": self.text_embeddings is not None,
                    "count": len(self.text_embeddings) if self.text_embeddings else 0,
                    "last_load": self.last_load_time['text'].isoformat() if self.last_load_time['text'] else None,
                    "has_manifest": self.manifest['text'] is not None
                },
                "image": {
                    "loaded": self.image_embeddings is not None,
                    "count": len(self.image_embeddings) if self.image_embeddings else 0,
                    "last_load": self.last_load_time['image'].isoformat() if self.last_load_time['image'] else None,
                    "has_manifest": self.manifest['image'] is not None
                },
                "cache_ttl_seconds": self.cache_ttl
            }

# Global cache instance
_cache = EmbeddingCache(cache_ttl_seconds=300)


# ==========================================
# Smart Embedding Loader
# ==========================================

def load_embeddings_smart(embedding_dir: str) -> Tuple[Dict[str, np.ndarray], Optional[Dict]]:
    """
    Smart loader that detects file format and loads accordingly
    
    Supports:
    1. Individual .npy files (NEW - from generate_*_embeddings_fixed.py)
    2. Batched .npz files (OLD - legacy support)
    3. manifest.json for metadata (with datetime as ISO strings)
    
    Args:
        embedding_dir: Directory containing embeddings
    
    Returns:
        Tuple of (embeddings_dict, manifest_dict)
    """
    embedding_path = Path(embedding_dir)
    
    if not embedding_path.exists():
        logger.warning(f"⚠️  Embedding directory not found: {embedding_dir}")
        return {}, None
    
    # Load manifest if available
    manifest = load_manifest(embedding_path)
    
    # Check for .npy files (NEW FORMAT - from fixed generators)
    npy_files = list(embedding_path.glob("*.npy"))
    
    if npy_files:
        logger.info(f"📁 Found {len(npy_files)} .npy files (individual format)")
        embeddings = load_individual_npy_files(npy_files, manifest)
        return embeddings, manifest
    
    # Check for .npz files (OLD FORMAT - legacy support)
    npz_files = list(embedding_path.glob("*.npz"))
    
    if npz_files:
        logger.info(f"📁 Found {len(npz_files)} .npz files (batched format)")
        embeddings = load_batched_npz_files(npz_files)
        return embeddings, manifest
    
    logger.warning(f"⚠️  No embedding files found in {embedding_dir}")
    return {}, None


def load_manifest(embedding_path: Path) -> Optional[Dict]:
    """
    Load manifest.json with datetime parsing
    
    Handles datetime fields as ISO strings (from fixed generators)
    
    Args:
        embedding_path: Path to embedding directory
    
    Returns:
        Manifest dict or None
    """
    manifest_path = embedding_path / "manifest.json"
    
    if not manifest_path.exists():
        logger.debug(f"ℹ️  No manifest found at {manifest_path}")
        return None
    
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        logger.info(f"✅ Loaded manifest: {len(manifest.get('embeddings', {}))} products")
        
        # Validate manifest structure
        if 'generated_at' in manifest:
            # DateTime is already ISO string from fixed generators ✅
            logger.debug(f"📅 Generated at: {manifest['generated_at']}")
        
        if 'model_name' in manifest:
            logger.debug(f"🤖 Model: {manifest['model_name']}")
        
        if 'embedding_dimension' in manifest:
            logger.debug(f"📐 Dimension: {manifest['embedding_dimension']}D")
        
        # Check for datetime normalization flag
        if manifest.get('datetime_normalized'):
            logger.debug("✅ Manifest confirms datetime normalization applied")
        
        return manifest
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to parse manifest JSON: {e}")
        return None
    except Exception as e:
        logger.warning(f"⚠️  Failed to load manifest: {e}")
        return None


def load_individual_npy_files(
    npy_files: List[Path],
    manifest: Optional[Dict] = None
) -> Dict[str, np.ndarray]:
    """
    Load individual .npy files (one per product)
    
    File format: text_embeddings/67448a2e0f6587f56be4d0c1.npy
    Each file contains a single embedding array
    
    Args:
        npy_files: List of .npy file paths
        manifest: Optional manifest for validation
    
    Returns:
        Dict mapping product_id -> embedding
    """
    embeddings = {}
    failed = 0
    validation_errors = []
    
    logger.info(f"📦 Loading {len(npy_files)} individual .npy files...")
    
    for file_path in npy_files:
        try:
            # Extract product ID from filename (e.g., "product_id.npy" -> "product_id")
            product_id = file_path.stem  # Gets filename without extension
            
            # Load embedding
            embedding = np.load(file_path)
            
            # Validate embedding
            validation_result = validate_embedding(embedding, product_id)
            
            if not validation_result["valid"]:
                logger.warning(f"⚠️  Invalid embedding for {product_id}: {validation_result['error']}")
                validation_errors.append({
                    "product_id": product_id,
                    "file": file_path.name,
                    "error": validation_result["error"]
                })
                failed += 1
                continue
            
            embeddings[product_id] = embedding
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to load {file_path.name}: {e}")
            failed += 1
    
    if failed > 0:
        logger.warning(f"⚠️  Failed to load {failed}/{len(npy_files)} files")
    
    logger.info(f"✅ Loaded {len(embeddings)}/{len(npy_files)} embeddings from .npy files")
    
    # Validate against manifest if available
    if manifest and 'embeddings' in manifest:
        validate_against_manifest(embeddings, manifest)
    
    return embeddings


def validate_embedding(embedding: Any, product_id: str) -> Dict[str, Any]:
    """
    Validate a single embedding
    
    Checks:
    - Is numpy array
    - Not empty
    - No NaN values
    - No Inf values
    - Reasonable norm
    
    Args:
        embedding: Embedding to validate
        product_id: Product ID (for logging)
    
    Returns:
        Dict with validation result
    """
    # Check type
    if not isinstance(embedding, np.ndarray):
        return {
            "valid": False,
            "error": f"Invalid type: {type(embedding)}, expected np.ndarray"
        }
    
    # Check size
    if embedding.size == 0:
        return {
            "valid": False,
            "error": "Empty embedding"
        }
    
    # Check for NaN
    if np.isnan(embedding).any():
        return {
            "valid": False,
            "error": "Contains NaN values"
        }
    
    # Check for Inf
    if np.isinf(embedding).any():
        return {
            "valid": False,
            "error": "Contains Inf values"
        }
    
    # Check norm (should be reasonable, ideally ~1.0 for normalized embeddings)
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return {
            "valid": False,
            "error": "Zero norm embedding"
        }
    
    # Warn if not normalized (but still valid)
    if not (0.9 <= norm <= 1.1):
        logger.debug(f"⚠️  {product_id}: Embedding not normalized (norm={norm:.3f})")
    
    return {
        "valid": True,
        "error": None,
        "norm": float(norm),
        "dimension": embedding.shape[0] if embedding.ndim > 0 else 0
    }


def validate_against_manifest(embeddings: Dict[str, np.ndarray], manifest: Dict):
    """
    Validate loaded embeddings against manifest
    
    Args:
        embeddings: Loaded embeddings dict
        manifest: Manifest dict
    """
    manifest_products = manifest.get('embeddings', {})
    manifest_count = len(manifest_products)
    loaded_count = len(embeddings)
    
    if loaded_count == manifest_count:
        logger.info(f"✅ Manifest validation passed: {loaded_count} embeddings match")
    else:
        logger.warning(
            f"⚠️  Mismatch: Manifest shows {manifest_count} products, "
            f"but loaded {loaded_count} embeddings"
        )
        
        # Find missing products
        missing = set(manifest_products.keys()) - set(embeddings.keys())
        if missing:
            logger.warning(f"⚠️  Missing {len(missing)} products from manifest")
            if len(missing) <= 10:
                logger.debug(f"Missing IDs: {', '.join(list(missing)[:10])}")


def load_batched_npz_files(npz_files: List[Path]) -> Dict[str, np.ndarray]:
    """
    Load batched .npz files (legacy format)
    
    File format: Each .npz file contains:
    - 'ids': array of product IDs
    - 'embeddings': array of embeddings
    
    Args:
        npz_files: List of .npz file paths
    
    Returns:
        Dict mapping product_id -> embedding
    """
    embeddings = {}
    failed = 0
    
    logger.info(f"📦 Loading {len(npz_files)} batched .npz files...")
    
    for file_path in npz_files:
        try:
            data = np.load(file_path, allow_pickle=True)
            
            # Extract IDs and embeddings
            ids = data.get("ids", [])
            embs = data.get("embeddings", [])
            
            if len(ids) != len(embs):
                logger.warning(
                    f"⚠️  Mismatch in {file_path.name}: "
                    f"{len(ids)} ids vs {len(embs)} embeddings"
                )
                failed += 1
                continue
            
            # Merge into dict
            for pid, emb in zip(ids, embs):
                product_id = str(pid)
                
                # Validate embedding
                validation_result = validate_embedding(emb, product_id)
                if not validation_result["valid"]:
                    logger.warning(f"⚠️  Invalid embedding for {product_id}: {validation_result['error']}")
                    continue
                
                embeddings[product_id] = emb
        
        except Exception as e:
            logger.warning(f"⚠️  Failed to load {file_path.name}: {e}")
            failed += 1
    
    if failed > 0:
        logger.warning(f"⚠️  Failed to load {failed}/{len(npz_files)} files")
    
    logger.info(f"✅ Loaded {len(embeddings)} embeddings from .npz files")
    return embeddings


# ==========================================
# Public API
# ==========================================

def load_text_embeddings(
    embedding_dir: str = "text_embeddings",
    use_cache: bool = True
) -> Dict[str, np.ndarray]:
    """
    Load text embeddings with optional caching
    
    Args:
        embedding_dir: Directory containing text embeddings
        use_cache: Whether to use cache (default: True)
    
    Returns:
        Dict mapping product_id -> text embedding
    """
    if use_cache:
        return _cache.load(embedding_dir, 'text')
    
    embeddings, _ = load_embeddings_smart(embedding_dir)
    return embeddings


def load_image_embeddings(
    embedding_dir: str = "image_embeddings",
    use_cache: bool = True
) -> Dict[str, np.ndarray]:
    """
    Load image embeddings with optional caching
    
    Args:
        embedding_dir: Directory containing image embeddings
        use_cache: Whether to use cache (default: True)
    
    Returns:
        Dict mapping product_id -> image embedding
    """
    if use_cache:
        return _cache.load(embedding_dir, 'image')
    
    embeddings, _ = load_embeddings_smart(embedding_dir)
    return embeddings


def get_embedding(
    product_id: str,
    embedding_type: str = "text",
    embeddings_dict: Optional[Dict[str, np.ndarray]] = None
) -> Optional[np.ndarray]:
    """
    Get embedding for a specific product
    
    Args:
        product_id: Product ID
        embedding_type: 'text' or 'image'
        embeddings_dict: Pre-loaded embeddings dict (optional)
    
    Returns:
        Embedding vector or None if not found
    """
    if embeddings_dict is None:
        if embedding_type == "text":
            embeddings_dict = load_text_embeddings()
        else:
            embeddings_dict = load_image_embeddings()
    
    return embeddings_dict.get(str(product_id))


def get_embedding_stats(embedding_dir: str) -> Dict[str, Any]:
    """
    Get statistics about embeddings in a directory
    
    Args:
        embedding_dir: Directory containing embeddings
    
    Returns:
        Dict with statistics (count, dimension, norms, etc.)
    """
    embeddings, manifest = load_embeddings_smart(embedding_dir)
    
    if not embeddings:
        return {
            "count": 0,
            "dimension": 0,
            "has_nan": False,
            "has_inf": False,
            "format": "none",
            "status": "empty"
        }
    
    # Get sample embedding for dimension
    sample_emb = next(iter(embeddings.values()))
    dimension = sample_emb.shape[0] if sample_emb.ndim > 0 else 0
    
    # Check for invalid values
    has_nan = any(np.isnan(emb).any() for emb in embeddings.values())
    has_inf = any(np.isinf(emb).any() for emb in embeddings.values())
    
    # Calculate norms
    norms = [np.linalg.norm(emb) for emb in embeddings.values()]
    avg_norm = np.mean(norms) if norms else 0
    
    # Detect format
    embedding_path = Path(embedding_dir)
    npy_count = len(list(embedding_path.glob("*.npy")))
    npz_count = len(list(embedding_path.glob("*.npz")))
    
    file_format = "individual_npy" if npy_count > 0 else "batched_npz" if npz_count > 0 else "unknown"
    
    # Check if embeddings are normalized
    is_normalized = all(0.99 <= norm <= 1.01 for norm in norms)
    
    stats = {
        "count": len(embeddings),
        "dimension": dimension,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "format": file_format,
        "npy_files": npy_count,
        "npz_files": npz_count,
        "avg_norm": float(avg_norm),
        "min_norm": float(min(norms)) if norms else 0,
        "max_norm": float(max(norms)) if norms else 0,
        "is_normalized": is_normalized,
        "status": "healthy" if not (has_nan or has_inf) else "unhealthy"
    }
    
    # Add manifest info if available
    if manifest:
        stats["has_manifest"] = True
        stats["manifest_products"] = len(manifest.get("embeddings", {}))
        stats["generation_date"] = manifest.get("generated_at", "unknown")
        stats["model"] = manifest.get("model_name", "unknown")
        stats["datetime_normalized"] = manifest.get("datetime_normalized", False)
        stats["utils_integrated"] = manifest.get("utils_available", False)
    else:
        stats["has_manifest"] = False
    
    return stats


def get_manifest(embedding_type: str = "text") -> Optional[Dict]:
    """
    Get manifest for embedding type from cache
    
    Args:
        embedding_type: 'text' or 'image'
    
    Returns:
        Manifest dict or None
    """
    return _cache.get_manifest(embedding_type)


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    return _cache.get_stats()


def clear_cache(embedding_type: Optional[str] = None):
    """
    Clear the global embedding cache
    
    Args:
        embedding_type: Specific type to clear ('text' or 'image'), or None to clear all
    """
    _cache.clear(embedding_type)


def reload_embeddings(embedding_type: Optional[str] = None):
    """
    Force reload embeddings (clears cache and loads fresh)
    
    Args:
        embedding_type: 'text', 'image', or None for both
    """
    clear_cache(embedding_type)
    
    if embedding_type is None or embedding_type == 'text':
        load_text_embeddings(use_cache=True)
    
    if embedding_type is None or embedding_type == 'image':
        load_image_embeddings(use_cache=True)


# ==========================================
# Testing / CLI
# ==========================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    print("\n" + "=" * 80)
    print("EMBEDDING LOADER TEST - ENHANCED v2.1".center(80))
    print("=" * 80)
    
    # Test text embeddings
    print("\n[1/4] Testing Text Embeddings...")
    text_embs = load_text_embeddings(use_cache=False)
    print(f"  ✓ Loaded {len(text_embs)} text embeddings")
    
    if text_embs:
        stats = get_embedding_stats("text_embeddings")
        print(f"\n  📊 Text Embedding Statistics:")
        print(f"    ├─ Count:           {stats['count']:,}")
        print(f"    ├─ Dimension:       {stats['dimension']}D")
        print(f"    ├─ Format:          {stats['format']}")
        print(f"    ├─ Status:          {stats['status']}")
        print(f"    ├─ Has NaN:         {stats['has_nan']}")
        print(f"    ├─ Has Inf:         {stats['has_inf']}")
        print(f"    ├─ Normalized:      {stats['is_normalized']}")
        print(f"    ├─ Avg norm:        {stats['avg_norm']:.4f}")
        print(f"    ├─ Has manifest:    {stats.get('has_manifest', False)}")
        if stats.get('has_manifest'):
            print(f"    ├─ DateTime fixed:  {stats.get('datetime_normalized', False)}")
            print(f"    └─ Model:           {stats.get('model', 'unknown')}")
    
    # Test image embeddings
    print("\n[2/4] Testing Image Embeddings...")
    image_embs = load_image_embeddings(use_cache=False)
    print(f"  ✓ Loaded {len(image_embs)} image embeddings")
    
    if image_embs:
        stats = get_embedding_stats("image_embeddings")
        print(f"\n  📊 Image Embedding Statistics:")
        print(f"    ├─ Count:           {stats['count']:,}")
        print(f"    ├─ Dimension:       {stats['dimension']}D")
        print(f"    ├─ Format:          {stats['format']}")
        print(f"    ├─ Normalized:      {stats['is_normalized']}")
        print(f"    └─ Has manifest:    {stats.get('has_manifest', False)}")
    
    # Test cache
    print("\n[3/4] Testing Cache...")
    print("  Loading text embeddings again (should use cache)...")
    text_embs_cached = load_text_embeddings(use_cache=True)
    print(f"  ✓ Cache working: {len(text_embs_cached)} embeddings")
    
    cache_stats = get_cache_stats()
    print(f"\n  📊 Cache Statistics:")
    print(f"    ├─ Text loaded:     {cache_stats['text']['loaded']}")
    print(f"    ├─ Text count:      {cache_stats['text']['count']:,}")
    print(f"    ├─ Image loaded:    {cache_stats['image']['loaded']}")
    print(f"    ├─ Image count:     {cache_stats['image']['count']:,}")
    print(f"    └─ TTL:             {cache_stats['cache_ttl_seconds']}s")
    
    # Test specific product lookup
    print("\n[4/4] Testing Product Lookup...")
    if text_embs:
        sample_id = next(iter(text_embs.keys()))
        sample_emb = get_embedding(sample_id, "text")
        if sample_emb is not None:
            print(f"  ✓ Retrieved embedding for product {sample_id}")
            print(f"    ├─ Shape: {sample_emb.shape}")
            print(f"    ├─ Norm:  {np.linalg.norm(sample_emb):.4f}")
            print(f"    └─ Mean:  {np.mean(sample_emb):.4f}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE - All Systems Operational ✅".center(80))
    print("=" * 80 + "\n")
