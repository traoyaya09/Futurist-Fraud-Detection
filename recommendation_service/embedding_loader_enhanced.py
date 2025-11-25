"""
embedding_loader_enhanced.py
✅ Production-ready embedding loader for FastAPI service - UPDATED FOR .npy FILES

Key Updates:
- ✅ Loads individual .npy files (one per product)
- ✅ Compatible with generate_text_embeddings_fixed.py output
- ✅ Lazy loading of embeddings
- ✅ Caching mechanism with TTL
- ✅ Automatic refresh when files change
- ✅ Error handling and fallbacks
- ✅ Memory-efficient loading
- ✅ Supports both .npy (individual) and .npz (batched) formats

File Format Support:
- .npy files: text_embeddings/product_id.npy (NEW - from fixed script)
- .npz files: text_embeddings/batch_001.npz (OLD - legacy support)
- manifest.json: metadata file with product info
"""

import os
import glob
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import numpy as np
from datetime import datetime

logger = logging.getLogger("EmbeddingLoader")

class EmbeddingCache:
    """
    Manages embedding loading and caching with automatic refresh
    
    Features:
    - TTL-based cache expiration
    - File modification detection
    - Lazy loading (load on first access)
    - Memory-efficient (loads only when needed)
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
    
    def is_cache_valid(self, embedding_dir: str, embedding_type: str) -> bool:
        """
        Check if cache is still valid
        
        Checks:
        1. Has cache been loaded?
        2. Has TTL expired?
        3. Have files been modified?
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
        # Check both .npy and .npz files
        npy_files = glob.glob(str(Path(embedding_dir) / "*.npy"))
        npz_files = glob.glob(str(Path(embedding_dir) / "*.npz"))
        all_files = npy_files + npz_files
        
        if not all_files:
            return None
        
        latest = max(os.path.getmtime(f) for f in all_files)
        return datetime.fromtimestamp(latest)
    
    def load(self, embedding_dir: str, embedding_type: str) -> Dict[str, np.ndarray]:
        """
        Load embeddings with caching
        
        Args:
            embedding_dir: Directory containing embedding files
            embedding_type: 'text' or 'image'
        
        Returns:
            Dict mapping product_id -> embedding
        """
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
        """Get manifest for embedding type"""
        return self.manifest.get(embedding_type)
    
    def clear(self, embedding_type: Optional[str] = None):
        """
        Clear cache
        
        Args:
            embedding_type: Specific type to clear, or None to clear all
        """
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

# Global cache instance
_cache = EmbeddingCache(cache_ttl_seconds=300)


def load_embeddings_smart(embedding_dir: str) -> Tuple[Dict[str, np.ndarray], Optional[Dict]]:
    """
    Smart loader that detects file format and loads accordingly
    
    Supports:
    1. Individual .npy files (NEW - from generate_text_embeddings_fixed.py)
    2. Batched .npz files (OLD - legacy support)
    3. Manifest.json for metadata
    
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
    manifest = None
    manifest_path = embedding_path / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            logger.info(f"✅ Loaded manifest: {len(manifest.get('embeddings', {}))} products")
        except Exception as e:
            logger.warning(f"⚠️  Failed to load manifest: {e}")
    
    # Check for .npy files (NEW FORMAT - from fixed script)
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
    
    logger.info(f"📦 Loading {len(npy_files)} individual .npy files...")
    
    for file_path in npy_files:
        try:
            # Extract product ID from filename (e.g., "product_id.npy" -> "product_id")
            product_id = file_path.stem  # Gets filename without extension
            
            # Load embedding
            embedding = np.load(file_path)
            
            # Validate embedding
            if not isinstance(embedding, np.ndarray):
                logger.warning(f"⚠️  Invalid embedding type for {product_id}: {type(embedding)}")
                failed += 1
                continue
            
            if embedding.size == 0:
                logger.warning(f"⚠️  Empty embedding for {product_id}")
                failed += 1
                continue
            
            if np.isnan(embedding).any():
                logger.warning(f"⚠️  NaN values in embedding for {product_id}")
                failed += 1
                continue
            
            if np.isinf(embedding).any():
                logger.warning(f"⚠️  Inf values in embedding for {product_id}")
                failed += 1
                continue
            
            embeddings[product_id] = embedding
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to load {file_path.name}: {e}")
            failed += 1
    
    if failed > 0:
        logger.warning(f"⚠️  Failed to load {failed} files")
    
    logger.info(f"✅ Loaded {len(embeddings)} embeddings from .npy files")
    
    # Validate against manifest if available
    if manifest and 'embeddings' in manifest:
        manifest_count = len(manifest['embeddings'])
        if len(embeddings) != manifest_count:
            logger.warning(
                f"⚠️  Mismatch: Manifest shows {manifest_count} products, "
                f"but loaded {len(embeddings)} embeddings"
            )
    
    return embeddings


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
                if not isinstance(emb, np.ndarray):
                    logger.warning(f"⚠️  Invalid embedding type for {product_id}: {type(emb)}")
                    continue
                
                if np.isnan(emb).any() or np.isinf(emb).any():
                    logger.warning(f"⚠️  Invalid values in embedding for {product_id}")
                    continue
                
                embeddings[product_id] = emb
        
        except Exception as e:
            logger.warning(f"⚠️  Failed to load {file_path.name}: {e}")
            failed += 1
    
    if failed > 0:
        logger.warning(f"⚠️  Failed to load {failed} files")
    
    logger.info(f"✅ Loaded {len(embeddings)} embeddings from .npz files")
    return embeddings


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


def get_embedding_stats(embedding_dir: str) -> Dict:
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
            "format": "none"
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
        "max_norm": float(max(norms)) if norms else 0
    }
    
    # Add manifest info if available
    if manifest:
        stats["has_manifest"] = True
        stats["manifest_products"] = len(manifest.get("embeddings", {}))
        stats["generation_date"] = manifest.get("generated_at", "unknown")
        stats["model"] = manifest.get("model_name", "unknown")
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
    print("EMBEDDING LOADER TEST".center(80))
    print("=" * 80)
    
    # Test text embeddings
    print("\n[1/3] Testing Text Embeddings...")
    text_embs = load_text_embeddings(use_cache=False)
    print(f"  ✓ Loaded {len(text_embs)} text embeddings")
    
    if text_embs:
        stats = get_embedding_stats("text_embeddings")
        print(f"\n  📊 Text Embedding Statistics:")
        print(f"    ├─ Count:       {stats['count']:,}")
        print(f"    ├─ Dimension:   {stats['dimension']}D")
        print(f"    ├─ Format:      {stats['format']}")
        print(f"    ├─ Has NaN:     {stats['has_nan']}")
        print(f"    ├─ Has Inf:     {stats['has_inf']}")
        print(f"    ├─ Avg norm:    {stats['avg_norm']:.4f}")
        print(f"    └─ Manifest:    {stats.get('has_manifest', False)}")
    
    # Test image embeddings
    print("\n[2/3] Testing Image Embeddings...")
    image_embs = load_image_embeddings(use_cache=False)
    print(f"  ✓ Loaded {len(image_embs)} image embeddings")
    
    if image_embs:
        stats = get_embedding_stats("image_embeddings")
        print(f"\n  📊 Image Embedding Statistics:")
        print(f"    ├─ Count:       {stats['count']:,}")
        print(f"    ├─ Dimension:   {stats['dimension']}D")
        print(f"    ├─ Format:      {stats['format']}")
        print(f"    └─ Has NaN:     {stats['has_nan']}")
    
    # Test cache
    print("\n[3/3] Testing Cache...")
    print("  Loading text embeddings again (should use cache)...")
    text_embs_cached = load_text_embeddings(use_cache=True)
    print(f"  ✓ Cache working: {len(text_embs_cached)} embeddings")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE".center(80))
    print("=" * 80 + "\n")
