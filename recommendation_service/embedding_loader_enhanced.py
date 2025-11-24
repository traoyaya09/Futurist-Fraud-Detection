"""
embedding_loader_enhanced.py
✅ Production-ready embedding loader for FastAPI service

Features:
- ✅ Lazy loading of embeddings
- ✅ Caching mechanism
- ✅ Automatic refresh when files change
- ✅ Error handling and fallbacks
- ✅ Memory-efficient loading
"""

import os
import glob
import logging
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from datetime import datetime

logger = logging.getLogger("EmbeddingLoader")

class EmbeddingCache:
    """
    Manages embedding loading and caching with automatic refresh
    """
    
    def __init__(self, cache_ttl_seconds: int = 300):
        self.cache_ttl = cache_ttl_seconds
        self.text_embeddings: Optional[Dict[str, np.ndarray]] = None
        self.image_embeddings: Optional[Dict[str, np.ndarray]] = None
        self.last_load_time: Optional[datetime] = None
        self.last_modified_time: Optional[datetime] = None
    
    def is_cache_valid(self, embedding_dir: str) -> bool:
        """Check if cache is still valid"""
        if self.last_load_time is None:
            return False
        
        # Check TTL
        elapsed = (datetime.utcnow() - self.last_load_time).total_seconds()
        if elapsed > self.cache_ttl:
            logger.info(f"🔄 Cache TTL expired ({elapsed:.0f}s > {self.cache_ttl}s)")
            return False
        
        # Check if files have been modified
        try:
            latest_mtime = self._get_latest_mtime(embedding_dir)
            if latest_mtime and self.last_modified_time:
                if latest_mtime > self.last_modified_time:
                    logger.info(f"🔄 Embedding files updated, refreshing cache")
                    return False
        except Exception as e:
            logger.warning(f"⚠️ Failed to check file modification time: {e}")
        
        return True
    
    def _get_latest_mtime(self, embedding_dir: str) -> Optional[datetime]:
        """Get the latest modification time of embedding files"""
        files = glob.glob(str(Path(embedding_dir) / "*.npz"))
        if not files:
            return None
        
        latest = max(os.path.getmtime(f) for f in files)
        return datetime.fromtimestamp(latest)
    
    def load(self, embedding_dir: str, embedding_type: str) -> Dict[str, np.ndarray]:
        """
        Load embeddings with caching
        
        Args:
            embedding_dir: Directory containing .npz files
            embedding_type: 'text' or 'image'
        
        Returns:
            Dict mapping product_id -> embedding
        """
        # Check cache
        if embedding_type == 'text' and self.text_embeddings:
            if self.is_cache_valid(embedding_dir):
                logger.debug(f"✅ Using cached text embeddings ({len(self.text_embeddings)} items)")
                return self.text_embeddings
        
        elif embedding_type == 'image' and self.image_embeddings:
            if self.is_cache_valid(embedding_dir):
                logger.debug(f"✅ Using cached image embeddings ({len(self.image_embeddings)} items)")
                return self.image_embeddings
        
        # Load from disk
        logger.info(f"📂 Loading {embedding_type} embeddings from {embedding_dir}...")
        embeddings = load_and_merge_embeddings(embedding_dir)
        
        # Update cache
        if embedding_type == 'text':
            self.text_embeddings = embeddings
        else:
            self.image_embeddings = embeddings
        
        self.last_load_time = datetime.utcnow()
        self.last_modified_time = self._get_latest_mtime(embedding_dir)
        
        logger.info(f"✅ Loaded {len(embeddings)} {embedding_type} embeddings")
        return embeddings
    
    def clear(self):
        """Clear cache"""
        self.text_embeddings = None
        self.image_embeddings = None
        self.last_load_time = None
        self.last_modified_time = None
        logger.info("🧹 Cache cleared")

# Global cache instance
_cache = EmbeddingCache(cache_ttl_seconds=300)

def load_and_merge_embeddings(
    embedding_dir: str,
    use_cache: bool = True
) -> Dict[str, np.ndarray]:
    """
    Load and merge all embedding files from directory
    
    Args:
        embedding_dir: Directory containing .npz files
        use_cache: Whether to use caching (default: True)
    
    Returns:
        Dict mapping product_id -> embedding vector
    """
    if not os.path.exists(embedding_dir):
        logger.warning(f"⚠️ Embedding directory not found: {embedding_dir}")
        return {}
    
    all_files = glob.glob(str(Path(embedding_dir) / "*.npz"))
    
    if not all_files:
        logger.warning(f"⚠️ No embedding files found in {embedding_dir}")
        return {}
    
    merged = {}
    failed_files = []
    
    for file_path in all_files:
        try:
            data = np.load(file_path, allow_pickle=True)
            
            # Extract IDs and embeddings
            ids = data.get("ids", [])
            embeddings = data.get("embeddings", [])
            
            if len(ids) != len(embeddings):
                logger.warning(f"⚠️ Mismatch in {file_path}: {len(ids)} ids vs {len(embeddings)} embeddings")
                continue
            
            # Merge into dict
            for pid, emb in zip(ids, embeddings):
                pid_str = str(pid)
                
                # Validate embedding
                if not isinstance(emb, np.ndarray):
                    logger.warning(f"⚠️ Invalid embedding type for {pid_str}: {type(emb)}")
                    continue
                
                if np.isnan(emb).any():
                    logger.warning(f"⚠️ NaN values in embedding for {pid_str}")
                    continue
                
                merged[pid_str] = emb
        
        except Exception as e:
            logger.warning(f"⚠️ Failed to load {file_path}: {e}")
            failed_files.append(file_path)
    
    if failed_files:
        logger.warning(f"⚠️ Failed to load {len(failed_files)} files")
    
    logger.info(f"✅ Merged {len(merged)} embeddings from {len(all_files)} files in {embedding_dir}")
    return merged

def load_text_embeddings(
    embedding_dir: str = "text_embeddings",
    use_cache: bool = True
) -> Dict[str, np.ndarray]:
    """
    Load text embeddings with optional caching
    """
    if use_cache:
        return _cache.load(embedding_dir, 'text')
    return load_and_merge_embeddings(embedding_dir, use_cache=False)

def load_image_embeddings(
    embedding_dir: str = "image_embeddings",
    use_cache: bool = True
) -> Dict[str, np.ndarray]:
    """
    Load image embeddings with optional caching
    """
    if use_cache:
        return _cache.load(embedding_dir, 'image')
    return load_and_merge_embeddings(embedding_dir, use_cache=False)

def load_merged_embeddings(
    model_dir: str = "models",
    embedding_type: str = "text"
) -> Dict[str, np.ndarray]:
    """
    Load merged embeddings from the final .npz file in models directory
    
    Args:
        model_dir: Directory containing merged embeddings
        embedding_type: 'text' or 'image'
    
    Returns:
        Dict mapping product_id -> embedding
    """
    filename = f"merged_{embedding_type}_embeddings.npz"
    file_path = Path(model_dir) / filename
    
    if not file_path.exists():
        logger.warning(f"⚠️ Merged embeddings not found: {file_path}")
        return {}
    
    try:
        data = np.load(file_path, allow_pickle=True)
        ids = data.get("ids", [])
        embeddings = data.get("embeddings", [])
        
        merged = {str(pid): emb for pid, emb in zip(ids, embeddings)}
        logger.info(f"✅ Loaded {len(merged)} {embedding_type} embeddings from {file_path}")
        return merged
    
    except Exception as e:
        logger.error(f"❌ Failed to load merged embeddings from {file_path}: {e}")
        return {}

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

def clear_cache():
    """Clear the global embedding cache"""
    _cache.clear()

# Statistics function
def get_embedding_stats(embedding_dir: str) -> Dict:
    """
    Get statistics about embeddings in a directory
    """
    embeddings = load_and_merge_embeddings(embedding_dir, use_cache=False)
    
    if not embeddings:
        return {"count": 0, "dimension": 0, "has_nan": False, "has_inf": False}
    
    sample_emb = next(iter(embeddings.values()))
    dimension = sample_emb.shape[0]
    
    has_nan = any(np.isnan(emb).any() for emb in embeddings.values())
    has_inf = any(np.isinf(emb).any() for emb in embeddings.values())
    
    # Calculate average norm
    norms = [np.linalg.norm(emb) for emb in embeddings.values()]
    avg_norm = np.mean(norms)
    
    return {
        "count": len(embeddings),
        "dimension": dimension,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "avg_norm": float(avg_norm),
        "min_norm": float(min(norms)),
        "max_norm": float(max(norms))
    }

if __name__ == "__main__":
    # Test loading
    import sys
    logging.basicConfig(level=logging.INFO)
    
    print("\n=== Testing Embedding Loader ===\n")
    
    # Test text embeddings
    print("Loading text embeddings...")
    text_embs = load_text_embeddings()
    print(f"✅ Loaded {len(text_embs)} text embeddings")
    
    if text_embs:
        stats = get_embedding_stats("text_embeddings")
        print(f"📊 Text embedding stats: {stats}")
    
    # Test image embeddings
    print("\nLoading image embeddings...")
    image_embs = load_image_embeddings()
    print(f"✅ Loaded {len(image_embs)} image embeddings")
    
    if image_embs:
        stats = get_embedding_stats("image_embeddings")
        print(f"📊 Image embedding stats: {stats}")
    
    # Test merged embeddings
    print("\nLoading merged embeddings...")
    merged_text = load_merged_embeddings(embedding_type="text")
    merged_image = load_merged_embeddings(embedding_type="image")
    print(f"✅ Loaded {len(merged_text)} merged text embeddings")
    print(f"✅ Loaded {len(merged_image)} merged image embeddings")
    
    print("\n=== Test Complete ===\n")
