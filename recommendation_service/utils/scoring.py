"""
Scoring Utilities for Recommendation Service
Handles all scoring logic including similarity calculations, boosting, and score combination
"""

import numpy as np
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger("RecommendationService.Scoring")


def cosine_similarity_matrix(query_vecs: np.ndarray, product_vecs: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between query and product vectors
    
    Args:
        query_vecs: Query vectors (can be 1D or 2D)
        product_vecs: Product vectors (2D array)
        
    Returns:
        Cosine similarity scores
    """
    query_vecs = np.atleast_2d(query_vecs)
    product_vecs = np.array(product_vecs)
    
    # Add small epsilon to avoid division by zero
    query_norms = np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-8
    prod_norms = np.linalg.norm(product_vecs, axis=1, keepdims=True) + 1e-8
    
    # Compute cosine similarity
    sim = (query_vecs @ product_vecs.T) / (query_norms @ prod_norms.T)
    return sim.squeeze()


def compute_field_boost(product: Dict[str, Any], query: str) -> float:
    """
    Compute boost score based on field matches (name, category, brand, etc.)
    
    Args:
        product: Product document
        query: Search query string
        
    Returns:
        Boost score (0.0 to 1.0)
    """
    boost = 0.0
    q_words = set(query.lower().split())
    
    # Name match (highest priority)
    if any(w in product.get("name", "").lower() for w in q_words):
        boost += 0.3
    
    # Category match
    if any(w in product.get("category", "").lower() for w in q_words):
        boost += 0.2
    
    # SubCategory match
    if any(w in product.get("subCategory", "").lower() for w in q_words):
        boost += 0.2
    
    # Brand match
    if any(w in product.get("brand", "").lower() for w in q_words):
        boost += 0.1
    
    # Tags match
    tags = product.get("tags", [])
    if isinstance(tags, list) and any(any(w in tag.lower() for w in q_words) for tag in tags):
        boost += 0.1
    
    # Description match (lower priority)
    if any(w in product.get("description", "").lower() for w in q_words):
        boost += 0.1
    
    return min(boost, 1.0)  # Cap at 1.0


def compute_popularity_score(product: Dict[str, Any]) -> float:
    """
    Compute popularity score based on ratings, reviews, and sales
    
    Args:
        product: Product document
        
    Returns:
        Popularity score (0.0 to 1.0)
    """
    score = 0.0
    
    # Rating component (0-5 -> 0-0.4)
    rating = product.get("rating", 0)
    if rating > 0:
        score += (rating / 5.0) * 0.4
    
    # Reviews count component (log scale, 0-0.3)
    reviews_count = product.get("reviewsCount", 0)
    if reviews_count > 0:
        # Use log scale: 1 review = 0.1, 10 reviews = 0.2, 100+ reviews = 0.3
        score += min(np.log10(reviews_count + 1) / 2.0, 0.3)
    
    # Featured/Bestseller flags (0-0.3)
    if product.get("isFeatured", False):
        score += 0.1
    if product.get("isBestseller", False):
        score += 0.2
    
    return min(score, 1.0)


def compute_recency_score(product: Dict[str, Any], decay_days: int = 30) -> float:
    """
    Compute recency score with exponential decay
    
    Args:
        product: Product document
        decay_days: Days for score to decay to ~0.37
        
    Returns:
        Recency score (0.0 to 1.0)
    """
    from datetime import datetime
    
    created_at = product.get("createdAt")
    if not created_at:
        return 0.0
    
    try:
        if isinstance(created_at, str):
            created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        else:
            created_date = created_at
        
        days_old = (datetime.utcnow() - created_date.replace(tzinfo=None)).days
        
        # Exponential decay: e^(-days/decay_days)
        score = np.exp(-days_old / decay_days)
        return float(score)
    except Exception as e:
        logger.warning(f"Error computing recency score: {e}")
        return 0.0


def compute_price_score(product: Dict[str, Any], user_price_preference: Optional[float] = None) -> float:
    """
    Compute price attractiveness score
    
    Args:
        product: Product document
        user_price_preference: User's typical price range (optional)
        
    Returns:
        Price score (0.0 to 1.0)
    """
    price = product.get("price", 0)
    discount_price = product.get("discountPrice")
    
    score = 0.0
    
    # Discount bonus
    if discount_price and discount_price < price:
        discount_percent = (price - discount_price) / price
        score += discount_percent * 0.5  # Up to 0.5 for 100% discount
    
    # Price range match (if user preference provided)
    if user_price_preference and price > 0:
        # Penalize if price is very different from preference
        price_diff = abs(price - user_price_preference) / user_price_preference
        if price_diff < 0.2:  # Within 20%
            score += 0.3
        elif price_diff < 0.5:  # Within 50%
            score += 0.2
        elif price_diff < 1.0:  # Within 100%
            score += 0.1
    
    # Promotion bonus
    if product.get("promotion"):
        score += 0.2
    
    return min(score, 1.0)


def compute_stock_score(product: Dict[str, Any]) -> float:
    """
    Compute stock availability score
    
    Args:
        product: Product document
        
    Returns:
        Stock score (0.0 to 1.0)
    """
    stock = product.get("stock", 0)
    
    if stock <= 0:
        return 0.0
    elif stock < 5:
        return 0.5  # Low stock warning
    elif stock < 20:
        return 0.8  # Moderate stock
    else:
        return 1.0  # Good stock


def combine_scores_vectorized(
    hybrid_scores: np.ndarray,
    text_sims: np.ndarray,
    image_sims: np.ndarray,
    interaction_boost: np.ndarray,
    field_boosts: np.ndarray,
    weights: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Combine multiple scores using weighted average
    
    Args:
        hybrid_scores: Collaborative filtering scores
        text_sims: Text similarity scores
        image_sims: Image similarity scores
        interaction_boost: User interaction boost scores
        field_boosts: Field match boost scores
        weights: Custom weights (if None, uses defaults)
        
    Returns:
        Combined scores
    """
    if weights is None:
        weights = {
            "hybrid": 0.25,
            "text": 0.25,
            "image": 0.15,
            "interaction": 0.20,
            "field": 0.15
        }
    
    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}
    
    # Combine scores
    combined = (
        weights.get("hybrid", 0.25) * hybrid_scores +
        weights.get("text", 0.25) * text_sims +
        weights.get("image", 0.15) * image_sims +
        weights.get("interaction", 0.20) * interaction_boost +
        weights.get("field", 0.15) * field_boosts
    )
    
    return combined


def combine_multiple_scores(
    products: List[Dict[str, Any]],
    base_scores: np.ndarray,
    query: Optional[str] = None,
    user_id: Optional[str] = None,
    include_popularity: bool = True,
    include_recency: bool = True,
    include_price: bool = True,
    weights: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Combine base scores with additional scoring factors
    
    Args:
        products: List of product documents
        base_scores: Base recommendation scores
        query: Search query (for field boosting)
        user_id: User ID (for personalization)
        include_popularity: Include popularity scoring
        include_recency: Include recency scoring
        include_price: Include price scoring
        weights: Custom weights for combining scores
        
    Returns:
        Final combined scores
    """
    if weights is None:
        weights = {
            "base": 0.50,
            "popularity": 0.20,
            "recency": 0.10,
            "price": 0.10,
            "field": 0.10
        }
    
    n_products = len(products)
    final_scores = base_scores.copy()
    
    # Add popularity scores
    if include_popularity:
        popularity_scores = np.array([compute_popularity_score(p) for p in products])
        final_scores = final_scores * weights["base"] + popularity_scores * weights["popularity"]
    
    # Add recency scores
    if include_recency:
        recency_scores = np.array([compute_recency_score(p) for p in products])
        final_scores = final_scores + recency_scores * weights["recency"]
    
    # Add price scores
    if include_price:
        price_scores = np.array([compute_price_score(p) for p in products])
        final_scores = final_scores + price_scores * weights["price"]
    
    # Add field boost scores
    if query:
        field_scores = np.array([compute_field_boost(p, query) for p in products])
        final_scores = final_scores + field_scores * weights["field"]
    
    # Normalize to [0, 1]
    if final_scores.max() > 0:
        final_scores = final_scores / final_scores.max()
    
    return final_scores


def apply_diversity_penalty(scores: np.ndarray, products: List[Dict[str, Any]], alpha: float = 0.5) -> np.ndarray:
    """
    Apply diversity penalty to avoid recommending too many similar products
    
    Args:
        scores: Current recommendation scores
        products: List of product documents
        alpha: Diversity weight (0 = no diversity, 1 = maximum diversity)
        
    Returns:
        Adjusted scores with diversity penalty
    """
    if alpha <= 0:
        return scores
    
    adjusted_scores = scores.copy()
    selected_categories = set()
    selected_brands = set()
    
    for i, product in enumerate(products):
        category = product.get("category", "")
        brand = product.get("brand", "")
        
        # Penalize if category already selected
        if category in selected_categories:
            adjusted_scores[i] *= (1 - alpha * 0.3)
        else:
            selected_categories.add(category)
        
        # Penalize if brand already selected
        if brand in selected_brands:
            adjusted_scores[i] *= (1 - alpha * 0.2)
        else:
            selected_brands.add(brand)
    
    return adjusted_scores


def compute_score_breakdown(
    product: Dict[str, Any],
    base_score: float,
    text_sim: float = 0.0,
    image_sim: float = 0.0,
    interaction_score: float = 0.0,
    query: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute detailed score breakdown for debugging/transparency
    
    Args:
        product: Product document
        base_score: Base recommendation score
        text_sim: Text similarity score
        image_sim: Image similarity score
        interaction_score: User interaction score
        query: Search query
        
    Returns:
        Dictionary with score breakdown
    """
    return {
        "base_score": float(base_score),
        "text_similarity": float(text_sim),
        "image_similarity": float(image_sim),
        "interaction_score": float(interaction_score),
        "popularity_score": compute_popularity_score(product),
        "recency_score": compute_recency_score(product),
        "price_score": compute_price_score(product),
        "stock_score": compute_stock_score(product),
        "field_boost": compute_field_boost(product, query) if query else 0.0
    }
