"""
Interaction Tracking Utilities
Handles user interaction tracking, scoring, and analysis
"""

import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger("RecommendationService.Interactions")


# Interaction type weights
INTERACTION_WEIGHTS = {
    "view": 1.0,
    "click": 1.5,
    "add_to_cart": 3.0,
    "wishlist": 2.5,
    "purchase": 5.0,
    "remove_from_cart": -1.0,
    "return": -2.0,
    "review": 4.0,
    "share": 2.0
}


def compute_interaction_score(interaction_type: str, recency_weight: float = 1.0) -> float:
    """
    Compute score for a single interaction
    
    Args:
        interaction_type: Type of interaction (view, click, purchase, etc.)
        recency_weight: Weight based on how recent the interaction was
        
    Returns:
        Interaction score
    """
    base_score = INTERACTION_WEIGHTS.get(interaction_type.lower(), 0.5)
    return base_score * recency_weight


def compute_recency_weight(timestamp: datetime, decay_days: int = 7) -> float:
    """
    Compute recency weight using exponential decay
    
    Args:
        timestamp: Interaction timestamp
        decay_days: Days for weight to decay to ~0.37
        
    Returns:
        Recency weight (0.0 to 1.0)
    """
    try:
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        days_ago = (datetime.utcnow() - timestamp.replace(tzinfo=None)).days
        weight = np.exp(-days_ago / decay_days)
        return float(max(0.1, weight))  # Minimum weight of 0.1
    except Exception as e:
        logger.warning(f"Error computing recency weight: {e}")
        return 0.5


def aggregate_user_interactions(
    interactions: List[Dict[str, Any]],
    decay_days: int = 7
) -> Dict[str, float]:
    """
    Aggregate user interactions by product with recency weighting
    
    Args:
        interactions: List of interaction documents
        decay_days: Days for recency decay
        
    Returns:
        Dictionary mapping product_id -> aggregated score
    """
    product_scores = defaultdict(float)
    
    for interaction in interactions:
        product_id = interaction.get("productId")
        interaction_type = interaction.get("interactionType")
        timestamp = interaction.get("timestamp", datetime.utcnow())
        
        if not product_id or not interaction_type:
            continue
        
        recency_weight = compute_recency_weight(timestamp, decay_days)
        score = compute_interaction_score(interaction_type, recency_weight)
        
        product_scores[product_id] += score
    
    return dict(product_scores)


def compute_user_category_preferences(
    interactions: List[Dict[str, Any]],
    products_map: Dict[str, Dict[str, Any]]
) -> Dict[str, float]:
    """
    Compute user's category preferences based on interactions
    
    Args:
        interactions: List of interaction documents
        products_map: Mapping of product_id -> product document
        
    Returns:
        Dictionary mapping category -> preference score
    """
    category_scores = defaultdict(float)
    
    for interaction in interactions:
        product_id = interaction.get("productId")
        interaction_type = interaction.get("interactionType")
        timestamp = interaction.get("timestamp", datetime.utcnow())
        
        if product_id not in products_map:
            continue
        
        product = products_map[product_id]
        category = product.get("category", "unknown")
        
        recency_weight = compute_recency_weight(timestamp)
        score = compute_interaction_score(interaction_type, recency_weight)
        
        category_scores[category] += score
    
    # Normalize scores
    total_score = sum(category_scores.values())
    if total_score > 0:
        category_scores = {k: v / total_score for k, v in category_scores.items()}
    
    return dict(category_scores)


def compute_user_brand_preferences(
    interactions: List[Dict[str, Any]],
    products_map: Dict[str, Dict[str, Any]]
) -> Dict[str, float]:
    """
    Compute user's brand preferences based on interactions
    
    Args:
        interactions: List of interaction documents
        products_map: Mapping of product_id -> product document
        
    Returns:
        Dictionary mapping brand -> preference score
    """
    brand_scores = defaultdict(float)
    
    for interaction in interactions:
        product_id = interaction.get("productId")
        interaction_type = interaction.get("interactionType")
        timestamp = interaction.get("timestamp", datetime.utcnow())
        
        if product_id not in products_map:
            continue
        
        product = products_map[product_id]
        brand = product.get("brand", "unknown")
        
        if not brand or brand == "unknown":
            continue
        
        recency_weight = compute_recency_weight(timestamp)
        score = compute_interaction_score(interaction_type, recency_weight)
        
        brand_scores[brand] += score
    
    # Normalize scores
    total_score = sum(brand_scores.values())
    if total_score > 0:
        brand_scores = {k: v / total_score for k, v in brand_scores.items()}
    
    return dict(brand_scores)


def compute_user_price_preference(
    interactions: List[Dict[str, Any]],
    products_map: Dict[str, Dict[str, Any]]
) -> Optional[float]:
    """
    Compute user's average price preference
    
    Args:
        interactions: List of interaction documents
        products_map: Mapping of product_id -> product document
        
    Returns:
        Average price preference or None
    """
    prices = []
    weights = []
    
    for interaction in interactions:
        product_id = interaction.get("productId")
        interaction_type = interaction.get("interactionType")
        
        if product_id not in products_map:
            continue
        
        product = products_map[product_id]
        price = product.get("discountPrice") or product.get("price")
        
        if not price or price <= 0:
            continue
        
        weight = INTERACTION_WEIGHTS.get(interaction_type.lower(), 0.5)
        prices.append(price)
        weights.append(weight)
    
    if not prices:
        return None
    
    # Weighted average
    weighted_avg = np.average(prices, weights=weights)
    return float(weighted_avg)


def compute_interaction_boost_batch(
    product_ids: List[str],
    user_interactions: Dict[str, float],
    category_preferences: Optional[Dict[str, float]] = None,
    brand_preferences: Optional[Dict[str, float]] = None,
    products_map: Optional[Dict[str, Dict[str, Any]]] = None
) -> np.ndarray:
    """
    Compute interaction boost scores for a batch of products
    
    Args:
        product_ids: List of product IDs
        user_interactions: Mapping of product_id -> interaction score
        category_preferences: User's category preferences
        brand_preferences: User's brand preferences
        products_map: Mapping of product_id -> product document
        
    Returns:
        Array of boost scores
    """
    boosts = np.zeros(len(product_ids))
    
    for i, product_id in enumerate(product_ids):
        # Direct interaction boost
        if product_id in user_interactions:
            boosts[i] += user_interactions[product_id] * 0.5
        
        # Category preference boost
        if category_preferences and products_map and product_id in products_map:
            product = products_map[product_id]
            category = product.get("category", "")
            if category in category_preferences:
                boosts[i] += category_preferences[category] * 0.3
        
        # Brand preference boost
        if brand_preferences and products_map and product_id in products_map:
            product = products_map[product_id]
            brand = product.get("brand", "")
            if brand in brand_preferences:
                boosts[i] += brand_preferences[brand] * 0.2
    
    # Normalize to [0, 1]
    if boosts.max() > 0:
        boosts = boosts / boosts.max()
    
    return boosts


def filter_already_interacted(
    product_ids: List[str],
    user_interactions: Dict[str, float],
    interaction_types_to_exclude: Optional[List[str]] = None
) -> List[str]:
    """
    Filter out products user has already interacted with
    
    Args:
        product_ids: List of product IDs
        user_interactions: User's interaction history
        interaction_types_to_exclude: Types of interactions to filter
        
    Returns:
        Filtered list of product IDs
    """
    if not interaction_types_to_exclude:
        interaction_types_to_exclude = ["purchase", "return"]
    
    # For now, just filter by presence in interactions
    # In production, you'd check specific interaction types
    filtered = [pid for pid in product_ids if pid not in user_interactions]
    
    return filtered


def get_trending_products(
    interactions: List[Dict[str, Any]],
    time_window_hours: int = 24,
    top_k: int = 20
) -> List[tuple]:
    """
    Get trending products based on recent interactions
    
    Args:
        interactions: List of all interactions
        time_window_hours: Time window for trending calculation
        top_k: Number of top trending products to return
        
    Returns:
        List of (product_id, trend_score) tuples
    """
    cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
    product_scores = defaultdict(float)
    
    for interaction in interactions:
        timestamp = interaction.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        if timestamp.replace(tzinfo=None) < cutoff_time:
            continue
        
        product_id = interaction.get("productId")
        interaction_type = interaction.get("interactionType")
        
        if not product_id or not interaction_type:
            continue
        
        # Higher weight for more recent interactions
        hours_ago = (datetime.utcnow() - timestamp.replace(tzinfo=None)).total_seconds() / 3600
        recency_factor = np.exp(-hours_ago / (time_window_hours / 2))
        
        score = compute_interaction_score(interaction_type) * recency_factor
        product_scores[product_id] += score
    
    # Sort by score
    trending = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    return trending


def compute_user_similarity(
    user1_interactions: Dict[str, float],
    user2_interactions: Dict[str, float]
) -> float:
    """
    Compute similarity between two users based on their interactions
    
    Args:
        user1_interactions: First user's product interactions
        user2_interactions: Second user's product interactions
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    # Get common products
    common_products = set(user1_interactions.keys()) & set(user2_interactions.keys())
    
    if not common_products:
        return 0.0
    
    # Jaccard similarity
    all_products = set(user1_interactions.keys()) | set(user2_interactions.keys())
    jaccard = len(common_products) / len(all_products)
    
    # Cosine similarity on common products
    if len(common_products) > 0:
        vec1 = np.array([user1_interactions.get(p, 0) for p in common_products])
        vec2 = np.array([user2_interactions.get(p, 0) for p in common_products])
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 > 0 and norm2 > 0:
            cosine = np.dot(vec1, vec2) / (norm1 * norm2)
        else:
            cosine = 0.0
    else:
        cosine = 0.0
    
    # Combine Jaccard and cosine
    similarity = 0.5 * jaccard + 0.5 * cosine
    
    return float(similarity)


def validate_interaction(interaction: Dict[str, Any]) -> bool:
    """
    Validate interaction data
    
    Args:
        interaction: Interaction document
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ["userId", "productId", "interactionType"]
    
    for field in required_fields:
        if field not in interaction or not interaction[field]:
            logger.warning(f"Missing required field: {field}")
            return False
    
    interaction_type = interaction["interactionType"].lower()
    if interaction_type not in INTERACTION_WEIGHTS:
        logger.warning(f"Unknown interaction type: {interaction_type}")
        return False
    
    return True
