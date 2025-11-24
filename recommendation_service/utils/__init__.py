"""
Utilities Package
Exports all utility functions and classes for the recommendation service
"""

from .scoring import (
    cosine_similarity_matrix,
    compute_field_boost,
    compute_popularity_score,
    compute_recency_score,
    compute_price_score,
    compute_stock_score,
    combine_scores_vectorized,
    combine_multiple_scores,
    apply_diversity_penalty,
    compute_score_breakdown
)

from .interactions import (
    INTERACTION_WEIGHTS,
    compute_interaction_score,
    compute_recency_weight,
    aggregate_user_interactions,
    compute_user_category_preferences,
    compute_user_brand_preferences,
    compute_user_price_preference,
    compute_interaction_boost_batch,
    filter_already_interacted,
    get_trending_products,
    compute_user_similarity,
    validate_interaction
)

from .metrics import (
    RecommendationMetrics,
    PerformanceTracker,
    ABTestTracker,
    compute_click_through_rate,
    compute_conversion_rate,
    compute_average_order_value,
    compute_revenue_per_impression
)

from .database import (
    normalize_product,
    fetch_products,
    fetch_product_by_id,
    fetch_products_by_ids,
    fetch_user_interactions,
    save_interaction,
    save_recommendation_log,
    update_product_embeddings,
    batch_update_embeddings,
    get_popular_products,
    get_trending_products as get_trending_products_from_db,
    get_category_products,
    count_products,
    create_indexes
)

__all__ = [
    # Scoring
    "cosine_similarity_matrix",
    "compute_field_boost",
    "compute_popularity_score",
    "compute_recency_score",
    "compute_price_score",
    "compute_stock_score",
    "combine_scores_vectorized",
    "combine_multiple_scores",
    "apply_diversity_penalty",
    "compute_score_breakdown",
    
    # Interactions
    "INTERACTION_WEIGHTS",
    "compute_interaction_score",
    "compute_recency_weight",
    "aggregate_user_interactions",
    "compute_user_category_preferences",
    "compute_user_brand_preferences",
    "compute_user_price_preference",
    "compute_interaction_boost_batch",
    "filter_already_interacted",
    "get_trending_products",
    "compute_user_similarity",
    "validate_interaction",
    
    # Metrics
    "RecommendationMetrics",
    "PerformanceTracker",
    "ABTestTracker",
    "compute_click_through_rate",
    "compute_conversion_rate",
    "compute_average_order_value",
    "compute_revenue_per_impression",
    
    # Database
    "normalize_product",
    "fetch_products",
    "fetch_product_by_id",
    "fetch_products_by_ids",
    "fetch_user_interactions",
    "save_interaction",
    "save_recommendation_log",
    "update_product_embeddings",
    "batch_update_embeddings",
    "get_popular_products",
    "get_trending_products_from_db",
    "get_category_products",
    "count_products",
    "create_indexes",
]
