"""
product_model.py
Core machine learning functions for product recommendations 

Features:
- Collaborative filtering (user-based) with proper ID mapping
- Content-based filtering (product similarity)
- Hybrid recommendation system
- Rating matrix creation with sparse operations
- Product similarity computation
- DateTime normalization for MongoDB data
- Integration with utils scoring functions
- Memory-efficient operations

Version: 2.0.0 - DateTime Fixed & Utils Integrated
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from pymongo import MongoClient
from pymongo.collection import Collection
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import logging

# Import utils for datetime normalization and scoring
try:
    from utils.database import normalize_product
    from utils.scoring import (
        compute_popularity_score,
        compute_recency_score,
        compute_price_score
    )
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    logging.warning("Utils not available - datetime normalization disabled")

logger = logging.getLogger("ProductModel")


# ==========================================
# Ratings Preprocessing
# ==========================================

def preprocess_ratings(ratings_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
    """
    Preprocess ratings data and create ID mappings
    
    Args:
        ratings_df: DataFrame with columns [userId, productId, rating, timestamp]
    
    Returns:
        Tuple of (cleaned DataFrame, user_to_idx mapping, product_to_idx mapping)
    """
    # Ensure required columns
    required_cols = ["userId", "productId", "rating"]
    missing_cols = [col for col in required_cols if col not in ratings_df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert IDs to strings
    ratings_df["userId"] = ratings_df["userId"].astype(str)
    ratings_df["productId"] = ratings_df["productId"].astype(str)
    
    # Filter out invalid ratings
    ratings_df = ratings_df[
        (ratings_df["rating"] >= 1) & 
        (ratings_df["rating"] <= 5)
    ].copy()
    
    # Remove duplicates (keep the latest rating)
    if "timestamp" in ratings_df.columns:
        ratings_df = ratings_df.sort_values("timestamp", ascending=False)
    
    ratings_df = ratings_df.drop_duplicates(
        subset=["userId", "productId"],
        keep="first"
    )
    
    # Create mappings - 
    user_ids = sorted(ratings_df["userId"].unique())
    product_ids = sorted(ratings_df["productId"].unique())
    
    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    product_to_idx = {pid: idx for idx, pid in enumerate(product_ids)}
    
    logger.info(
        f"Preprocessed {len(ratings_df)} ratings from "
        f"{len(user_ids)} users and {len(product_ids)} products"
    )
    
    return ratings_df, user_to_idx, product_to_idx


# ==========================================
# Ratings Matrix Creation
# ==========================================

def create_ratings_matrix(
    ratings_df: pd.DataFrame,
    user_to_idx: Dict[str, int],
    product_to_idx: Dict[str, int]
) -> csr_matrix:
    """
    Create a sparse user-item ratings matrix
    
    Args:
        ratings_df: Preprocessed ratings DataFrame
        user_to_idx: User ID to matrix index mapping
        product_to_idx: Product ID to matrix index mapping
    
    Returns:
        Sparse matrix (users x products)
    """
    # Create coordinate lists using proper mappings
    rows = ratings_df["userId"].map(user_to_idx).values
    cols = ratings_df["productId"].map(product_to_idx).values
    data = ratings_df["rating"].values
    
    # Create sparse matrix
    matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(user_to_idx), len(product_to_idx))
    )
    
    sparsity = (1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])) * 100
    
    logger.info(
        f"Created ratings matrix: {matrix.shape[0]} users x "
        f"{matrix.shape[1]} products, "
        f"sparsity: {sparsity:.2f}%"
    )
    
    return matrix


# ==========================================
# Collaborative Filtering 
# ==========================================

def collaborative_filtering(
    ratings_matrix: csr_matrix,
    user_to_idx: Dict[str, int],
    product_to_idx: Dict[str, int],
    top_k: int = 50,
    min_similarity: float = 0.1
) -> Dict[str, Dict[str, float]]:
    """
    Train collaborative filtering model using user-user similarity
    
    Args:
        ratings_matrix: Sparse user-item matrix
        user_to_idx: User ID to matrix index mapping
        product_to_idx: Product ID to matrix index mapping
        top_k: Number of similar users to consider
        min_similarity: Minimum similarity threshold
    
    Returns:
        Dictionary mapping user IDs to {product_id: score}
    """
    logger.info("Training collaborative filtering model...")
    
    n_users, n_products = ratings_matrix.shape
    
    # Create reverse mappings
    idx_to_user = {idx: uid for uid, idx in user_to_idx.items()}
    idx_to_product = {idx: pid for pid, idx in product_to_idx.items()}
    
    # Compute user similarity efficiently
    # For large datasets, compute in batches
    if n_users > 5000:
        logger.info("Large dataset - using batch processing")
        user_similarity = compute_similarity_in_batches(ratings_matrix, batch_size=1000)
    else:
        user_similarity = cosine_similarity(ratings_matrix, dense_output=False)
    
    # Build recommendations dictionary with proper ID mapping 
    recommendations = {}
    
    for user_idx in range(n_users):
        user_id = idx_to_user[user_idx]  # Use actual user ID, not index
        
        # Get user's ratings
        user_ratings = ratings_matrix[user_idx].toarray().flatten()
        
        # Find similar users (excluding self)
        if hasattr(user_similarity, "toarray"):
            similarities = user_similarity[user_idx].toarray().flatten()
        else:
            similarities = user_similarity[user_idx]
        
        # Get top-k similar users with minimum similarity
        similar_mask = similarities > min_similarity
        similar_indices = np.where(similar_mask)[0]
        similar_scores = similarities[similar_indices]
        
        # Sort and get top-k (excluding self)
        sorted_idx = np.argsort(similar_scores)[::-1]
        similar_indices = similar_indices[sorted_idx]
        similar_indices = similar_indices[similar_indices != user_idx][:top_k]
        
        if len(similar_indices) == 0:
            continue
        
        # Compute weighted average of similar users' ratings
        predicted_ratings = np.zeros(n_products)
        total_similarity = 0
        
        for similar_user_idx in similar_indices:
            sim_score = similarities[similar_user_idx]
            if sim_score <= min_similarity:
                continue
            
            similar_user_ratings = ratings_matrix[similar_user_idx].toarray().flatten()
            predicted_ratings += sim_score * similar_user_ratings
            total_similarity += sim_score
        
        # Normalize by sum of similarities
        if total_similarity > 0:
            predicted_ratings /= total_similarity
        
        # Store predictions with actual IDs 
        recommendations[user_id] = {}
        
        # Only store non-zero predictions
        non_zero_idx = np.where(predicted_ratings > 0)[0]
        for product_idx in non_zero_idx:
            product_id = idx_to_product[product_idx]  #  Use actual product ID
            score = float(predicted_ratings[product_idx])
            
            # Filter out products user has already rated
            if user_ratings[product_idx] == 0:
                recommendations[user_id][product_id] = score
    
    logger.info(f"Collaborative filtering model trained for {len(recommendations)} users")
    return recommendations


def compute_similarity_in_batches(
    matrix: csr_matrix,
    batch_size: int = 1000
) -> np.ndarray:
    """
    Compute cosine similarity in batches to save memory
    
    Args:
        matrix: Sparse user-item matrix
        batch_size: Number of users per batch
    
    Returns:
        Full similarity matrix
    """
    n_users = matrix.shape[0]
    n_batches = (n_users + batch_size - 1) // batch_size
    
    similarity_matrix = np.zeros((n_users, n_users))
    
    logger.info(f"Computing similarity in {n_batches} batches...")
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_users)
        
        batch = matrix[start_idx:end_idx]
        batch_similarity = cosine_similarity(batch, matrix)
        
        similarity_matrix[start_idx:end_idx, :] = batch_similarity
        
        if (i + 1) % 10 == 0:
            logger.info(f"Processed batch {i+1}/{n_batches}")
    
    return similarity_matrix


# ==========================================
# Content-Based Filtering 
# ==========================================

def content_based_filtering(
    product_ids: List[str],
    products_collection: Collection,
    top_k: int = 20,
    use_utils: bool = True
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Compute product-to-product similarity based on content features
    
    Args:
        product_ids: List of product IDs
        products_collection: MongoDB products collection
        top_k: Number of similar products to store per product
        use_utils: Whether to use utils scoring functions
    
    Returns:
        Dictionary mapping product IDs to list of (similar_product_id, score)
    """
    logger.info("Computing content-based similarity...")
    
    # Fetch products with datetime normalization 
    products_cursor = products_collection.find(
        {"_id": {"$in": product_ids}},
        {
            "_id": 1,
            "category": 1,
            "subCategory": 1,
            "brand": 1,
            "price": 1,
            "discountPrice": 1,
            "tags": 1,
            "rating": 1,
            "reviewsCount": 1,
            "isFeatured": 1,
            "isBestseller": 1,
            "createdAt": 1
        }
    )
    
    # Normalize products if utils available 
    if UTILS_AVAILABLE and use_utils:
        products = [normalize_product(p) for p in products_cursor]
    else:
        products = list(products_cursor)
    
    if not products:
        logger.warning("No products found for content-based filtering")
        return {}
    
    # Build feature vectors
    feature_vectors = []
    product_id_map = {}
    
    for idx, product in enumerate(products):
        product_id = str(product["_id"])
        product_id_map[idx] = product_id
        
        # Create feature vector
        features = {
            "category": product.get("category", "").lower(),
            "subCategory": product.get("subCategory", "").lower(),
            "brand": product.get("brand", "").lower(),
            "price_bin": int(product.get("price", 0) // 50),  # Price bins of $50
            "tags": set(product.get("tags", [])),
            # Add quality signals if utils available
            "popularity": compute_popularity_score(product) if UTILS_AVAILABLE and use_utils else 0,
            "recency": compute_recency_score(product) if UTILS_AVAILABLE and use_utils else 0,
            "price_score": compute_price_score(product) if UTILS_AVAILABLE and use_utils else 0
        }
        
        feature_vectors.append(features)
    
    # Compute pairwise similarity
    n_products = len(products)
    similarity_matrix = np.zeros((n_products, n_products))
    
    logger.info(f"Computing similarity for {n_products} products...")
    
    for i in range(n_products):
        for j in range(i+1, n_products):
            feat_i = feature_vectors[i]
            feat_j = feature_vectors[j]
            
            score = compute_content_similarity(feat_i, feat_j, use_utils=use_utils)
            
            similarity_matrix[i, j] = score
            similarity_matrix[j, i] = score
        
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i+1}/{n_products} products")
    
    # Build similarity dictionary
    similarity_dict = {}
    
    for i in range(n_products):
        product_id = product_id_map[i]
        
        # Get top-k similar products
        similar_indices = np.argsort(similarity_matrix[i])[::-1][1:top_k+1]
        
        similar_products = [
            (product_id_map[idx], float(similarity_matrix[i, idx]))
            for idx in similar_indices
            if similarity_matrix[i, idx] > 0
        ]
        
        similarity_dict[product_id] = similar_products
    
    logger.info(f"Content similarity computed for {len(similarity_dict)} products")
    
    return similarity_dict


def compute_content_similarity(
    feat_i: Dict,
    feat_j: Dict,
    use_utils: bool = True
) -> float:
    """
    Compute similarity between two product feature vectors
    
    Args:
        feat_i: Feature dict for product i
        feat_j: Feature dict for product j
        use_utils: Whether to include quality signals
    
    Returns:
        Similarity score (0-1)
    """
    score = 0.0
    
    # Category match (40%)
    if feat_i["category"] == feat_j["category"]:
        score += 0.4
        
        # Subcategory match (bonus if category matches) (30%)
        if feat_i["subCategory"] == feat_j["subCategory"]:
            score += 0.3
    
    # Brand match (20%)
    if feat_i["brand"] and feat_i["brand"] == feat_j["brand"]:
        score += 0.2
    
    # Price similarity (10%)
    price_diff = abs(feat_i["price_bin"] - feat_j["price_bin"])
    if price_diff == 0:
        score += 0.1
    elif price_diff == 1:
        score += 0.05
    
    # Tag overlap (10%)
    if feat_i["tags"] and feat_j["tags"]:
        tag_overlap = len(feat_i["tags"] & feat_j["tags"])
        tag_union = len(feat_i["tags"] | feat_j["tags"])
        if tag_union > 0:
            score += 0.1 * (tag_overlap / tag_union)
    
    # Quality signals bonus (if utils available)
    if use_utils and UTILS_AVAILABLE:
        # Boost if both are popular/high quality
        avg_popularity = (feat_i["popularity"] + feat_j["popularity"]) / 2
        avg_recency = (feat_i["recency"] + feat_j["recency"]) / 2
        
        quality_bonus = 0.05 * avg_popularity + 0.05 * avg_recency
        score += quality_bonus
    
    return min(score, 1.0)


# ==========================================
# Hybrid Recommendations 
# ==========================================

def hybrid_recommendations(
    user_id: str,
    collaborative_model: Dict[str, Dict[str, float]],
    content_similarity: Dict[str, List[Tuple[str, float]]],
    user_history: List[str],
    top_k: int = 20,
    collaborative_weight: float = 0.6,
    content_weight: float = 0.4,
    min_score: float = 0.1
) -> List[Tuple[str, float]]:
    """
    Generate hybrid recommendations combining collaborative and content-based
    
    Args:
        user_id: User ID
        collaborative_model: Collaborative filtering predictions
        content_similarity: Content-based similarity scores
        user_history: List of product IDs the user has interacted with
        top_k: Number of recommendations to return
        collaborative_weight: Weight for collaborative score
        content_weight: Weight for content-based score
        min_score: Minimum score threshold
    
    Returns:
        List of (product_id, score) tuples sorted by score
    """
    # Get collaborative scores
    collab_scores = collaborative_model.get(user_id, {})
    
    # Get content-based scores from user history
    content_scores = {}
    
    for product_id in user_history:
        if product_id in content_similarity:
            for similar_product_id, similarity in content_similarity[product_id]:
                # Skip if already in history
                if similar_product_id not in user_history:
                    content_scores[similar_product_id] = content_scores.get(
                        similar_product_id, 0
                    ) + similarity
    
    # Normalize content scores to [0, 1]
    if content_scores:
        max_content = max(content_scores.values())
        if max_content > 0:
            content_scores = {
                pid: score / max_content
                for pid, score in content_scores.items()
            }
    
    # Combine scores
    all_product_ids = set(collab_scores.keys()) | set(content_scores.keys())
    hybrid_scores = {}
    
    for product_id in all_product_ids:
        # Skip if in user history
        if product_id in user_history:
            continue
        
        collab_score = collab_scores.get(product_id, 0)
        content_score = content_scores.get(product_id, 0)
        
        # Compute hybrid score
        hybrid_score = (
            collaborative_weight * collab_score +
            content_weight * content_score
        )
        
        # Only include if above threshold
        if hybrid_score >= min_score:
            hybrid_scores[product_id] = hybrid_score
    
    # Sort and return top-k
    recommendations = sorted(
        hybrid_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]
    
    logger.info(
        f"Generated {len(recommendations)} hybrid recommendations for user {user_id} "
        f"(collab: {len(collab_scores)}, content: {len(content_scores)})"
    )
    
    return recommendations


# ==========================================
# Model Evaluation 
# ==========================================

def evaluate_model(
    ratings_df: pd.DataFrame,
    collaborative_model: Dict[str, Dict[str, float]],
    train_ratio: float = 0.8
) -> Dict[str, float]:
    """
    Evaluate recommendation model using train/test split
    
    Args:
        ratings_df: Ratings DataFrame
        collaborative_model: Trained model
        train_ratio: Ratio of data to use for training
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Split data
    n_train = int(len(ratings_df) * train_ratio)
    train_df = ratings_df.iloc[:n_train]
    test_df = ratings_df.iloc[n_train:]
    
    # Compute metrics
    n_test = len(test_df)
    n_correct_1star = 0
    n_correct_halfstar = 0
    total_error = 0
    n_found = 0
    
    for _, row in test_df.iterrows():
        user_id = str(row["userId"])
        product_id = str(row["productId"])
        actual_rating = row["rating"]
        
        # Get prediction (default to 2.5 if not found)
        predicted_rating = collaborative_model.get(user_id, {}).get(product_id, 2.5)
        
        if product_id in collaborative_model.get(user_id, {}):
            n_found += 1
        
        error = abs(actual_rating - predicted_rating)
        total_error += error
        
        if error <= 1.0:
            n_correct_1star += 1
        if error <= 0.5:
            n_correct_halfstar += 1
    
    metrics = {
        "mae": total_error / n_test if n_test > 0 else 0,  # Mean Absolute Error
        "rmse": np.sqrt(np.mean([(r["rating"] - collaborative_model.get(str(r["userId"]), {}).get(str(r["productId"]), 2.5))**2 for _, r in test_df.iterrows()])),
        "accuracy_1star": n_correct_1star / n_test if n_test > 0 else 0,  # Within 1 star
        "accuracy_halfstar": n_correct_halfstar / n_test if n_test > 0 else 0,  # Within 0.5 star
        "coverage": len(collaborative_model) / ratings_df["userId"].nunique(),
        "prediction_coverage": n_found / n_test if n_test > 0 else 0
    }
    
    logger.info(
        f"Model evaluation: MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}, "
        f"Acc@1={metrics['accuracy_1star']:.3f}, Coverage={metrics['coverage']:.3f}"
    )
    
    return metrics


# ==========================================
# Helper Functions
# ==========================================

def save_model(
    collaborative_model: Dict[str, Dict[str, float]],
    content_similarity: Dict[str, List[Tuple[str, float]]],
    user_to_idx: Dict[str, int],
    product_to_idx: Dict[str, int],
    output_path: str
):
    """
    Save trained models to disk
    
    Args:
        collaborative_model: Collaborative filtering model
        content_similarity: Content-based similarity dict
        user_to_idx: User ID mapping
        product_to_idx: Product ID mapping
        output_path: Path to save model
    """
    import joblib
    
    model_data = {
        "collaborative_model": collaborative_model,
        "content_similarity": content_similarity,
        "user_to_idx": user_to_idx,
        "product_to_idx": product_to_idx
    }
    
    joblib.dump(model_data, output_path)
    logger.info(f"Model saved to {output_path}")


def load_model(input_path: str) -> Tuple[
    Dict[str, Dict[str, float]],
    Dict[str, List[Tuple[str, float]]],
    Dict[str, int],
    Dict[str, int]
]:
    """
    Load trained models from disk
    
    Args:
        input_path: Path to saved model
    
    Returns:
        Tuple of (collaborative_model, content_similarity, user_to_idx, product_to_idx)
    """
    import joblib
    
    model_data = joblib.load(input_path)
    
    logger.info(f"Model loaded from {input_path}")
    
    return (
        model_data["collaborative_model"],
        model_data["content_similarity"],
        model_data["user_to_idx"],
        model_data["product_to_idx"]
    )
