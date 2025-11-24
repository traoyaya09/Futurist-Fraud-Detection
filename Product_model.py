"""
product_model.py
Core machine learning functions for product recommendations

Features:
- Collaborative filtering (user-based)
- Content-based filtering (product similarity)
- Hybrid recommendation system
- Rating matrix creation
- Product similarity computation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import logging

logger = logging.getLogger("ProductModel")


# ==========================================
# Ratings Preprocessing
# ==========================================

def preprocess_ratings(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess ratings data
    
    Args:
        ratings_df: DataFrame with columns [userId, productId, rating, timestamp]
    
    Returns:
        Cleaned DataFrame
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
    
    logger.info(f"Preprocessed {len(ratings_df)} ratings")
    return ratings_df


# ==========================================
# Ratings Matrix Creation
# ==========================================

def create_ratings_matrix(ratings_df: pd.DataFrame) -> csr_matrix:
    """
    Create a sparse user-item ratings matrix
    
    Args:
        ratings_df: Preprocessed ratings DataFrame
    
    Returns:
        Sparse matrix (users x products)
    """
    # Create user and product ID mappings
    user_ids = ratings_df["userId"].unique()
    product_ids = ratings_df["productId"].unique()
    
    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    product_to_idx = {pid: idx for idx, pid in enumerate(product_ids)}
    
    # Create coordinate lists
    rows = ratings_df["userId"].map(user_to_idx).values
    cols = ratings_df["productId"].map(product_to_idx).values
    data = ratings_df["rating"].values
    
    # Create sparse matrix
    matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(user_ids), len(product_ids))
    )
    
    logger.info(
        f"Created ratings matrix: {matrix.shape[0]} users x "
        f"{matrix.shape[1]} products, "
        f"sparsity: {(1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])) * 100:.2f}%"
    )
    
    return matrix


# ==========================================
# Collaborative Filtering
# ==========================================

def collaborative_filtering(
    ratings_matrix: csr_matrix,
    top_k: int = 50,
    similarity_metric: str = "cosine"
) -> Dict[str, Dict[str, float]]:
    """
    Train collaborative filtering model using user-user similarity
    
    Args:
        ratings_matrix: Sparse user-item matrix
        top_k: Number of similar users to consider
        similarity_metric: Similarity metric to use
    
    Returns:
        Dictionary mapping user IDs to product scores
    """
    logger.info("Training collaborative filtering model...")
    
    # Convert to dense for computation (if small enough)
    if ratings_matrix.shape[0] > 10000:
        logger.warning("Large user base - using sparse operations")
        # Use sparse matrix operations
        user_similarity = cosine_similarity(ratings_matrix, dense_output=False)
    else:
        user_similarity = cosine_similarity(ratings_matrix.toarray())
    
    # Build recommendations dictionary
    recommendations = {}
    n_users, n_products = ratings_matrix.shape
    
    for user_idx in range(n_users):
        # Get user's ratings
        user_ratings = ratings_matrix[user_idx].toarray().flatten()
        
        # Find similar users (excluding self)
        similarities = user_similarity[user_idx]
        if hasattr(similarities, "toarray"):
            similarities = similarities.toarray().flatten()
        
        # Get top-k similar users
        similar_users = np.argsort(similarities)[::-1][1:top_k+1]
        
        # Compute weighted average of similar users' ratings
        predicted_ratings = np.zeros(n_products)
        
        for similar_user_idx in similar_users:
            sim_score = similarities[similar_user_idx]
            if sim_score <= 0:
                continue
            
            similar_user_ratings = ratings_matrix[similar_user_idx].toarray().flatten()
            predicted_ratings += sim_score * similar_user_ratings
        
        # Normalize by sum of similarities
        sim_sum = np.sum([similarities[idx] for idx in similar_users if similarities[idx] > 0])
        if sim_sum > 0:
            predicted_ratings /= sim_sum
        
        # Store predictions (product_id -> score)
        user_id = str(user_idx)
        recommendations[user_id] = {}
        
        for product_idx in range(n_products):
            if predicted_ratings[product_idx] > 0:
                product_id = str(product_idx)
                recommendations[user_id][product_id] = float(predicted_ratings[product_idx])
    
    logger.info(f"Collaborative filtering model trained for {len(recommendations)} users")
    return recommendations


# ==========================================
# Content-Based Filtering
# ==========================================

def content_based_filtering(
    product_ids: List[str],
    mongo_uri: str,
    db_name: str,
    products_collection: str = "products",
    top_k: int = 20
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Compute product-to-product similarity based on content features
    
    Args:
        product_ids: List of product IDs
        mongo_uri: MongoDB connection URI
        db_name: Database name
        products_collection: Products collection name
        top_k: Number of similar products to store per product
    
    Returns:
        Dictionary mapping product IDs to list of (similar_product_id, score)
    """
    logger.info("Computing content-based similarity...")
    
    # Connect to MongoDB
    client = MongoClient(mongo_uri)
    db = client[db_name]
    products_col = db[products_collection]
    
    # Fetch products
    products = list(products_col.find(
        {"_id": {"$in": product_ids}},
        {
            "_id": 1,
            "category": 1,
            "subCategory": 1,
            "brand": 1,
            "price": 1,
            "tags": 1
        }
    ))
    
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
            "tags": set(product.get("tags", []))
        }
        
        feature_vectors.append(features)
    
    # Compute pairwise similarity
    n_products = len(products)
    similarity_matrix = np.zeros((n_products, n_products))
    
    for i in range(n_products):
        for j in range(i+1, n_products):
            feat_i = feature_vectors[i]
            feat_j = feature_vectors[j]
            
            score = 0.0
            
            # Category match
            if feat_i["category"] == feat_j["category"]:
                score += 0.4
                
                # Subcategory match (bonus if category matches)
                if feat_i["subCategory"] == feat_j["subCategory"]:
                    score += 0.3
            
            # Brand match
            if feat_i["brand"] and feat_i["brand"] == feat_j["brand"]:
                score += 0.2
            
            # Price similarity
            price_diff = abs(feat_i["price_bin"] - feat_j["price_bin"])
            if price_diff <= 1:
                score += 0.1
            
            # Tag overlap
            if feat_i["tags"] and feat_j["tags"]:
                tag_overlap = len(feat_i["tags"] & feat_j["tags"])
                tag_union = len(feat_i["tags"] | feat_j["tags"])
                if tag_union > 0:
                    score += 0.1 * (tag_overlap / tag_union)
            
            similarity_matrix[i, j] = score
            similarity_matrix[j, i] = score
    
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
    client.close()
    
    return similarity_dict


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
    content_weight: float = 0.4
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
    
    Returns:
        List of (product_id, score) tuples
    """
    # Get collaborative scores
    collab_scores = collaborative_model.get(user_id, {})
    
    # Get content-based scores from user history
    content_scores = {}
    
    for product_id in user_history:
        if product_id in content_similarity:
            for similar_product_id, similarity in content_similarity[product_id]:
                if similar_product_id not in user_history:
                    content_scores[similar_product_id] = content_scores.get(
                        similar_product_id, 0
                    ) + similarity
    
    # Normalize content scores
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
        if product_id in user_history:
            continue
        
        collab_score = collab_scores.get(product_id, 0)
        content_score = content_scores.get(product_id, 0)
        
        hybrid_score = (
            collaborative_weight * collab_score +
            content_weight * content_score
        )
        
        hybrid_scores[product_id] = hybrid_score
    
    # Sort and return top-k
    recommendations = sorted(
        hybrid_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]
    
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
    n_correct = 0
    total_error = 0
    
    for _, row in test_df.iterrows():
        user_id = str(row["userId"])
        product_id = str(row["productId"])
        actual_rating = row["rating"]
        
        predicted_rating = collaborative_model.get(user_id, {}).get(product_id, 2.5)
        
        error = abs(actual_rating - predicted_rating)
        total_error += error
        
        if error <= 1.0:
            n_correct += 1
    
    metrics = {
        "mae": total_error / n_test if n_test > 0 else 0,  # Mean Absolute Error
        "accuracy": n_correct / n_test if n_test > 0 else 0,  # Within 1 star
        "coverage": len(collaborative_model) / ratings_df["userId"].nunique()
    }
    
    logger.info(f"Model evaluation: MAE={metrics['mae']:.3f}, Accuracy={metrics['accuracy']:.3f}")
    
    return metrics
