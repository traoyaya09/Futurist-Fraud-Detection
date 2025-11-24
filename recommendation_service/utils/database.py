"""
Database Utilities
Helper functions for MongoDB operations
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pymongo.collection import Collection
from pymongo import UpdateOne, ASCENDING, DESCENDING
import logging

logger = logging.getLogger("RecommendationService.Database")


def normalize_product(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize product document to match backend ProductModel.js schema
    Ensures compatibility with MongoDB Product collection
    
    Args:
        raw: Raw product document from MongoDB
        
    Returns:
        Normalized product document
    """
    return {
        "_id": str(raw.get("_id", "")),
        "name": raw.get("name", "Unnamed Product"),
        "description": raw.get("description", "No description available"),
        "shortDescription": raw.get("shortDescription", ""),
        "price": raw.get("price"),
        "discountPrice": raw.get("discountPrice"),
        "category": raw.get("category", "Uncategorized"),
        "subCategory": raw.get("subCategory", ""),
        "brand": raw.get("brand", ""),
        "stock": raw.get("stock", 0),
        "imageUrl": raw.get("imageUrl") or raw.get("image") or "https://via.placeholder.com/400",
        "images": raw.get("images", []),
        "rating": raw.get("rating", 0),
        "reviewsCount": raw.get("reviewsCount", 0),
        "reviews": raw.get("reviews", []),
        "isFeatured": raw.get("isFeatured", False),
        "isNewProduct": raw.get("isNewProduct", False),
        "isBestseller": raw.get("isBestseller", False),
        "isOnSale": raw.get("isOnSale", False),
        "promotion": raw.get("promotion"),
        "tags": raw.get("tags", []),
        "status": raw.get("status", "active"),
        "createdAt": raw.get("createdAt")
    }


def fetch_products(
    collection: Collection,
    skip: int = 0,
    limit: int = 20,
    query: Optional[str] = None,
    category: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    sort_by: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Fetch products from MongoDB with filtering and pagination
    
    Args:
        collection: MongoDB collection
        skip: Number of documents to skip
        limit: Maximum number of documents to return
        query: Text search query
        category: Filter by category
        min_price: Minimum price filter
        max_price: Maximum price filter
        sort_by: Sort field (e.g., "price", "-rating")
        
    Returns:
        List of product documents
    """
    filter_query = {"stock": {"$gt": 0}, "status": "active"}
    
    # Text search
    if query:
        filter_query["$text"] = {"$search": query}
    
    # Category filter
    if category:
        filter_query["category"] = category
    
    # Price range filter
    if min_price is not None or max_price is not None:
        price_filter = {}
        if min_price is not None:
            price_filter["$gte"] = min_price
        if max_price is not None:
            price_filter["$lte"] = max_price
        filter_query["price"] = price_filter
    
    # Build sort
    sort_criteria = []
    if sort_by:
        if sort_by.startswith("-"):
            sort_criteria.append((sort_by[1:], DESCENDING))
        else:
            sort_criteria.append((sort_by, ASCENDING))
    else:
        sort_criteria.append(("_id", ASCENDING))
    
    try:
        cursor = collection.find(filter_query).skip(skip).limit(limit)
        
        if sort_criteria:
            cursor = cursor.sort(sort_criteria)
        
        products = list(cursor)
        return [normalize_product(p) for p in products]
    
    except Exception as e:
        logger.error(f"Error fetching products: {e}")
        return []


def fetch_product_by_id(
    collection: Collection,
    product_id: str
) -> Optional[Dict[str, Any]]:
    """
    Fetch a single product by ID
    
    Args:
        collection: MongoDB collection
        product_id: Product ID
        
    Returns:
        Product document or None
    """
    try:
        product = collection.find_one({"_id": product_id})
        if product:
            return normalize_product(product)
        return None
    except Exception as e:
        logger.error(f"Error fetching product {product_id}: {e}")
        return None


def fetch_products_by_ids(
    collection: Collection,
    product_ids: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Fetch multiple products by IDs
    
    Args:
        collection: MongoDB collection
        product_ids: List of product IDs
        
    Returns:
        Dictionary mapping product_id -> product document
    """
    try:
        products = collection.find({"_id": {"$in": product_ids}})
        return {str(p["_id"]): normalize_product(p) for p in products}
    except Exception as e:
        logger.error(f"Error fetching products by IDs: {e}")
        return {}


def fetch_user_interactions(
    collection: Collection,
    user_id: str,
    limit: int = 100,
    days_back: int = 90
) -> List[Dict[str, Any]]:
    """
    Fetch user's recent interactions
    
    Args:
        collection: MongoDB collection
        user_id: User ID
        limit: Maximum number of interactions
        days_back: Number of days to look back
        
    Returns:
        List of interaction documents
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        interactions = collection.find({
            "userId": user_id,
            "timestamp": {"$gte": cutoff_date}
        }).sort("timestamp", DESCENDING).limit(limit)
        
        return list(interactions)
    
    except Exception as e:
        logger.error(f"Error fetching interactions for user {user_id}: {e}")
        return []


def save_interaction(
    collection: Collection,
    user_id: str,
    product_id: str,
    interaction_type: str,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Save a user interaction
    
    Args:
        collection: MongoDB collection
        user_id: User ID
        product_id: Product ID
        interaction_type: Type of interaction
        metadata: Additional metadata
        
    Returns:
        True if successful, False otherwise
    """
    try:
        interaction_doc = {
            "userId": user_id,
            "productId": product_id,
            "interactionType": interaction_type,
            "timestamp": datetime.utcnow(),
            "metadata": metadata or {}
        }
        
        collection.insert_one(interaction_doc)
        return True
    
    except Exception as e:
        logger.error(f"Error saving interaction: {e}")
        return False


def save_recommendation_log(
    collection: Collection,
    user_id: Optional[str],
    query: Optional[str],
    recommended_products: List[str],
    scores: List[float],
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Log a recommendation request
    
    Args:
        collection: MongoDB collection
        user_id: User ID (optional)
        query: Search query (optional)
        recommended_products: List of recommended product IDs
        scores: Recommendation scores
        metadata: Additional metadata
        
    Returns:
        True if successful, False otherwise
    """
    try:
        log_doc = {
            "userId": user_id,
            "query": query,
            "recommendedProducts": recommended_products,
            "scores": scores,
            "timestamp": datetime.utcnow(),
            "metadata": metadata or {}
        }
        
        collection.insert_one(log_doc)
        return True
    
    except Exception as e:
        logger.error(f"Error saving recommendation log: {e}")
        return False


def update_product_embeddings(
    collection: Collection,
    product_id: str,
    text_embedding: Optional[List[float]] = None,
    image_embedding: Optional[List[float]] = None
) -> bool:
    """
    Update product embeddings in database
    
    Args:
        collection: MongoDB collection
        product_id: Product ID
        text_embedding: Text embedding vector
        image_embedding: Image embedding vector
        
    Returns:
        True if successful, False otherwise
    """
    try:
        update_doc = {}
        
        if text_embedding is not None:
            update_doc["textEmbedding"] = text_embedding
        
        if image_embedding is not None:
            update_doc["imageEmbedding"] = image_embedding
        
        if update_doc:
            update_doc["embeddingsUpdatedAt"] = datetime.utcnow()
            collection.update_one(
                {"_id": product_id},
                {"$set": update_doc}
            )
        
        return True
    
    except Exception as e:
        logger.error(f"Error updating embeddings for product {product_id}: {e}")
        return False


def batch_update_embeddings(
    collection: Collection,
    embeddings_data: List[Dict[str, Any]]
) -> int:
    """
    Batch update product embeddings
    
    Args:
        collection: MongoDB collection
        embeddings_data: List of {product_id, text_embedding, image_embedding}
        
    Returns:
        Number of successful updates
    """
    try:
        operations = []
        
        for data in embeddings_data:
            product_id = data.get("product_id")
            if not product_id:
                continue
            
            update_doc = {"embeddingsUpdatedAt": datetime.utcnow()}
            
            if "text_embedding" in data:
                update_doc["textEmbedding"] = data["text_embedding"]
            
            if "image_embedding" in data:
                update_doc["imageEmbedding"] = data["image_embedding"]
            
            operations.append(
                UpdateOne(
                    {"_id": product_id},
                    {"$set": update_doc}
                )
            )
        
        if operations:
            result = collection.bulk_write(operations, ordered=False)
            return result.modified_count
        
        return 0
    
    except Exception as e:
        logger.error(f"Error batch updating embeddings: {e}")
        return 0


def get_popular_products(
    collection: Collection,
    limit: int = 20,
    days_back: int = 30
) -> List[Dict[str, Any]]:
    """
    Get popular products based on ratings and reviews
    
    Args:
        collection: MongoDB collection
        limit: Maximum number of products
        days_back: Consider products from last N days
        
    Returns:
        List of popular products
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        products = collection.find({
            "status": "active",
            "stock": {"$gt": 0},
            "rating": {"$gte": 4.0},
            "reviewsCount": {"$gte": 10}
        }).sort([
            ("rating", DESCENDING),
            ("reviewsCount", DESCENDING)
        ]).limit(limit)
        
        return [normalize_product(p) for p in products]
    
    except Exception as e:
        logger.error(f"Error fetching popular products: {e}")
        return []


def get_trending_products(
    interactions_collection: Collection,
    products_collection: Collection,
    limit: int = 20,
    hours_back: int = 24
) -> List[Dict[str, Any]]:
    """
    Get trending products based on recent interactions
    
    Args:
        interactions_collection: Interactions collection
        products_collection: Products collection
        limit: Maximum number of products
        hours_back: Consider interactions from last N hours
        
    Returns:
        List of trending products
    """
    try:
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        # Aggregate interactions
        pipeline = [
            {"$match": {"timestamp": {"$gte": cutoff_time}}},
            {"$group": {
                "_id": "$productId",
                "count": {"$sum": 1}
            }},
            {"$sort": {"count": DESCENDING}},
            {"$limit": limit}
        ]
        
        trending_ids = [doc["_id"] for doc in interactions_collection.aggregate(pipeline)]
        
        if not trending_ids:
            return []
        
        # Fetch product details
        products = products_collection.find({
            "_id": {"$in": trending_ids},
            "status": "active",
            "stock": {"$gt": 0}
        })
        
        # Maintain order
        products_map = {str(p["_id"]): normalize_product(p) for p in products}
        return [products_map[pid] for pid in trending_ids if pid in products_map]
    
    except Exception as e:
        logger.error(f"Error fetching trending products: {e}")
        return []


def get_category_products(
    collection: Collection,
    category: str,
    limit: int = 20,
    skip: int = 0
) -> List[Dict[str, Any]]:
    """
    Get products from a specific category
    
    Args:
        collection: MongoDB collection
        category: Category name
        limit: Maximum number of products
        skip: Number of products to skip
        
    Returns:
        List of products in category
    """
    try:
        products = collection.find({
            "category": category,
            "status": "active",
            "stock": {"$gt": 0}
        }).skip(skip).limit(limit)
        
        return [normalize_product(p) for p in products]
    
    except Exception as e:
        logger.error(f"Error fetching category products: {e}")
        return []


def count_products(
    collection: Collection,
    filters: Optional[Dict[str, Any]] = None
) -> int:
    """
    Count products matching filters
    
    Args:
        collection: MongoDB collection
        filters: Filter criteria
        
    Returns:
        Product count
    """
    try:
        if filters is None:
            filters = {"status": "active", "stock": {"$gt": 0}}
        
        return collection.count_documents(filters)
    
    except Exception as e:
        logger.error(f"Error counting products: {e}")
        return 0


def create_indexes(
    products_collection: Collection,
    interactions_collection: Collection,
    logs_collection: Collection
):
    """
    Create necessary indexes for optimal performance
    
    Args:
        products_collection: Products collection
        interactions_collection: Interactions collection
        logs_collection: Logs collection
    """
    try:
        # Products indexes
        products_collection.create_index([("status", ASCENDING), ("stock", ASCENDING)])
        products_collection.create_index([("category", ASCENDING)])
        products_collection.create_index([("rating", DESCENDING)])
        products_collection.create_index([("createdAt", DESCENDING)])
        products_collection.create_index([("price", ASCENDING)])
        
        # Text index for search
        products_collection.create_index([
            ("name", "text"),
            ("description", "text"),
            ("tags", "text")
        ])
        
        # Interactions indexes
        interactions_collection.create_index([("userId", ASCENDING), ("timestamp", DESCENDING)])
        interactions_collection.create_index([("productId", ASCENDING)])
        interactions_collection.create_index([("timestamp", DESCENDING)])
        interactions_collection.create_index([("interactionType", ASCENDING)])
        
        # Logs indexes
        logs_collection.create_index([("userId", ASCENDING), ("timestamp", DESCENDING)])
        logs_collection.create_index([("timestamp", DESCENDING)])
        
        logger.info("✅ Database indexes created successfully")
    
    except Exception as e:
        logger.error(f"Error creating indexes: {e}")
