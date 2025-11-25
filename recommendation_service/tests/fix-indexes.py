"""
fix-indexes.py
🔧 Fix MongoDB index conflicts by removing old indexes and creating new ones

This script resolves the IndexOptionsConflict error by:
1. Dropping the old conflicting text index
2. Creating the new text index with correct fields
"""

import os
import sys
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

def fix_indexes():
    """Fix MongoDB text index conflicts"""
    print("\n" + "="*80)
    print("🔧 MongoDB Index Fix Utility")
    print("="*80)
    
    # Connect to MongoDB
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        print("❌ Error: MONGO_URI not found in environment")
        return False
    
    print(f"📡 Connecting to MongoDB...")
    client = MongoClient(mongo_uri)
    
    # Get database and collection
    db_name = os.getenv("MONGO_DB_NAME", "futurist_ecommerce")
    collection_name = os.getenv("MONGO_COLLECTION_PRODUCTS", "products")
    
    db = client[db_name]
    collection = db[collection_name]
    
    print(f"📂 Database: {db_name}")
    print(f"📦 Collection: {collection_name}")
    
    # List current indexes
    print("\n🔍 Current indexes:")
    indexes = collection.list_indexes()
    for idx in indexes:
        print(f"  - {idx['name']}: {idx.get('key', {})}")
    
    # Drop old conflicting text index
    try:
        print("\n🗑️  Dropping old text index...")
        collection.drop_index("name_text_description_text_brand_text")
        print("✅ Old index dropped successfully")
    except Exception as e:
        if "index not found" in str(e).lower():
            print("ℹ️  Old index not found (already removed)")
        else:
            print(f"⚠️  Error dropping index: {e}")
    
    # Create new text index
    try:
        print("\n📝 Creating new text index...")
        collection.create_index(
            [
                ("name", "text"),
                ("description", "text"),
                ("tags", "text")
            ],
            name="name_text_description_text_tags_text",
            weights={
                "name": 10,
                "description": 5,
                "tags": 8
            },
            default_language="english"
        )
        print("✅ New text index created successfully")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("ℹ️  Index already exists with correct configuration")
        else:
            print(f"❌ Error creating index: {e}")
            return False
    
    # Create other necessary indexes
    print("\n📝 Creating other indexes...")
    
    # Category index
    try:
        collection.create_index("category", name="category_idx")
        print("✅ Category index created")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("ℹ️  Category index already exists")
    
    # Price index
    try:
        collection.create_index("price", name="price_idx")
        print("✅ Price index created")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("ℹ️  Price index already exists")
    
    # Brand index
    try:
        collection.create_index("brand", name="brand_idx")
        print("✅ Brand index created")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("ℹ️  Brand index already exists")
    
    # Rating index
    try:
        collection.create_index("averageRating", name="rating_idx")
        print("✅ Rating index created")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("ℹ️  Rating index already exists")
    
    # Compound index for recommendations
    try:
        collection.create_index(
            [("category", 1), ("averageRating", -1)],
            name="category_rating_idx"
        )
        print("✅ Compound category-rating index created")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("ℹ️  Compound index already exists")
    
    # List final indexes
    print("\n✅ Final indexes:")
    indexes = collection.list_indexes()
    for idx in indexes:
        print(f"  - {idx['name']}")
    
    client.close()
    print("\n" + "="*80)
    print("🎉 Index fix completed successfully!")
    print("="*80)
    return True

if __name__ == "__main__":
    try:
        success = fix_indexes()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
