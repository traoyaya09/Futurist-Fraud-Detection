"""
quick-fix.py
⚡ Quick fix for immediate issues

Fixes:
1. MongoDB index conflicts
2. Missing critical dependencies
3. Hugging Face compatibility issues

Run this FIRST before starting the service or training
"""

import os
import sys
import subprocess

def fix_huggingface():
    """Fix Hugging Face compatibility issues"""
    print("\n🔧 Fixing Hugging Face compatibility...")
    try:
        # Upgrade to compatible versions
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--upgrade",
            "huggingface-hub>=0.20.0",
            "transformers>=4.30.0"
        ])
        print("  Hugging Face packages upgraded")
        return True
    except Exception as e:
        print(f"  Failed to upgrade: {e}")
        return False

def install_missing_packages():
    """Install critical missing packages"""
    print("\n📦 Installing missing packages...")
    
    packages = [
        "spacy",
        "clip-by-openai"
    ]
    
    for pkg in packages:
        try:
            print(f"  Installing {pkg}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"    {pkg} installed")
        except:
            print(f"     {pkg} failed (optional)")
    
    return True

def fix_indexes():
    """Fix MongoDB indexes"""
    print("\n🔧 Fixing MongoDB indexes...")
    
    try:
        from pymongo import MongoClient
        from dotenv import load_dotenv
        
        load_dotenv()
        
        mongo_uri = os.getenv("MONGO_URI")
        if not mongo_uri:
            print("  MONGO_URI not found in .env")
            return False
        
        client = MongoClient(mongo_uri)
        db_name = os.getenv("MONGO_DB_NAME", "futurist_ecommerce")
        collection_name = os.getenv("MONGO_COLLECTION_PRODUCTS", "products")
        
        collection = client[db_name][collection_name]
        
        # Drop old index
        try:
            collection.drop_index("name_text_description_text_brand_text")
            print("    Dropped old conflicting index")
        except Exception as e:
            if "index not found" not in str(e).lower():
                print(f"  ℹ️  Old index: {e}")
        
        # Create new index
        try:
            collection.create_index(
                [("name", "text"), ("description", "text"), ("tags", "text")],
                name="name_text_description_text_tags_text",
                weights={"name": 10, "description": 5, "tags": 8}
            )
            print("    Created new text index")
        except Exception as e:
            if "already exists" in str(e).lower():
                print("  ℹ️  Index already correct")
            else:
                print(f"     Index creation: {e}")
        
        client.close()
        print("  MongoDB indexes fixed")
        return True
        
    except Exception as e:
        print(f"  Failed to fix indexes: {e}")
        return False

def main():
    """Run all quick fixes"""
    print("="*80)
    print("⚡ QUICK FIX - Resolving immediate issues")
    print("="*80)
    
    success = True
    
    # Fix 1: Hugging Face
    if not fix_huggingface():
        print("\n   Hugging Face fix failed, but continuing...")
    
    # Fix 2: Missing packages
    if not install_missing_packages():
        print("\n   Some packages failed to install")
    
    # Fix 3: MongoDB indexes
    if not fix_indexes():
        print("\n  MongoDB index fix failed")
        success = False
    
    print("\n" + "="*80)
    if success:
        print("  QUICK FIX COMPLETED!")
        print("="*80)
        print("\n  You can now:")
        print("  1. Start the service:")
        print("     uvicorn recommendation_service_enhanced:app --reload")
        print("  2. Or run training:")
        print("     python setup-and-train.py")
    else:
        print("   SOME FIXES FAILED")
        print("="*80)
        print("\nCheck errors above and fix manually")
    print()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
