#!/usr/bin/env python3
"""
Simple script to test the configuration
"""
import sys
from pathlib import Path
import os

# Add the project root directory to Python path
project_root = str(Path(__file__).parent / "backend")
sys.path.insert(0, project_root)

# Set the working directory to the backend directory so .env can be loaded
os.chdir(project_root)

from app.core.config import settings

def test_config():
    """Test if the configuration loads correctly"""
    try:
        print("Testing configuration loading...")
        print(f"  Database URL: {'SET' if settings.database_url else 'NOT SET'}")
        print(f"  Open Router Key: {'SET' if settings.open_router_key else 'NOT SET'}")
        print(f"  Qdrant URL: {'SET' if settings.qdrant_url else 'NOT SET'}")
        print(f"  Vector Store Type: {settings.vector_store_type}")
        print(f"  Collection Name: {settings.collection_name}")

        # Check if the API key looks valid (starts with 'sk-or-')
        if settings.open_router_key.startswith('sk-or-'):
            print("  ✓ Open Router API key format looks correct")
        else:
            print("  ✗ Open Router API key format may be incorrect")

        return True
    except Exception as e:
        print(f"  ✗ Error in config test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_config()
    if success:
        print("\n✓ Configuration test completed successfully!")
    else:
        print("\n✗ Configuration test failed!")
        sys.exit(1)