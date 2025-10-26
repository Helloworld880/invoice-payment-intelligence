# tests/simple_test.py
import sys
import os

# Add project root to path
sys.path.append('.')

def test_basic_imports():
    """Test basic imports"""
    try:
        from src.utils.config import Config
        from src.utils.helpers import format_currency
        print("✅ Basic imports successful!")
        return True
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        return False

def test_config():
    """Test configuration"""
    try:
        from src.utils.config import Config
        config = Config()
        app_name = config.get('app.name')
        print(f"✅ Config loaded: {app_name}")
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_helpers():
    """Test helper functions"""
    try:
        from src.utils.helpers import format_currency
        result = format_currency(1000)
        print(f"✅ Helpers work: {result}")
        return True
    except Exception as e:
        print(f"❌ Helpers test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Running Simple Tests...")
    print("=" * 50)
    
    tests = [test_basic_imports, test_config, test_helpers]
    passed = 0
    
    for test in tests:
        if test():
            passed += 1
    
    print("=" * 50)
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All simple tests passed!")
    else:
        print("⚠️ Some tests failed. Check the errors above.")