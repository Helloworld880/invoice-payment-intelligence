# tests/run_tests_windows.py
import os
import sys
import unittest

def main():
    """Run all tests in the tests directory."""
    
    # Add project root to sys.path for imports
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    # Discover and load all test files starting with 'test_'
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=os.path.dirname(__file__), pattern='test_*.py')

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return appropriate exit code
    sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == "__main__":
    main()
