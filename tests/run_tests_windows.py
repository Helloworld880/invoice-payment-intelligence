# tests/run_tests_windows.py
import sys
import os
import subprocess

def run_tests_windows():
    """Run tests on Windows with proper path setup"""
    
    # Get the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Add project root to Python path
    sys.path.insert(0, project_root)
    
    print(f"🔧 Project root: {project_root}")
    print(f"🔧 Python path: {sys.path}")
    
    test_files = [
        'test_setup.py',
        'test_utils.py', 
        'test_app.py'
    ]
    
    for test_file in test_files:
        test_path = os.path.join(current_dir, test_file)
        print(f"\n{'='*60}")
        print(f"🚀 Running {test_file}...")
        print('='*60)
        
        try:
            # Read and execute the test file
            with open(test_path, 'r', encoding='utf-8') as f:
                test_code = f.read()
            
            # Add the project root to the code
            exec(test_code, {'__name__': '__main__', 'project_root': project_root})
            
        except Exception as e:
            print(f"❌ Error running {test_file}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_tests_windows()