# test_setup.py
import os
import sys
import subprocess

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.config import Config
from src.utils.helpers import format_currency, calculate_financial_impact
from src.utils.validators import DataValidator, InputValidator

def check_project_structure():
    """Check critical files, directories, and model availability."""
    
    print("Checking project structure...\n")
    
    paths_to_check = [
        'app.py',
        'scripts/train_run.py',
        'src/Model_trainer.py',
        'src/Data_processor.py',
        'data/raw/',
        'src/models/saved_models/'
    ]
    
    for path in paths_to_check:
        exists = os.path.exists(path)
        status = "Found" if exists else "Missing"
        print(f"{status}: {path}")
    
    model_path = 'src/models/saved_models/payment_predictor.joblib'
    if os.path.exists(model_path):
        print(f"Model exists: {model_path}")
    else:
        print(f"Model NOT FOUND: {model_path}")
        print("Running training pipeline...")
        # Run the training script safely
        subprocess.run([sys.executable, 'scripts/train_run.py'], check=True)

if __name__ == "__main__":
    check_project_structure()
