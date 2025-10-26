# test_setup.py
import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.config import Config
from src.utils.helpers import format_currency, calculate_financial_impact
from src.utils.validators import DataValidator, InputValidator

def check_project_structure():
    print("🔍 Checking project structure...")
    
    # Check critical files and directories
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
        status = "✅" if exists else "❌"
        print(f"{status} {path}: {exists}")
    
    # Check if model exists
    model_path = 'src/models/saved_models/payment_predictor.joblib'
    model_exists = os.path.exists(model_path)
    print(f"{'✅' if model_exists else '❌'} Model file: {model_path} - {model_exists}")
    
    if not model_exists:
        print("\n🚨 MODEL NOT FOUND! Running training pipeline...")
        # Run the training
        os.system('python scripts/train_run.py')

if __name__ == "__main__":
    check_project_structure()