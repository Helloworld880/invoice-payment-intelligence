# scripts/train_run.py - Humanized training pipeline

import sys
import os
import pandas as pd
import joblib

# Add project root to path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Model_trainer import InvoicePredictor
from src.Data_processor import DataProcessor

def run_training_pipeline():
    """Full pipeline: data preparation → model training → save model"""
    print("🚀 Starting the training pipeline...")

    # --- Step 0: Ensure directory structure ---
    os.makedirs('src/models/saved_models', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)

    # --- Step 1: Load or generate dataset ---
    data_path = 'data/raw/invoice_data_cleaned.csv'
    if not os.path.exists(data_path):
        print("📊 Dataset not found. Generating realistic invoice dataset...")
        from generate_dataset import generate_realistic_invoice_data
        df_raw = generate_realistic_invoice_data()
    else:
        print(f"📊 Loading existing dataset from {data_path}...")
        df_raw = pd.read_csv(data_path)

    print(f"✅ Dataset ready. Shape: {df_raw.shape}")

    # --- Step 2: Process and prepare data ---
    print("🔧 Processing and calculating features...")
    processor = DataProcessor()
    df_processed = processor.prepare_training_data(df_raw)

    # Add target columns for training
    df_processed['DelayStatus'] = df_processed['is_delayed']
    df_processed['DelayDays'] = df_processed['payment_delay_days']

    print(f"✅ Data processing complete. Columns available: {list(df_processed.columns)}")

    # --- Step 3: Train the model ---
    print("🤖 Training invoice payment prediction model...")
    trainer = InvoicePredictor()
    trained_model, feature_columns = trainer.train_models(df_processed)

    # --- Step 4: Save the trained model ---
    model_path = 'src/models/saved_models/payment_predictor.joblib'
    model_data = {
        'model': trained_model,
        'feature_columns': feature_columns,
        'classifier': trainer.classifier,
        'regressor': trainer.regressor,
        'feature_importance': trainer.feature_importance
    }

    joblib.dump(model_data, model_path)
    print(f"💾 Model saved successfully at: {model_path}")
    print(f"📋 Total features used: {len(feature_columns)}")

    print("🎉 Training pipeline completed successfully!")

if __name__ == "__main__":
    run_training_pipeline()
