# scripts/train_run.py - CORRECTED for YOUR structure
import sys
import os
import pandas as pd
import numpy as np
import joblib

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.Model_trainer import InvoicePredictor
from src.Data_processor import DataProcessor

def run_training_pipeline():
    print("🚀 Starting training pipeline...")
    
    # Create directories according to YOUR structure
    os.makedirs('src/models/saved_models', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    
    # Step 1: Generate dataset if not exists
    data_path = 'data/raw/invoice_data_cleaned.csv'
    if not os.path.exists(data_path):
        print("📊 Generating dataset...")
        from generate_dataset import generate_realistic_invoice_data
        df_raw = generate_realistic_invoice_data()
    else:
        print("📊 Loading existing dataset...")
        df_raw = pd.read_csv(data_path)
    
    # Step 2: Process data
    print("🔧 Processing data...")
    processor = DataProcessor()
    df_processed = processor.prepare_training_data(df_raw)
    
    # Add target variables
    df_processed['DelayStatus'] = df_processed['is_delayed']
    df_processed['DelayDays'] = df_processed['payment_delay_days']
    
    # Step 3: Train model
    print("🤖 Training model...")
    trainer = InvoicePredictor()
    trained_model, feature_columns = trainer.train_models(df_processed)
    
    # Step 4: Save model - CORRECT PATH for YOUR structure
    model_path = 'src/models/saved_models/payment_predictor.joblib'
    model_data = {
        'model': trained_model,
        'feature_columns': feature_columns,
        'classifier': trainer.classifier,
        'regressor': trainer.regressor,
        'feature_importance': trainer.feature_importance
    }
    
    joblib.dump(model_data, model_path)
    print(f"✅ Model saved successfully at: {model_path}")
    print(f"📋 Features used: {len(feature_columns)}")
    print("🎉 Training pipeline completed!")

if __name__ == "__main__":
    run_training_pipeline()