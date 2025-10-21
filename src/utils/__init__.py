import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import streamlit as st

def generate_realistic_invoice_data():
    """Generate realistic invoice payment dataset for business analysis"""
    print("📊 Generating realistic invoice payment dataset...")
    
    np.random.seed(42)
    
    # Business parameters
    industries = ['Technology', 'Manufacturing', 'Retail', 'Healthcare', 'Construction', 'Professional Services']
    industry_risk = {
        'Technology': {'avg_delay': 8, 'credit_mean': 720},
        'Manufacturing': {'avg_delay': 15, 'credit_mean': 650},
        'Retail': {'avg_delay': 12, 'credit_mean': 680},
        'Healthcare': {'avg_delay': 18, 'credit_mean': 620},
        'Construction': {'avg_delay': 22, 'credit_mean': 580},
        'Professional Services': {'avg_delay': 10, 'credit_mean': 700}
    }
    
    customer_segments = {
        'Excellent': {'prob': 0.15, 'credit_range': (750, 850), 'delay_multiplier': 0.3},
        'Good': {'prob': 0.25, 'credit_range': (680, 749), 'delay_multiplier': 0.6},
        'Average': {'prob': 0.35, 'credit_range': (600, 679), 'delay_multiplier': 1.0}}
    
def __init__(self):
    # Model path according to your structure
    self.model_path = os.path.join("src", "models", "saved_models", "payment_predictor.joblib")
    self.predictor = None
    self.feature_columns = None
    self.classifier = None
    self.regressor = None
    
    # Initialize session state for data sharing between pages
    if 'analyzed_data' not in st.session_state:
        st.session_state.analyzed_data = None
    
    self.load_model()