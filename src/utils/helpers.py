import logging
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
from typing import Dict, Any, Optional

# ---------------- Logging Setup ---------------- #
def setup_logging() -> logging.Logger:
    """Setup application logging"""
    logger = logging.getLogger('InvoicePaymentApp')
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create handlers
        file_handler = logging.FileHandler('logs/app.log')
        stream_handler = logging.StreamHandler()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    
    return logger

# ---------------- File Upload Validation ---------------- #
def validate_file_upload(uploaded_file, max_size_mb: int = 100) -> bool:
    """Validate uploaded file for security and size"""
    try:
        # Check file size
        if uploaded_file.size > max_size_mb * 1024 * 1024:
            st.error(f"❌ File too large. Maximum size is {max_size_mb}MB")
            return False
        
        # Check file type
        if uploaded_file.type not in ['text/csv', 'application/vnd.ms-excel']:
            st.error("❌ Only CSV files are allowed")
            return False
            
        return True
        
    except Exception as e:
        st.error(f"❌ File validation error: {str(e)}")
        return False

# ---------------- Currency Formatting ---------------- #
def format_currency(amount: float) -> str:
    """Format a number as standard currency with commas and 2 decimals"""
    return "${:,.2f}".format(amount)

# ---------------- Financial Impact Calculation ---------------- #
def calculate_financial_impact(data: pd.DataFrame) -> Dict[str, float]:
    """Calculate financial impact of payment delays"""
    high_risk_data = data[data['risk_level'] == 'High']
    if len(high_risk_data) == 0:
        return {
            'opportunity_cost': 0,
            'estimated_savings': 0,
            'high_risk_amount': 0,
            'avg_high_risk_delay': 0
        }
    
    total_high_risk_amount = high_risk_data['invoice_amount'].sum()
    avg_delay = high_risk_data['predicted_delay_days'].mean()
    
    # Assume 8% annual interest rate for opportunity cost
    daily_interest_rate = 0.08 / 365
    opportunity_cost = total_high_risk_amount * daily_interest_rate * avg_delay
    
    # Assume 70% of high-risk invoices can be recovered with intervention
    estimated_savings = opportunity_cost * 0.7
    
    return {
        'opportunity_cost': opportunity_cost,
        'estimated_savings': estimated_savings,
        'high_risk_amount': total_high_risk_amount,
        'avg_high_risk_delay': avg_delay
    }

# ---------------- Data Preprocessing ---------------- #
@st.cache_data(ttl=3600)
def preprocess_data(_data: pd.DataFrame) -> pd.DataFrame:
    """Cacheable data preprocessing"""
    processed = _data.copy()
    
    # Ensure numeric columns
    numeric_cols = ['invoice_amount', 'customer_credit_score', 'due_days',
                    'avg_payment_delay_history', 'payment_consistency']
    
    for col in numeric_cols:
        if col in processed.columns:
            processed[col] = pd.to_numeric(processed[col], errors='coerce').fillna(0)
    
    return processed

# ---------------- Demo Predictions ---------------- #
def create_demo_predictions(input_data: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Create realistic demo predictions when model is not available"""
    np.random.seed(42)
    
    delay_probs = []
    predicted_delays = []
    
    for _, row in input_data.iterrows():
        base_risk = 0.3
        
        # Credit score impact
        credit_score = row.get('customer_credit_score', 650)
        if credit_score < 600:
            base_risk += 0.3
        elif credit_score < 680:
            base_risk += 0.15
        
        # Industry risk factors
        industry = row.get('customer_industry', 'Technology')
        high_risk_industries = ['Construction', 'Healthcare']
        medium_risk_industries = ['Manufacturing', 'Retail']
        
        if industry in high_risk_industries:
            base_risk += 0.2
        elif industry in medium_risk_industries:
            base_risk += 0.1
        
        # Payment history impact
        history_delay = row.get('avg_payment_delay_history', 10)
        if history_delay > 20:
            base_risk += 0.2
        elif history_delay > 10:
            base_risk += 0.1
        
        # Invoice amount impact
        amount = row.get('invoice_amount', 10000)
        if amount > 50000:
            base_risk += 0.1
        elif amount > 20000:
            base_risk += 0.05
        
        # Payment consistency impact
        consistency = row.get('payment_consistency', 0.8)
        base_risk += (1 - consistency) * 0.2
        
        # Add randomness
        delay_prob = min(0.95, base_risk + np.random.normal(0, 0.05))
        predicted_delay = max(0, history_delay * delay_prob * np.random.uniform(0.8, 1.2))
        
        delay_probs.append(delay_prob)
        predicted_delays.append(predicted_delay)
    
    return {
        'delay_probs': np.array(delay_probs),
        'predicted_delays': np.array(predicted_delays)
    }
