import pandas as pd
import numpy as np
from datetime import datetime

class DataProcessor:
    """Process and validate invoice data for machine learning"""
    
    def __init__(self):
        self.required_columns = [
            'invoice_id', 'customer_id', 'customer_industry', 'customer_credit_score',
            'invoice_amount', 'issue_date', 'due_date', 'payment_date'
        ]
    
    def validate_data(self, df):
        """Validate invoice data structure and quality"""
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print("✅ Data validation passed")
        return True
    
    def calculate_features(self, df):
        """Calculate derived features from raw data"""
        # Convert dates
        df['issue_date'] = pd.to_datetime(df['issue_date'])
        df['due_date'] = pd.to_datetime(df['due_date'])
        df['payment_date'] = pd.to_datetime(df['payment_date'])
        
        # Ensure target variable exists
        if 'payment_delay_days' not in df.columns:
            df['payment_delay_days'] = (df['payment_date'] - df['due_date']).dt.days
        
        if 'is_delayed' not in df.columns:
            df['is_delayed'] = (df['payment_delay_days'] > 0).astype(int)
        
        # Create business-relevant features
        df['days_until_month_end'] = (df['issue_date'] + pd.offsets.MonthEnd(0) - df['issue_date']).dt.days
        df['issue_day_of_week'] = df['issue_date'].dt.dayofweek
        df['due_day_of_week'] = df['due_date'].dt.dayofweek
        df['issue_month'] = df['issue_date'].dt.month
        
        # Amount categories
        df['amount_category'] = pd.cut(
            df['invoice_amount'],
            bins=[0, 1000, 5000, 20000, float('inf')],
            labels=['Small', 'Medium', 'Large', 'Very Large']
        )
        
        # Credit score categories
        df['credit_category'] = pd.cut(
            df['customer_credit_score'],
            bins=[0, 580, 670, 740, 800, 850],
            labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
        )
        
        print("✅ Feature calculation completed")
        return df
    
    def prepare_training_data(self, df):
        """Prepare complete dataset for model training"""
        self.validate_data(df)
        df_processed = self.calculate_features(df)
        return df_processed