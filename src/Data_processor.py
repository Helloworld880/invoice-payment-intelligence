import pandas as pd
import numpy as np

class DataProcessor:
    """Process and validate invoice data for machine learning and analytics."""

    def __init__(self):
        # Define the columns that must exist in the dataset
        self.required_columns = [
            'invoice_id', 'customer_id', 'customer_industry', 'customer_credit_score',
            'invoice_amount', 'issue_date', 'due_date', 'payment_date'
        ]

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate the structure and completeness of invoice data.

        Args:
            df (pd.DataFrame): Input invoice dataset

        Returns:
            bool: True if validation passes, else raises ValueError
        """
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"❌ Missing required columns: {missing_cols}")

        print("✅ Data validation passed")
        return True

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate derived features for ML models and analytics.

        Args:
            df (pd.DataFrame): Raw invoice data

        Returns:
            pd.DataFrame: DataFrame with additional calculated features
        """
        # Convert date columns to datetime objects
        date_cols = ['issue_date', 'due_date', 'payment_date']
        for col in date_cols:
            df[col] = pd.to_datetime(df[col])

        # Target variable: payment delay in days
        if 'payment_delay_days' not in df.columns:
            df['payment_delay_days'] = (df['payment_date'] - df['due_date']).dt.days

        # Binary delay indicator
        if 'is_delayed' not in df.columns:
            df['is_delayed'] = (df['payment_delay_days'] > 0).astype(int)

        # Time-based features
        df['days_until_month_end'] = (df['issue_date'] + pd.offsets.MonthEnd(0) - df['issue_date']).dt.days
        df['issue_day_of_week'] = df['issue_date'].dt.dayofweek
        df['due_day_of_week'] = df['due_date'].dt.dayofweek
        df['issue_month'] = df['issue_date'].dt.month

        # Invoice amount categories
        df['amount_category'] = pd.cut(
            df['invoice_amount'],
            bins=[0, 1000, 5000, 20000, float('inf')],
            labels=['Small', 'Medium', 'Large', 'Very Large']
        )

        # Customer credit score categories
        df['credit_category'] = pd.cut(
            df['customer_credit_score'],
            bins=[0, 580, 670, 740, 800, 850],
            labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
        )

        print("✅ Feature calculation completed")
        return df

    def prepare_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and process the raw invoice data to create a dataset ready for ML training.

        Args:
            df (pd.DataFrame): Raw invoice dataset

        Returns:
            pd.DataFrame: Processed dataset with features
        """
        self.validate_data(df)
        df_processed = self.calculate_features(df)
        print("✅ Training data prepared successfully")
        return df_processed
