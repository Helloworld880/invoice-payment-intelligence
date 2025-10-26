import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class FeatureEngineer:
    """Feature engineering for payment delay prediction."""

    def __init__(self):
        # StandardScaler for numerical features
        self.scaler = StandardScaler()
        # LabelEncoders for categorical features
        self.encoders = {}
        # Keep track of feature names used in the model
        self.feature_names = []

    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate advanced business and time-based features.

        Args:
            df (pd.DataFrame): Input invoice dataset

        Returns:
            pd.DataFrame: Dataset with new engineered features
        """
        # Customer behavior: invoice amount vs customer average
        customer_avg = df.groupby('customer_id')['invoice_amount'].mean()
        df['amount_vs_customer_avg'] = df['invoice_amount'] / df['customer_id'].map(customer_avg)

        # Customer payment trends
        if 'payment_delay_days' in df.columns:
            df['customer_delay_trend'] = df.groupby('customer_id')['payment_delay_days'].transform(
                lambda x: x.rolling(3, min_periods=1).mean()
            )

        # Time-based features
        df['is_q4'] = df['issue_date'].dt.month.isin([10, 11, 12]).astype(int)
        df['is_month_end'] = (df['issue_date'].dt.day > 25).astype(int)

        return df

    def prepare_model_features(self, df: pd.DataFrame, is_training: bool = True) -> np.ndarray:
        """
        Prepare dataset for ML model training or prediction.

        Args:
            df (pd.DataFrame): Input invoice dataset
            is_training (bool): Flag indicating training or inference mode

        Returns:
            np.ndarray: Scaled feature array ready for model input
        """
        df = self.create_advanced_features(df)

        # Define features for the model
        numerical_features = [
            'invoice_amount', 'customer_credit_score', 'due_days',
            'avg_payment_delay_history', 'payment_consistency',
            'days_until_month_end', 'amount_vs_customer_avg'
        ]
        categorical_features = ['customer_industry', 'amount_category', 'credit_category']

        # Filter features that actually exist in the dataset
        available_numerical = [f for f in numerical_features if f in df.columns]
        available_categorical = [f for f in categorical_features if f in df.columns]

        # Prepare numerical features
        X_numerical = df[available_numerical]

        # Encode categorical features
        X_categorical_encoded = []
        for col in available_categorical:
            if is_training:
                self.encoders[col] = LabelEncoder()
                encoded = self.encoders[col].fit_transform(df[col].astype(str))
            else:
                # Handle unseen categories in prediction
                unseen_mask = ~df[col].astype(str).isin(self.encoders[col].classes_)
                if unseen_mask.any():
                    df.loc[unseen_mask, col] = 'Unknown'
                encoded = self.encoders[col].transform(df[col].astype(str))

            X_categorical_encoded.append(encoded)

        # Combine numerical and categorical features
        if X_categorical_encoded:
            X_categorical = np.column_stack(X_categorical_encoded)
            X_combined = np.column_stack([X_numerical.values, X_categorical])
            self.feature_names = available_numerical + available_categorical
        else:
            X_combined = X_numerical.values
            self.feature_names = available_numerical

        # Scale all features
        if is_training:
            X_scaled = self.scaler.fit_transform(X_combined)
        else:
            X_scaled = self.scaler.transform(X_combined)

        return X_scaled

    def get_feature_names(self) -> list:
        """Return the list of feature names used in the model."""
        return self.feature_names
