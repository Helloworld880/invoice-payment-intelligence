# Model_trainer.py - Fixed version
import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

class InvoicePredictor:
    def __init__(self):
        self.classifier = None
        self.regressor = None
        self.feature_importance = None

    def train_models(self, df):
        """Train classification and regression models - FIXED"""
        print("🤖 Training machine learning models...")

        # Prepare features - exclude target and non-feature columns
        exclude_cols = ['DelayStatus', 'DelayDays', 'invoice_id', 'customer_id', 
                       'issue_date', 'due_date', 'payment_date', 'payment_delay_days', 'is_delayed']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols]
        
        # Convert categorical columns
        X_encoded = pd.get_dummies(X, drop_first=True)
        
        # Define targets
        y_class = df['DelayStatus']
        y_reg = df['DelayDays']

        # Train/test split
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
            X_encoded, y_class, y_reg, test_size=0.2, random_state=42
        )

        # Initialize and train models
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        
        self.classifier.fit(X_train, y_class_train)
        self.regressor.fit(X_train, y_reg_train)

        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_encoded.columns,
            'importance': self.classifier.feature_importances_
        }).sort_values(by='importance', ascending=False)

        # Evaluate models
        self.evaluate_models(X_test, y_class_test, y_reg_test)

        # Return trained model and feature columns for saving
        return self.classifier, list(X_encoded.columns)

    def evaluate_models(self, X_test, y_class_test, y_reg_test):
        """Evaluate trained models"""
        # Classification evaluation
        y_class_pred = self.classifier.predict(X_test)
        y_class_proba = self.classifier.predict_proba(X_test)[:, 1]
        
        print("\n📊 Classification Model Performance:")
        print(classification_report(y_class_test, y_class_pred))
        print(f"ROC-AUC Score: {roc_auc_score(y_class_test, y_class_proba):.3f}")

        # Regression evaluation
        y_reg_pred = self.regressor.predict(X_test)
        mae = mean_absolute_error(y_reg_test, y_reg_pred)
        rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
        
        print("\n📈 Regression Model Performance:")
        print(f"Mean Absolute Error: {mae:.2f} days")
        print(f"Root Mean Squared Error: {rmse:.2f} days")

        # Create evaluation plots
        self.create_evaluation_plots()

    def create_evaluation_plots(self):
        """Save feature importance plot"""
        if self.feature_importance is not None and not self.feature_importance.empty:
            plots_dir = "models/plots"
            os.makedirs(plots_dir, exist_ok=True)
            
            top_features = self.feature_importance.head(10)
            plt.figure(figsize=(10, 6))
            plt.barh(top_features['feature'], top_features['importance'], color='skyblue')
            plt.gca().invert_yaxis()
            plt.xlabel('Feature Importance')
            plt.title('Top 10 Important Features for Payment Delay Prediction')
            plt.tight_layout()
            
            plot_path = os.path.join(plots_dir, "top_features.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Feature importance plot saved: {plot_path}")
        else:
            print("⚠️ No feature importance data available for plotting.")