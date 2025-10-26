import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    mean_absolute_error, mean_squared_error
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class ModelTrainer:
    """Train and evaluate invoice payment delay prediction models (classification & regression)."""
    
    def __init__(self):
        self.classifier = RandomForestClassifier(random_state=42)
        self.regressor = RandomForestRegressor(random_state=42)
        self.feature_importance = None

    def train_models(self, df):
        """Train both classification and regression models."""
        print("🤖 Starting model training...")

        # --- Prepare features ---
        X = df.drop(columns=[
            'DelayStatus', 'DelayDays',
            'invoice_id', 'customer_id',
            'issue_date', 'due_date', 'payment_date'
        ], errors='ignore')

        # One-hot encode categorical features
        X = pd.get_dummies(X, drop_first=True)

        # Targets
        y_class = df['DelayStatus']
        y_reg = df['DelayDays']

        # Split dataset
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
            X, y_class, y_reg, test_size=0.2, random_state=42
        )

        # --- Train models ---
        self.classifier.fit(X_train, y_class_train)
        self.regressor.fit(X_train, y_reg_train)

        # --- Evaluate models ---
        self.evaluate_models(X_test, y_class_test, y_reg_test)

        # --- Compute feature importance ---
        try:
            self.feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': self.classifier.feature_importances_
            }).sort_values(by='Importance', ascending=False)
        except Exception as e:
            print(f"⚠️ Could not compute feature importance: {e}")
            self.feature_importance = None

    def evaluate_models(self, X_test, y_class_test, y_reg_test):
        """Evaluate classification and regression models with metrics and plots."""
        # --- Classification evaluation ---
        y_class_pred = self.classifier.predict(X_test)
        y_class_proba = self.classifier.predict_proba(X_test)[:, 1]

        print("📊 Classification Model Performance:")
        print(classification_report(y_class_test, y_class_pred))
        print(f"ROC-AUC Score: {roc_auc_score(y_class_test, y_class_proba):.3f}\n")

        # --- Regression evaluation ---
        y_reg_pred = self.regressor.predict(X_test)
        mae = mean_absolute_error(y_reg_test, y_reg_pred)
        rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))

        print("📈 Regression Model Performance:")
        print(f"Mean Absolute Error: {mae:.2f} days")
        print(f"Root Mean Squared Error: {rmse:.2f} days")

        # --- Generate evaluation plots ---
        self.create_evaluation_plots(y_class_test, y_class_proba, y_reg_test, y_reg_pred)

    def create_evaluation_plots(self, y_class_test, y_class_proba, y_reg_test, y_reg_pred):
        """Create and save top feature importance plot."""
        plots_dir = "models/plots"
        os.makedirs(plots_dir, exist_ok=True)

        if self.feature_importance is not None and not self.feature_importance.empty:
            top_features = self.feature_importance.head(10)
            plt.figure(figsize=(8, 5))
            plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
            plt.gca().invert_yaxis()
            plt.title("Top 10 Important Features")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "top_features.png"))
            plt.close()
            print(f"✅ Feature importance plot saved: {plots_dir}/top_features.png")
        else:
            print("⚠️ Skipped feature importance plot — no data available.")

    def save_models(self, path):
        """Save trained models and feature importance."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'classifier': self.classifier,
            'regressor': self.regressor,
            'feature_importance': self.feature_importance
        }, path)
        print(f"💾 Models saved at: {path}")


if __name__ == "__main__":
    print("🤖 Starting full training workflow...")

    # Load dataset
    dataset_path = "data/raw/invoice_data_cleaned.csv"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        exit()

    df = pd.read_csv(dataset_path)
    print(f"✅ Dataset loaded: {dataset_path} | Shape: {df.shape}")

    # Initialize trainer and train models
    trainer = ModelTrainer()
    trainer.train_models(df)

    # Save models
    trainer.save_models("models/model.pkl")

    print("✅ Training complete. Models ready for deployment!")
