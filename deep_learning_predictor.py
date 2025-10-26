# deep_learning_predictor.py - Humanized Deep Learning Predictor
import os
import logging
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DeepLearningPredictor:
    """
    Deep learning model for invoice payment delay prediction.
    Handles preprocessing, training, evaluation, saving, and loading.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False

    # -----------------------------
    # Model Building
    # -----------------------------
    def build_model(self, input_dim: int) -> bool:
        """Build a neural network model."""
        try:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            self.model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            self.logger.info("✅ Neural network model built successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to build model: {e}")
            return False

    # -----------------------------
    # Data Preprocessing
    # -----------------------------
    def preprocess_data(self, X: pd.DataFrame, y=None, training: bool = False) -> np.ndarray | None:
        """
        Preprocess input data for training or prediction.
        Encodes categorical features and scales numerical features.
        """
        try:
            X_processed = X.copy()

            # Encode categorical feature: customer_industry
            if 'customer_industry' in X_processed.columns:
                if training:
                    X_processed['industry_encoded'] = self.label_encoder.fit_transform(
                        X_processed['customer_industry']
                    )
                else:
                    X_processed['industry_encoded'] = self.label_encoder.transform(
                        X_processed['customer_industry']
                    )
                X_processed.drop('customer_industry', axis=1, inplace=True)

            # Select numerical features
            numerical_features = [
                'invoice_amount', 'customer_credit_score', 'due_days',
                'avg_payment_delay_history', 'payment_consistency', 'industry_encoded'
            ]
            X_numerical = X_processed[numerical_features]

            # Scale features
            X_scaled = self.scaler.fit_transform(X_numerical) if training else self.scaler.transform(X_numerical)
            return X_scaled

        except Exception as e:
            self.logger.error(f"❌ Data preprocessing failed: {e}")
            return None

    # -----------------------------
    # Training
    # -----------------------------
    def train(self, X: pd.DataFrame, y: np.ndarray, epochs: int = 100, validation_split: float = 0.2):
        """Train the deep learning model."""
        try:
            X_processed = self.preprocess_data(X, y, training=True)
            if X_processed is None:
                return None

            self.build_model(X_processed.shape[1])

            history = self.model.fit(
                X_processed, y,
                epochs=epochs,
                validation_split=validation_split,
                batch_size=32,
                verbose=0,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
                ]
            )

            self.is_trained = True
            self.logger.info("✅ Deep learning model trained successfully")
            return history.history

        except Exception as e:
            self.logger.error(f"❌ Model training failed: {e}")
            return None

    # -----------------------------
    # Prediction
    # -----------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the deep learning model."""
        if not self.is_trained or self.model is None:
            self.logger.warning("Model not trained, returning random predictions")
            return np.random.random(len(X))

        try:
            X_processed = self.preprocess_data(X, training=False)
            if X_processed is None:
                return np.random.random(len(X))

            return self.model.predict(X_processed, verbose=0).flatten()

        except Exception as e:
            self.logger.error(f"❌ Prediction failed: {e}")
            return np.random.random(len(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability scores (same as predictions)."""
        return self.predict(X)

    # -----------------------------
    # Evaluation
    # -----------------------------
    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
        """Evaluate model performance on test data."""
        if not self.is_trained:
            return {}

        try:
            X_processed = self.preprocess_data(X_test, training=False)
            evaluation = self.model.evaluate(X_processed, y_test, verbose=0)

            metrics = {
                'loss': evaluation[0],
                'accuracy': evaluation[1],
                'precision': evaluation[2],
                'recall': evaluation[3]
            }
            return metrics

        except Exception as e:
            self.logger.error(f"❌ Model evaluation failed: {e}")
            return {}

    # -----------------------------
    # Save & Load Model
    # -----------------------------
    def save_model(self, filepath: str = 'models/deep_learning_model.h5') -> bool:
        """Save the trained model and preprocessing objects."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.model.save(filepath)
            joblib.dump(self.scaler, 'models/scaler.joblib')
            joblib.dump(self.label_encoder, 'models/label_encoder.joblib')
            self.logger.info(f"✅ Model saved to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to save model: {e}")
            return False

    def load_model(self, filepath: str = 'models/deep_learning_model.h5') -> bool:
        """Load a pre-trained model along with preprocessing objects."""
        try:
            self.model = tf.keras.models.load_model(filepath)
            self.scaler = joblib.load('models/scaler.joblib')
            self.label_encoder = joblib.load('models/label_encoder.joblib')
            self.is_trained = True
            self.logger.info(f"✅ Model loaded from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            return False

# -----------------------------
# Global Predictor Instance
# -----------------------------
dl_predictor = DeepLearningPredictor()
