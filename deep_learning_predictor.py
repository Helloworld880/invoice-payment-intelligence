# deep_learning_predictor.py - Advanced ML with Deep Learning
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
import joblib
import os

class DeepLearningPredictor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def build_model(self, input_dim):
        """Build a neural network model for payment delay prediction"""
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
            self.logger.error(f"❌ Failed to build model: {str(e)}")
            return False
    
    def preprocess_data(self, X, y=None, training=False):
        """Preprocess data for neural network"""
        try:
            # Handle categorical variables
            X_processed = X.copy()
            
            # Encode categorical features
            if 'customer_industry' in X_processed.columns:
                if training:
                    X_processed['industry_encoded'] = self.label_encoder.fit_transform(X_processed['customer_industry'])
                else:
                    X_processed['industry_encoded'] = self.label_encoder.transform(X_processed['customer_industry'])
                X_processed = X_processed.drop('customer_industry', axis=1)
            
            # Select numerical features
            numerical_features = ['invoice_amount', 'customer_credit_score', 'due_days', 
                                'avg_payment_delay_history', 'payment_consistency', 'industry_encoded']
            X_numerical = X_processed[numerical_features]
            
            # Scale features
            if training:
                X_scaled = self.scaler.fit_transform(X_numerical)
            else:
                X_scaled = self.scaler.transform(X_numerical)
            
            return X_scaled
            
        except Exception as e:
            self.logger.error(f"❌ Data preprocessing failed: {str(e)}")
            return None
    
    def train(self, X, y, epochs=100, validation_split=0.2):
        """Train the deep learning model"""
        try:
            # Preprocess data
            X_processed = self.preprocess_data(X, y, training=True)
            
            if X_processed is None:
                return None
            
            # Build model
            self.build_model(X_processed.shape[1])
            
            # Train model
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
            self.logger.error(f"❌ Model training failed: {str(e)}")
            return None
    
    def predict(self, X):
        """Make predictions using the deep learning model"""
        if not self.is_trained or self.model is None:
            self.logger.warning("Model not trained, returning random predictions")
            return np.random.random(len(X))
        
        try:
            X_processed = self.preprocess_data(X, training=False)
            
            if X_processed is None:
                return np.random.random(len(X))
            
            predictions = self.model.predict(X_processed, verbose=0)
            return predictions.flatten()
            
        except Exception as e:
            self.logger.error(f"❌ Prediction failed: {str(e)}")
            return np.random.random(len(X))
    
    def predict_proba(self, X):
        """Predict probability scores"""
        return self.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
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
            self.logger.error(f"❌ Model evaluation failed: {str(e)}")
            return {}
    
    def save_model(self, filepath='models/deep_learning_model.h5'):
        """Save the trained model"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.model.save(filepath)
            
            # Save preprocessing objects
            joblib.dump(self.scaler, 'models/scaler.joblib')
            joblib.dump(self.label_encoder, 'models/label_encoder.joblib')
            
            self.logger.info(f"✅ Model saved to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to save model: {str(e)}")
            return False
    
    def load_model(self, filepath='models/deep_learning_model.h5'):
        """Load a pre-trained model"""
        try:
            self.model = tf.keras.models.load_model(filepath)
            self.scaler = joblib.load('models/scaler.joblib')
            self.label_encoder = joblib.load('models/label_encoder.joblib')
            self.is_trained = True
            self.logger.info(f"✅ Model loaded from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {str(e)}")
            return False

# Global deep learning predictor instance
dl_predictor = DeepLearningPredictor()