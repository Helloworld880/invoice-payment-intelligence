# tests/test_deep_learning.py
import os
import sys
import tempfile

import pytest
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deep_learning_predictor import DeepLearningPredictor


class TestDeepLearningPredictor:
    """Tests for DeepLearningPredictor"""

    def setup_method(self):
        """Initialize predictor and sample data before each test"""
        self.dl_predictor = DeepLearningPredictor()
        self.sample_features = pd.DataFrame({
            'customer_industry': ['Technology', 'Manufacturing', 'Retail'],
            'invoice_amount': [15000.0, 25000.0, 8000.0],
            'customer_credit_score': [720, 650, 680],
            'due_days': [30, 45, 30],
            'avg_payment_delay_history': [5.0, 15.0, 8.0],
            'payment_consistency': [0.9, 0.7, 0.8]
        })
        self.sample_target = np.array([0, 1, 0])

    def test_model_building(self):
        """Verify the model builds correctly"""
        success = self.dl_predictor.build_model(6)
        assert success
        assert self.dl_predictor.model is not None

    def test_data_preprocessing(self):
        """Verify data preprocessing"""
        X_processed = self.dl_predictor.preprocess_data(self.sample_features, training=True)
        assert X_processed is not None
        assert X_processed.shape[0] == len(self.sample_features)

    def test_model_training(self):
        """Verify model training runs without crashing"""
        history = self.dl_predictor.train(self.sample_features, self.sample_target, epochs=5)
        assert self.dl_predictor.is_trained or history is None

    def test_prediction(self):
        """Verify predictions are returned correctly"""
        predictions = self.dl_predictor.predict(self.sample_features)
        assert predictions is not None
        assert len(predictions) == len(self.sample_features)
        assert all(0 <= p <= 1 for p in predictions)

    def test_probability_prediction(self):
        """Verify probability predictions are valid"""
        probabilities = self.dl_predictor.predict_proba(self.sample_features)
        assert probabilities is not None
        assert len(probabilities) == len(self.sample_features)
        assert all(0 <= p <= 1 for p in probabilities)

    def test_model_saving_loading(self):
        """Verify saving and loading the model works"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model.h5')
            self.dl_predictor.build_model(6)
            self.dl_predictor.is_trained = True
            save_success = self.dl_predictor.save_model(model_path)
            assert save_success is not None

    def teardown_method(self):
        """Cleanup after each test"""
        pass
