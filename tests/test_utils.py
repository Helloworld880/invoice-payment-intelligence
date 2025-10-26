# tests/test_utils.py
import sys
import os
import pandas as pd
import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.config import Config
from src.utils.helpers import format_currency, calculate_financial_impact, create_demo_predictions
from src.utils.validators import DataValidator, InputValidator


class TestConfig:
    """Tests for configuration management"""

    def test_config_initialization(self):
        config = Config()
        assert config is not None
        assert isinstance(config._config, dict)

    def test_config_get_method(self):
        config = Config()
        assert config.get('app.name') is not None
        assert config.get('non.existent.key', 'default') == 'default'

    def test_required_columns_property(self):
        config = Config()
        required_columns = config.required_columns
        assert isinstance(required_columns, list)
        assert 'customer_industry' in required_columns
        assert 'customer_credit_score' in required_columns
        assert 'invoice_amount' in required_columns


class TestHelpers:
    """Tests for helper functions"""

    def test_format_currency(self):
        results = [
            format_currency(1000),
            format_currency(500.50),
            format_currency(15000),
            format_currency(2500000)
        ]
        for r in results:
            assert isinstance(r, str)
            assert "$" in r

    def test_calculate_financial_impact(self):
        test_data = pd.DataFrame({
            'risk_level': ['High', 'High', 'Low', 'Medium'],
            'invoice_amount': [10000, 20000, 5000, 8000],
            'predicted_delay_days': [10, 15, 2, 5]
        })
        result = calculate_financial_impact(test_data)
        assert isinstance(result, dict)
        for key in ['opportunity_cost', 'estimated_savings', 'high_risk_amount', 'avg_high_risk_delay']:
            assert key in result

    def test_create_demo_predictions(self):
        test_data = pd.DataFrame({
            'customer_credit_score': [700, 550, 650],
            'customer_industry': ['Technology', 'Construction', 'Retail'],
            'invoice_amount': [10000, 50000, 15000],
            'avg_payment_delay_history': [5.0, 20.0, 10.0],
            'payment_consistency': [0.9, 0.6, 0.8]
        })
        predictions = create_demo_predictions(test_data)
        assert 'delay_probs' in predictions
        assert 'predicted_delays' in predictions
        assert len(predictions['delay_probs']) == 3
        assert len(predictions['predicted_delays']) == 3


class TestDataValidator:
    """Tests for data validation"""

    def test_validate_user_data_success(self):
        config = Config()
        validator = DataValidator(config)
        valid_data = pd.DataFrame({
            'customer_industry': ['Technology', 'Manufacturing'],
            'customer_credit_score': [700, 650],
            'invoice_amount': [10000, 15000]
        })
        result = validator.validate_user_data(valid_data)
        assert result['is_valid'] is True
        assert len(result['errors']) == 0
        assert result['cleaned_data'] is not None


class TestInputValidator:
    """Tests for single input validation"""

    def test_validate_single_input_success(self):
        config = Config()
        validator = InputValidator(config)
        valid_input = {
            'invoice_amount': 10000,
            'customer_credit_score': 700,
            'payment_consistency': 0.8,
            'due_days': 30
        }
        is_valid, errors = validator.validate_single_input(valid_input)
        assert is_valid is True
        assert len(errors) == 0
