import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from src.utils.config import Config

class DataValidator:
    """Data validation utilities"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def validate_user_data(self, data: pd.DataFrame) -> Dict[str, any]:
        """Comprehensive data validation"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'cleaned_data': None
        }
        
        try:
            # Check required columns
            missing_columns = self._check_required_columns(data)
            if missing_columns:
                validation_result['is_valid'] = False
                validation_result['errors'].append(
                    f"Missing required columns: {', '.join(missing_columns)}"
                )
                return validation_result
            
            # Validate data types and ranges
            data_errors = self._validate_data_ranges(data)
            validation_result['errors'].extend(data_errors)
            
            # Check for duplicates
            duplicate_checks = self._check_duplicates(data)
            validation_result['warnings'].extend(duplicate_checks)
            
            # Check for outliers
            outlier_warnings = self._check_outliers(data)
            validation_result['warnings'].extend(outlier_warnings)
            
            # Clean data if valid
            if not data_errors:
                validation_result['cleaned_data'] = self._clean_data(data)
            else:
                validation_result['is_valid'] = False
            
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def _check_required_columns(self, data: pd.DataFrame) -> List[str]:
        """Check for missing required columns"""
        required_columns = self.config.required_columns
        return [col for col in required_columns if col not in data.columns]
    
    def _validate_data_ranges(self, data: pd.DataFrame) -> List[str]:
        """Validate data ranges and types"""
        errors = []
        
        # Credit score validation
        if 'customer_credit_score' in data.columns:
            invalid_scores = data[~data['customer_credit_score'].between(300, 850)]
            if len(invalid_scores) > 0:
                errors.append(f"Found {len(invalid_scores)} invalid credit scores (must be 300-850)")
        
        # Invoice amount validation
        if 'invoice_amount' in data.columns:
            negative_amounts = data[data['invoice_amount'] <= 0]
            if len(negative_amounts) > 0:
                errors.append(f"Found {len(negative_amounts)} invoices with zero/negative amounts")
        
        # Payment consistency validation
        if 'payment_consistency' in data.columns:
            invalid_consistency = data[~data['payment_consistency'].between(0, 1)]
            if len(invalid_consistency) > 0:
                errors.append(f"Found {len(invalid_consistency)} invalid payment consistency values (must be 0-1)")
        
        # Due days validation
        if 'due_days' in data.columns:
            invalid_due_days = data[data['due_days'] <= 0]
            if len(invalid_due_days) > 0:
                errors.append(f"Found {len(invalid_due_days)} invalid due days (must be positive)")
        
        return errors
    
    def _check_duplicates(self, data: pd.DataFrame) -> List[str]:
        """Check for duplicate records"""
        warnings = []
        
        # Check for duplicate invoice IDs if present
        if 'invoice_id' in data.columns:
            duplicates = data[data.duplicated('invoice_id', keep=False)]
            if len(duplicates) > 0:
                warnings.append(f"Found {len(duplicates)} duplicate invoice IDs")
        
        # Check for duplicate rows
        duplicate_rows = data[data.duplicated()]
        if len(duplicate_rows) > 0:
            warnings.append(f"Found {len(duplicate_rows)} duplicate rows")
        
        return warnings
    
    def _check_outliers(self, data: pd.DataFrame) -> List[str]:
        """Check for potential outliers"""
        warnings = []
        
        if 'invoice_amount' in data.columns:
            Q1 = data['invoice_amount'].quantile(0.25)
            Q3 = data['invoice_amount'].quantile(0.75)
            IQR = Q3 - Q1
            outlier_threshold = Q3 + 1.5 * IQR
            
            outliers = data[data['invoice_amount'] > outlier_threshold]
            if len(outliers) > 0:
                warnings.append(f"Found {len(outliers)} potential invoice amount outliers")
        
        return warnings
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data for processing"""
        cleaned_data = data.copy()
        
        # Fill missing optional columns with defaults
        defaults = {
            'due_days': 30,
            'avg_payment_delay_history': 10.0,
            'payment_consistency': 0.8
        }
        
        for col, default_val in defaults.items():
            if col in cleaned_data.columns:
                cleaned_data[col] = cleaned_data[col].fillna(default_val)
            else:
                cleaned_data[col] = default_val
        
        return cleaned_data
# src/utils/validators.py

class InputValidator:
    """Validate single invoice input"""

    def __init__(self, config):
        self.config = config

    def validate_single_input(self, input_data: dict):
        # Assign defaults for optional fields
        if 'payment_consistency' not in input_data:
            input_data['payment_consistency'] = 1.0  # default 100% consistent
        if 'due_days' not in input_data:
            input_data['due_days'] = 30  # default 30 days

        # Simple validation logic (extend as needed)
        errors = []
        if input_data['invoice_amount'] <= 0:
            errors.append("Invoice amount must be positive")
        if not 300 <= input_data['customer_credit_score'] <= 850:
            errors.append("Credit score must be between 300 and 850")
        if not 0 <= input_data['payment_consistency'] <= 1:
            errors.append("Payment consistency must be between 0 and 1")

        is_valid = len(errors) == 0
        return is_valid, errors
