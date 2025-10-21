import yaml
import os
from typing import Dict, Any, List

class Config:
    """Application configuration manager"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "config.yaml"
        self._config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        default_config = {
            'app': {
                'name': 'Invoice Payment Intelligence',
                'version': '2.0.0',
                'debug': False,
                'log_level': 'INFO'
            },
            'model': {
                'paths': [
                    'src/models/saved_models/payment_predictor.joblib',
                    'models/saved_models/payment_predictor.joblib',
                    'payment_predictor.joblib'
                ],
                'risk_thresholds': {
                    'low': 0.4,
                    'medium': 0.7,
                    'high': 1.0
                },
                'demo_mode': True
            },
            'data': {
                'required_columns': [
                    'customer_industry', 
                    'customer_credit_score', 
                    'invoice_amount'
                ],
                'optional_columns': [
                    'due_days',
                    'avg_payment_delay_history', 
                    'payment_consistency',
                    'invoice_id',
                    'customer_id'
                ],
                'max_file_size_mb': 100,
                'allowed_file_types': ['csv']
            },
            'ui': {
                'theme': 'light',
                'page_size': 25,
                'auto_refresh': True
            }
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as file:
                    user_config = yaml.safe_load(file)
                    return self._merge_configs(default_config, user_config)
            return default_config
        except Exception as e:
            print(f"Warning: Could not load config file. Using defaults. Error: {e}")
            return default_config
    
    def _merge_configs(self, default: Dict, user: Dict) -> Dict:
        """Recursively merge user config with defaults"""
        merged = default.copy()
        for key, value in user.items():
            if isinstance(value, dict) and key in merged:
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        keys = key.split('.')
        value = self._config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    @property
    def required_columns(self) -> List[str]:
        return self.get('data.required_columns')
    
    @property
    def risk_thresholds(self) -> Dict[str, float]:
        return self.get('model.risk_thresholds')
    
    @property
    def model_paths(self) -> List[str]:
        return self.get('model.paths')
