# config.py - Configuration Management
import os
import yaml
from dataclasses import dataclass
from typing import Dict, Any
import logging

@dataclass
class DatabaseConfig:
    url: str = "sqlite:///invoices.db"
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30

@dataclass
class ModelConfig:
    risk_threshold_low: float = 0.4
    risk_threshold_medium: float = 0.7
    risk_threshold_high: float = 1.0
    use_deep_learning: bool = False
    model_path: str = "models/saved_models"

@dataclass
class SparkConfig:
    master: str = "local[*]"
    executor_memory: str = "2g"
    driver_memory: str = "1g"
    shuffle_partitions: int = 200

@dataclass
class AppConfig:
    port: int = 8501
    host: str = "0.0.0.0"
    debug: bool = False
    log_level: str = "INFO"

class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.model = ModelConfig()
        self.spark = SparkConfig()
        self.app = AppConfig()
        self.logger = logging.getLogger(__name__)
        self.load_from_env()
        self.load_from_yaml()
    
    def load_from_env(self):
        """Load configuration from environment variables"""
        try:
            # Database
            if os.getenv('DATABASE_URL'):
                self.database.url = os.getenv('DATABASE_URL')
            
            # Model
            if os.getenv('USE_DEEP_LEARNING'):
                self.model.use_deep_learning = os.getenv('USE_DEEP_LEARNING').lower() == 'true'
            
            if os.getenv('MODEL_PATH'):
                self.model.model_path = os.getenv('MODEL_PATH')
            
            # Spark
            if os.getenv('SPARK_MASTER'):
                self.spark.master = os.getenv('SPARK_MASTER')
            
            # App
            if os.getenv('STREAMLIT_SERVER_PORT'):
                self.app.port = int(os.getenv('STREAMLIT_SERVER_PORT'))
            
            if os.getenv('LOG_LEVEL'):
                self.app.log_level = os.getenv('LOG_LEVEL')
                
        except Exception as e:
            self.logger.warning(f"Environment configuration loading failed: {str(e)}")
    
    def load_from_yaml(self, filepath='config.yaml'):
        """Load configuration from YAML file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as file:
                    yaml_config = yaml.safe_load(file)
                
                # Update configuration from YAML
                if 'database' in yaml_config:
                    self._update_from_dict(self.database, yaml_config['database'])
                if 'model' in yaml_config:
                    self._update_from_dict(self.model, yaml_config['model'])
                if 'spark' in yaml_config:
                    self._update_from_dict(self.spark, yaml_config['spark'])
                if 'app' in yaml_config:
                    self._update_from_dict(self.app, yaml_config['app'])
                    
                self.logger.info("✅ Configuration loaded from YAML file")
                
        except Exception as e:
            self.logger.warning(f"YAML configuration loading failed: {str(e)}")
    
    def _update_from_dict(self, config_obj, config_dict):
        """Update configuration object from dictionary"""
        for key, value in config_dict.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'database': {
                'url': self.database.url,
                'pool_size': self.database.pool_size,
                'max_overflow': self.database.max_overflow,
                'pool_timeout': self.database.pool_timeout
            },
            'model': {
                'risk_threshold_low': self.model.risk_threshold_low,
                'risk_threshold_medium': self.model.risk_threshold_medium,
                'risk_threshold_high': self.model.risk_threshold_high,
                'use_deep_learning': self.model.use_deep_learning,
                'model_path': self.model.model_path
            },
            'spark': {
                'master': self.spark.master,
                'executor_memory': self.spark.executor_memory,
                'driver_memory': self.spark.driver_memory,
                'shuffle_partitions': self.spark.shuffle_partitions
            },
            'app': {
                'port': self.app.port,
                'host': self.app.host,
                'debug': self.app.debug,
                'log_level': self.app.log_level
            }
        }
    
    def save_to_yaml(self, filepath='config.yaml'):
        """Save configuration to YAML file"""
        try:
            with open(filepath, 'w') as file:
                yaml.dump(self.to_dict(), file, default_flow_style=False)
            self.logger.info(f"✅ Configuration saved to {filepath}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save configuration: {str(e)}")

# Global configuration instance
config = Config()