# config.py - Humanized Configuration Management
import os
import yaml
import logging
from dataclasses import dataclass
from typing import Dict, Any

# -----------------------------
# Dataclasses for Configuration
# -----------------------------
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

# -----------------------------
# Main Config Class
# -----------------------------
class Config:
    """Central configuration manager for the application."""

    def __init__(self):
        # Initialize sub-configs
        self.database = DatabaseConfig()
        self.model = ModelConfig()
        self.spark = SparkConfig()
        self.app = AppConfig()
        
        # Logger
        self.logger = logging.getLogger(__name__)

        # Load environment variables and YAML file
        self.load_from_env()
        self.load_from_yaml()

    # -----------------------------
    # Load from Environment Variables
    # -----------------------------
    def load_from_env(self):
        """Override config with environment variables if available."""
        try:
            self.database.url = os.getenv("DATABASE_URL", self.database.url)
            self.model.use_deep_learning = os.getenv("USE_DEEP_LEARNING", str(self.model.use_deep_learning)).lower() == "true"
            self.model.model_path = os.getenv("MODEL_PATH", self.model.model_path)
            self.spark.master = os.getenv("SPARK_MASTER", self.spark.master)
            self.app.port = int(os.getenv("STREAMLIT_SERVER_PORT", self.app.port))
            self.app.log_level = os.getenv("LOG_LEVEL", self.app.log_level)
        except Exception as e:
            self.logger.warning(f"Environment configuration loading failed: {e}")

    # -----------------------------
    # Load from YAML File
    # -----------------------------
    def load_from_yaml(self, filepath: str = "config.yaml"):
        """Load configuration values from a YAML file."""
        if not os.path.exists(filepath):
            self.logger.info("No YAML configuration file found, skipping.")
            return

        try:
            with open(filepath, "r") as file:
                yaml_config = yaml.safe_load(file)

            # Update each sub-config
            for key, sub_config in [("database", self.database),
                                    ("model", self.model),
                                    ("spark", self.spark),
                                    ("app", self.app)]:
                if key in yaml_config:
                    self._update_from_dict(sub_config, yaml_config[key])

            self.logger.info("✅ Configuration loaded from YAML file")

        except Exception as e:
            self.logger.warning(f"YAML configuration loading failed: {e}")

    # -----------------------------
    # Helper: Update Dataclass from Dict
    # -----------------------------
    @staticmethod
    def _update_from_dict(config_obj, config_dict: Dict[str, Any]):
        """Update a dataclass instance from a dictionary."""
        for key, value in config_dict.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)

    # -----------------------------
    # Convert Config to Dict
    # -----------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Convert full configuration to a dictionary."""
        return {
            "database": vars(self.database),
            "model": vars(self.model),
            "spark": vars(self.spark),
            "app": vars(self.app),
        }

    # -----------------------------
    # Save Config to YAML
    # -----------------------------
    def save_to_yaml(self, filepath: str = "config.yaml"):
        """Persist current configuration to a YAML file."""
        try:
            with open(filepath, "w") as file:
                yaml.dump(self.to_dict(), file, default_flow_style=False)
            self.logger.info(f"✅ Configuration saved to {filepath}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save configuration: {e}")

# -----------------------------
# Global Configuration Instance
# -----------------------------
config = Config()
