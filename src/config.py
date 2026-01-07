"""
Configuration management for HGPS Prediction System.
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model configuration settings."""
    # Data settings
    n_hgps_train: int = 100
    n_controls_train: int = 400
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42

    # QML settings
    n_qubits: int = 6
    qml_feature_dim: int = 6
    qsvm_shots: int = 1024
    qnn_max_iter: int = 100

    # CNN settings
    image_size: tuple = (224, 224)
    cnn_backbone: str = "resnet18"
    cnn_pretrained: bool = True

    # Training settings
    batch_size: int = 32
    learning_rate: float = 0.001
    n_epochs: int = 50
    early_stopping_patience: int = 10

    # Classical ML settings
    rf_n_estimators: int = 100
    xgb_n_estimators: int = 100
    svm_kernel: str = "rbf"


@dataclass
class PathConfig:
    """Path configuration settings."""
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    @property
    def models_dir(self) -> Path:
        return self.base_dir / "models"

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "logs"

    @property
    def results_dir(self) -> Path:
        return self.base_dir / "results"

    def ensure_dirs(self):
        """Create all necessary directories."""
        for dir_path in [self.models_dir, self.data_dir, self.logs_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class APIConfig:
    """API configuration settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 4

    # Security
    api_key: Optional[str] = None
    cors_origins: list = field(default_factory=lambda: ["*"])
    rate_limit: int = 100  # requests per minute

    # Timeouts
    request_timeout: int = 30
    model_load_timeout: int = 120


@dataclass
class Settings:
    """Main settings class combining all configurations."""
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    api: APIConfig = field(default_factory=APIConfig)

    # Environment
    environment: str = "development"
    log_level: str = "INFO"
    device: str = "auto"  # auto, cpu, cuda

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables."""
        settings = cls()

        # Override from environment
        settings.environment = os.getenv("ENVIRONMENT", "development")
        settings.log_level = os.getenv("LOG_LEVEL", "INFO")
        settings.device = os.getenv("DEVICE", "auto")

        # API settings
        settings.api.host = os.getenv("API_HOST", "0.0.0.0")
        settings.api.port = int(os.getenv("API_PORT", "8000"))
        settings.api.debug = os.getenv("DEBUG", "false").lower() == "true"
        settings.api.api_key = os.getenv("API_KEY")

        # Model settings
        settings.model.random_state = int(os.getenv("RANDOM_STATE", "42"))

        return settings

    def is_production(self) -> bool:
        return self.environment == "production"


# Global settings instance
settings = Settings.from_env()
