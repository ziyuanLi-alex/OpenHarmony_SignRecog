from ultralytics import YOLO
import yaml
import os
import logging
from typing import Optional, Dict, Any
import torch


# noinspection SpellCheckingInspection
class ModelTrainer:
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the YOLO model trainer.

        Args:
            config_path: Path to the training configuration file
        """
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        self.model = None
        self.results = None

    # noinspection PyMethodMayBeStatic
    def _setup_logging(self) -> logging.Logger:
        """Configure logging for the training process."""
        logger = logging.getLogger('ModelTrainer')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load training configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Dictionary containing configuration parameters
        """
        default_config = {
            'model_path': 'yolo11m.pt',
            'data_yaml': './dataset/data.yaml',
            'epochs': 80,
            'imgsz': 640,
            'patience': 15,
            'batch_size': 16,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'save_period': 10,
            'workers': 4
        }

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                default_config.update(user_config)
            except Exception as e:
                self.logger.error(f"Error loading config file: {e}")
                self.logger.info("Using default configuration")

        return default_config

    def initialize_model(self) -> None:
        """Initialize the YOLO model with the specified configuration."""
        try:
            self.model = YOLO(self.config['model_path'])
            self.logger.info(f"Model initialized from {self.config['model_path']}")
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            raise

    def train(self) -> None:
        """Train the model using the specified configuration."""
        if self.model is None:
            self.initialize_model()

        try:
            self.logger.info("Starting training...")
            self.results = self.model.train(
                data=self.config['data_yaml'],
                epochs=self.config['epochs'],
                imgsz=self.config['imgsz'],
                patience=self.config['patience'],
                batch=self.config['batch_size'],
                device=self.config['device'],
                workers=self.config['workers'],
                save_period=self.config['save_period']
            )
            self.logger.info("Training completed successfully")

        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise

    def validate(self) -> None:
        """Validate the trained model."""
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")

        try:
            self.logger.info("Starting validation...")
            results = self.model.val()
            self.logger.info(f"Validation completed: mAP@0.5 = {results.box.map50}")
        except Exception as e:
            self.logger.error(f"Error during validation: {e}")
            raise

    def export_model(self, format: str = 'onnx') -> None:
        """
        Export the trained model to specified format.

        Args:
            format: Export format ('onnx', 'torchscript', etc.)
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")

        try:
            self.logger.info(f"Exporting model to {format}...")
            self.model.export(format=format)
            self.logger.info(f"Model exported successfully")
        except Exception as e:
            self.logger.error(f"Error exporting model: {e}")
            raise


def main():
    # Example usage
    trainer = ModelTrainer()
    trainer.initialize_model()
    trainer.train()
    trainer.validate()
    trainer.export_model('onnx')

if __name__ == "__main__":
    main()