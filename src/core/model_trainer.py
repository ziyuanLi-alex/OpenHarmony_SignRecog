# noinspection SpellCheckingInspection
from ultralytics import YOLO
import yaml
import os
from pathlib import Path
import logging
from typing import Optional, Dict, Any
import torch
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]  # 获取项目根目录
sys.path.append(str(ROOT_DIR))


class ModelTrainer:
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the YOLO model trainer.

        Args:
            config_path: Path to the training configuration file
        """
        self.logger = self._setup_logging()
        # 保存配置文件路径的根目录
        self.config_root = Path(config_path).parent if config_path else Path.cwd()
        self.config = self._load_config(config_path)
        self.model = None
        self.results = None

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
            'batch_size': 8,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'save_period': 10,
            'workers': 4
        }

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                default_config.update(user_config)
                
                # 确保data_yaml是绝对路径
                if 'data_yaml' in user_config:
                    data_yaml_path =    Path(user_config['data_yaml'])
                    if not data_yaml_path.is_absolute():
                        # 如果是相对路径，则相对于配置文件的位置
                        # data_yaml_path = self.config_root / data_yaml_path
                        data_yaml_path = ROOT_DIR / data_yaml_path
                        default_config['data_yaml'] = str(data_yaml_path)
                        self.logger.info(f"Using data_yaml path: {data_yaml_path}")
                        
                    # 验证data_yaml文件是否存在
                    if not data_yaml_path.exists():
                        self.logger.error(f"data_yaml file not found: {data_yaml_path}")
                        raise FileNotFoundError(f"data_yaml file not found: {data_yaml_path}")
                    
            except Exception as e:
                self.logger.error(f"Error loading config file: {e}")
                self.logger.info("Using default configuration")

        return default_config

    def initialize_model(self) -> None:
        """Initialize the YOLO model with the specified configuration."""
        try:
            # 确保model_path是正确的路径
            model_path = Path(self.config['model_path'])
            if not model_path.is_absolute():
                model_path = self.config_root / model_path
                self.config['model_path'] = str(model_path)

            self.model = YOLO(str(model_path))
            self.logger.info(f"Model initialized from {model_path}")
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            raise

    def train(self) -> None:
        """Train the model using the specified configuration."""
        if self.model is None:
            self.initialize_model()

        try:
            self.logger.info("Starting training...")
            # 确保配置文件存在
            data_yaml_path = Path(self.config['data_yaml'])
            if not data_yaml_path.exists():
                raise FileNotFoundError(f"Data YAML file not found: {data_yaml_path}")
                
            self.logger.info(f"Using data config from: {data_yaml_path}")
            
            self.results = self.model.train(
                data=str(data_yaml_path),
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

    # validate 和 export_model 方法保持不变...

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