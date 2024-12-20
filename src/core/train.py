import os
from pathlib import Path
import yaml
import logging
from typing import Dict, Any
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]  # 获取项目根目录
sys.path.append(str(ROOT_DIR))

from src.core.model_trainer import ModelTrainer

def setup_logging() -> logging.Logger:
    """设置日志配置"""
    logger = logging.getLogger('Training')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def verify_directory_structure(project_root: Path, logger: logging.Logger) -> None:
    """验证必要的目录结构"""
    required_dirs = [
        'dataset/rgb/train/images',
        'dataset/rgb/valid/images',
        'dataset/rgb/test/images',
        'configs'
    ]
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            logger.warning(f"Directory not found: {full_path}")
            full_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {full_path}")

def setup_training_config(project_root: Path, logger: logging.Logger) -> Dict[str, str]:
    """设置训练配置并返回相关路径"""
    # 验证并创建必要的目录结构
    verify_directory_structure(project_root, logger)
    
    # 确定配置文件路径
    config_dir = project_root / 'configs'
    data_yaml_path = config_dir / 'data.yaml'
    model_config_path = config_dir / 'model_config.yaml'
    
    # 数据集配置
    data_config = {
        'path': str(project_root / 'dataset/rgb'),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'names': {
            0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e',
            5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'k',
            10: 'l', 11: 'm', 12: 'n', 13: 'o', 14: 'p',
            15: 'q', 16: 'r', 17: 's', 18: 't', 19: 'u',
            20: 'v', 21: 'w', 22: 'x', 23: 'y'
        }
    }
    
    # 保存数据配置
    try:
        with open(data_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_config, f, sort_keys=False)
        logger.info(f"Data configuration saved to {data_yaml_path}")
    except Exception as e:
        logger.error(f"Failed to save data configuration: {e}")
        raise
    
    # 更新model_config中的data_yaml路径
    try:
        with open(model_config_path, 'r', encoding='utf-8') as f:
            model_config = yaml.safe_load(f)
        
        # 更新data_yaml路径为相对于项目根目录的路径
        model_config['data_yaml'] = str(data_yaml_path.relative_to(project_root))
        
        with open(model_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(model_config, f, sort_keys=False)
        logger.info(f"Model configuration updated in {model_config_path}")
        
    except Exception as e:
        logger.error(f"Failed to update model configuration: {e}")
        raise
    
    return {
        'data_yaml': str(data_yaml_path),
        'model_config': str(model_config_path)
    }

def main():
    # 设置日志
    logger = setup_logging()
    
    try:
        # 获取项目根目录
        project_root = Path(__file__).parent.parent.parent
        logger.info(f"Project root directory: {project_root}")
        
        # 设置配置文件
        config_paths = setup_training_config(project_root, logger)
        logger.info(f"Configuration files prepared: {config_paths}")
        
        # 初始化训练器
        trainer = ModelTrainer(config_paths['model_config'])
        
        # 训练流程
        logger.info("Initializing model...")
        trainer.initialize_model()
        
        logger.info("Starting training...")
        trainer.train()
        
        logger.info("Running validation...")
        trainer.validate()
        
        logger.info("Exporting model to ONNX format...")
        trainer.export_model('onnx')
        
        logger.info("Training process completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()