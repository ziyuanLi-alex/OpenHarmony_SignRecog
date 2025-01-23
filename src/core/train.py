import os
from pathlib import Path
import yaml
import logging
from typing import Dict, Any
import sys
import argparse  # 新增：用于处理命令行参数

ROOT_DIR = Path(__file__).resolve().parents[2]
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

def verify_directory_structure(project_root: Path, dataset_type: str, logger: logging.Logger) -> None:
    """验证必要的目录结构"""
    required_dirs = [
        f'dataset/{dataset_type}/train/images',
        f'dataset/{dataset_type}/valid/images',
        f'dataset/{dataset_type}/test/images',
        'configs'
    ]
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            logger.warning(f"Directory not found: {full_path}")
            full_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {full_path}")

def setup_training_config(project_root: Path, dataset_type: str, logger: logging.Logger) -> Dict[str, str]:
    """设置训练配置并返回相关路径"""
    # 验证并创建必要的目录结构
    verify_directory_structure(project_root, dataset_type, logger)
    
    # 确定配置文件路径
    config_dir = project_root / 'configs'
    data_yaml_path = config_dir / f'data_{dataset_type}.yaml'
    model_config_path = config_dir / f'model_config_{dataset_type}.yaml'
    
    # 数据集配置
    data_config = {
        'path': str(project_root / f'dataset/{dataset_type}'),
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
    
    # 复制并更新model_config
    try:
        # 读取基础配置
        base_config_path = project_root / 'configs' / 'model_config.yaml'
        with open(base_config_path, 'r', encoding='utf-8') as f:
            model_config = yaml.safe_load(f)
        
        # 更新data_yaml路径为相对于项目根目录的路径
        model_config['data_yaml'] = str(data_yaml_path.relative_to(project_root))
        
        # 使用基础模型路径（如果配置中没有指定，则使用默认值）
        if 'model_path' not in model_config:
            model_config['model_path'] = 'yolo11l.pt'  # 或 'yolov8m.pt', 'yolov8l.pt' 等
        
        # 训练结果将保存在特定数据集的目录中
        model_config['project'] = 'runs/detect'
        model_config['name'] = f'train_{dataset_type}'
        
        # 保存特定数据集的配置
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



def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train YOLO model on RGB or Depth dataset')
    parser.add_argument('--dataset', 
                       type=str, 
                       choices=['rgb', 'depth'],
                       default='rgb',
                       help='选择训练数据集类型 (rgb 或 depth)')
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    dataset_type = args.dataset
    
    # 设置日志
    logger = setup_logging()
    logger.info(f"Selected dataset type: {dataset_type}")
    
    try:
        # 获取项目根目录
        project_root = Path(__file__).parent.parent.parent
        logger.info(f"Project root directory: {project_root}")
        
        # 设置配置文件
        config_paths = setup_training_config(project_root, dataset_type, logger)
        logger.info(f"Configuration files prepared: {config_paths}")
        
        # 初始化训练器
        trainer = ModelTrainer(config_paths['model_config'])
        
        # 训练流程
        logger.info("Initializing model...")
        trainer.initialize_model()
        
        logger.info(f"Starting training on {dataset_type} dataset...")
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