# src/core/__init__.py
"""
Core functionality for YOLO tracking and model training.
"""
from .yolo_tracker import YOLOTracker
from .model_trainer import ModelTrainer

__all__ = ['YOLOTracker', 'ModelTrainer']