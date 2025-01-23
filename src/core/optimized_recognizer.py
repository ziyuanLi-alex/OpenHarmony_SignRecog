from ultralytics import YOLO
from src.core.temporal_buffer import TemporalBuffer
from src.utils.performance_monitor import PerformanceMonitor
import torch


class OptimizedSignLanguageRecognizer:
    def __init__(self):
        # 1. 双流模型选择轻量级backbone
        self.rgb_model = YOLO('yolov11m.pt')  
        self.depth_model = YOLO('yolov11m.pt')
        
        # 2. 适度的时序缓存
        self.temporal_buffer = TemporalBuffer(
            buffer_size=15,  # 0.5秒,假设30fps
            max_memory_usage='4GB'  # 控制内存使用
        )
        
        # 3. 轻量级的自适应融合
        self.adaptive_fusion = LightAdaptiveFusion(
            use_simplified_quality_estimation=True  # 使用简化质量评估
        )
        
        # 4. 性能监控
        self.performance_monitor = PerformanceMonitor(
            gpu_memory_threshold='3GB',  # GPU内存告警阈值
            ram_threshold='16GB',        # RAM使用告警阈值
            fps_target=30               # 目标帧率
        )

    def process_frame(self, rgb_frame, depth_frame):
        # 1. 性能检查
        if self.performance_monitor.should_skip_frame():
            return self.fast_forward(rgb_frame)  # 跳帧处理
            
        # 2. 特征提取和融合
        with torch.cuda.amp.autocast('cuda'):  # 使用半精度加速
            rgb_features = self.rgb_model(rgb_frame)
            depth_features = self.depth_model(depth_frame)
        
        # 3. 时序处理(简化版)
        self.temporal_buffer.update(rgb_features, depth_features)
        temporal_features = self.temporal_buffer.get_averaged_features()
        
        # 4. 轻量级融合
        final_prediction = self.adaptive_fusion.fuse(
            rgb_features,
            depth_features,
            temporal_features
        )
        
        return final_prediction
