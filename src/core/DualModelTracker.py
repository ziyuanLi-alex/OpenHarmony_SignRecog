import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any
import yaml
from ultralytics import YOLO
from pathlib import Path
import requests
import sys
import time

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]  # 获取项目根目录
sys.path.append(str(ROOT_DIR))

# 导入Berxel相机相关依赖
from src.devices.BerxelSdkDriver.BerxelHawkFrame import *
from src.devices.BerxelSdkDriver.BerxelHawkDevice import *
from src.devices.BerxelSdkDriver.BerxelHawkContext import *
from src.devices.BerxelSdkDriver.BerxelHawkDefines import *

class DualModelTracker:
    def __init__(self, 
                 model_path: str,  # 保持与BerxelTracker一致的参数
                 config_path: Optional[str] = None,
                 test_mode: bool = False,
                 test_post: bool = False):
        """
        初始化双模型追踪器
        
        Args:
            model_path: 主模型路径（RGB模型）
            config_path: 配置文件路径
            test_mode: 测试模式标志
            test_post: 测试POST请求标志
        """
        # 基础设置
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # 从配置中获取深度模型路径
        depth_model_path = self.config.get('depth_model_path', model_path)
        
        # 初始化模型
        self.rgb_model = YOLO(model_path)
        self.depth_model = YOLO(depth_model_path)
        
        # 相机设置
        self.__context = None
        self.__device = None
        self.__deviceList = []
        
        # 跟踪状态
        self.previous_rgb_class = None
        self.previous_depth_class = None
        self.rgb_stable_count = 0
        self.depth_stable_count = 0
        
        # 最新帧缓存
        self.latest_tracked_frame = None
        
        # 测试相关设置
        self.test_mode = test_mode
        self.test_post = test_post
        self.server_url = self.config.get("server_url", "http://localhost:5000")
        
    def _setup_logging(self) -> logging.Logger:
        """配置日志系统"""
        logger = logging.getLogger('DualModelTracker')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = {
            'server_url': 'http://localhost:5000',
            'required_stable_frames': 3,
            'display_window': True,
            'confidence_threshold': 0.5,
            'fusion_weights': {'rgb': 0.6, 'depth': 0.4}
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                default_config.update(user_config)
            except Exception as e:
                self.logger.error(f"Error loading config: {e}")
        
        return default_config

    def open_device(self) -> bool:
        """初始化并打开Berxel相机"""
        self.__context = BerxelHawkContext()
        if self.__context is None:
            self.logger.error("初始化失败")
            return False

        self.__context.initCamera()
        self.__deviceList = self.__context.getDeviceList()

        if len(self.__deviceList) < 1:
            self.logger.error("未找到设备")
            return False

        self.__device = self.__context.openDevice(self.__deviceList[0])
        return self.__device is not None

    def start_streams(self) -> bool:
        """启动数据流"""
        if self.__device is None:
            return False

        self.__device.setDenoiseStatus(False)
        
        # 设置深度流模式
        frameMode = self.__device.getCurrentFrameMode(
            BerxelHawkStreamType.forward_dict['BERXEL_HAWK_DEPTH_STREAM'])
        self.__device.setFrameMode(
            BerxelHawkStreamType.forward_dict['BERXEL_HAWK_DEPTH_STREAM'], 
            frameMode)

        # 启动RGB和深度流
        ret = self.__device.startStreams(
            BerxelHawkStreamType.forward_dict['BERXEL_HAWK_COLOR_STREAM'] |
            BerxelHawkStreamType.forward_dict['BERXEL_HAWK_DEPTH_STREAM']
        )
        
        return ret == 0

    def capture_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """捕获一帧RGB和深度数据"""
        rgb_frame = None
        depth_frame = None
        
        # 读取RGB帧
        hawkColorFrame = self.__device.readColorFrame(30)
        if hawkColorFrame is not None:
            try:
                colorFrameBuffer = hawkColorFrame.getDataAsUint8()
                rgb_frame = np.ndarray(
                    shape=(hawkColorFrame.getHeight(), 
                          hawkColorFrame.getWidth(), 3),
                    dtype=np.uint8, 
                    buffer=colorFrameBuffer)
            finally:
                self.__device.releaseFrame(hawkColorFrame)
        
        # 读取深度帧
        hawkDepthFrame = self.__device.readDepthFrame(30)
        if hawkDepthFrame is not None:
            try:
                depthFrameBuffer = hawkDepthFrame.getDataAsUint16()
                depth_frame = np.ndarray(
                    shape=(hawkDepthFrame.getHeight(), 
                          hawkDepthFrame.getWidth()),
                    dtype=np.uint16, 
                    buffer=depthFrameBuffer)
            finally:
                self.__device.releaseFrame(hawkDepthFrame)
        
        return rgb_frame, depth_frame

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """
        处理单帧图像（保持与BerxelTracker接口一致）
        此方法主要用于兼容性，实际处理在process_dual_frames中进行
        """
        if self.latest_tracked_frame is not None:
            return self.latest_tracked_frame, self.previous_rgb_class
        return frame, None

    def process_dual_frames(self, rgb_frame: np.ndarray, depth_frame: np.ndarray) -> Tuple[np.ndarray, Optional[str], float]:
        """处理RGB和深度帧"""
        # RGB预测
        rgb_results = self.rgb_model.track(rgb_frame, persist=True)
        rgb_class = None
        rgb_conf = 0.0
        
        if len(rgb_results[0].boxes) > 0:
            rgb_box = rgb_results[0].boxes[0]
            rgb_class_id = int(rgb_box.cls[0].item())
            rgb_class = self.rgb_model.names[rgb_class_id]
            rgb_conf = float(rgb_box.conf[0].item())
            
            if rgb_class == self.previous_rgb_class:
                self.rgb_stable_count += 1
            else:
                self.rgb_stable_count = 0
            self.previous_rgb_class = rgb_class

        # 深度图预处理和预测
        depth_visual = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
        depth_visual = cv2.applyColorMap(depth_visual.astype(np.uint8), cv2.COLORMAP_JET)
        
        depth_results = self.depth_model.track(depth_visual, persist=True)
        depth_class = None
        depth_conf = 0.0
        
        if len(depth_results[0].boxes) > 0:
            depth_box = depth_results[0].boxes[0]
            depth_class_id = int(depth_box.cls[0].item())
            depth_class = self.depth_model.names[depth_class_id]
            depth_conf = float(depth_box.conf[0].item())
            
            if depth_class == self.previous_depth_class:
                self.depth_stable_count += 1
            else:
                self.depth_stable_count = 0
            self.previous_depth_class = depth_class

        # 决策融合
        final_class, confidence = self._fuse_predictions(
            rgb_class, rgb_conf, self.rgb_stable_count,
            depth_class, depth_conf, self.depth_stable_count
        )

        # 可视化
        annotated_frame = rgb_frame.copy()
        self._draw_predictions(annotated_frame, rgb_class, rgb_conf, 
                             depth_class, depth_conf, 
                             final_class, confidence)
        
        self.latest_tracked_frame = annotated_frame
        return annotated_frame, final_class, confidence

    def _fuse_predictions(self, rgb_class, rgb_conf, rgb_stable,
                         depth_class, depth_conf, depth_stable) -> Tuple[Optional[str], float]:
        """融合两个模型的预测结果"""
        if rgb_class == depth_class and rgb_class is not None:
            return rgb_class, max(rgb_conf, depth_conf)
        
        weights = self.config['fusion_weights']
        rgb_score = rgb_conf * weights['rgb'] * (rgb_stable + 1)
        depth_score = depth_conf * weights['depth'] * (depth_stable + 1)
        
        if rgb_score > depth_score and rgb_score > 0:
            return rgb_class, rgb_conf
        elif depth_score > rgb_score and depth_score > 0:
            return depth_class, depth_conf
        
        return None, 0.0

    def _draw_predictions(self, frame, rgb_class, rgb_conf,
                         depth_class, depth_conf,
                         final_class, confidence):
        """在图像上绘制预测结果"""
        if rgb_class:
            cv2.putText(frame, f"RGB: {rgb_class} ({rgb_conf:.2f})",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if depth_class:
            cv2.putText(frame, f"Depth: {depth_class} ({depth_conf:.2f})",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        if final_class:
            cv2.putText(frame, f"Final: {final_class} ({confidence:.2f})",
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def start_tracking(self):
        """开始追踪"""
        if not self.open_device() or not self.start_streams():
            self.logger.error("Failed to initialize device")
            return
            
        self.logger.info("Starting dual model tracking...")
        
        try:
            while True:
                rgb_frame, depth_frame = self.capture_frame()
                if rgb_frame is None or depth_frame is None:
                    continue
                
                tracked_frame, final_class, confidence = self.process_dual_frames(
                    cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR),
                    depth_frame
                )
                
                if self.config['display_window']:
                    cv2.imshow("Dual Model Tracking", tracked_frame)
                    depth_display = ((depth_frame / 10000.) * 255).astype(np.uint8)
                    depth_visual = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                    cv2.imshow("Depth View", depth_visual)
                
                if final_class and confidence > self.config['confidence_threshold']:
                    self.post_class_name(final_class)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            self.cleanup()

    def get_latest_frame(self) -> Any:
        """获取最新处理后的帧"""
        return self.latest_tracked_frame

    def post_class_name(self, class_name: str) -> None:
        """发送识别结果到服务器"""
        url = f"{self.server_url}/recognize"
        data = {"class_name": class_name}

        if self.test_post:
            self.logger.info(f"Pseudo-posting data: {data}")
            return

        try:
            response = requests.post(url, json=data, timeout=1.0)
            if response.status_code != 200:
                self.logger.error(f"Failed to post: {response.status_code}")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error posting data: {e}")

    def cleanup(self) -> None:
        """清理资源"""
        if self.__device:
            self.__device.stopStream(
                BerxelHawkStreamType.forward_dict['BERXEL_HAWK_COLOR_STREAM'] |
                BerxelHawkStreamType.forward_dict['BERXEL_HAWK_DEPTH_STREAM']
            )

        if self.__context and self.__device:
            self.__context.clsoeDevice(self.__device)
            self.__context.destroyCamera()

        cv2.destroyAllWindows()