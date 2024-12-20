import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any
import yaml
from ultralytics import YOLO
from pathlib import Path
import requests
import sys
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]  # 获取项目根目录
sys.path.append(str(ROOT_DIR))

from src.devices.BerxelSdkDriver.BerxelHawkFrame import *
from src.devices.BerxelSdkDriver.BerxelHawkDevice import *
from src.devices.BerxelSdkDriver.BerxelHawkContext import *
from src.devices.BerxelSdkDriver.BerxelHawkDefines import *


class BerxelTracker:
    def __init__(self, 
                 model_path: str, 
                 config_path: Optional[str] = None,
                 test_mode: bool = False,
                 test_post: bool = False):
        """
        初始化BerxelTracker
        
        Args:
            model_path: YOLO模型路径
            config_path: 配置文件路径
        """
        # 基础设置
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
        # YOLO模型设置
        self.model = YOLO(model_path) if model_path else None
        self.previous_class_name = None
        self.stable_frame_count = 0
        
        # Berxel相机设置
        self.__context = None
        self.__device = None
        self.__deviceList = []
        
        # 通道控制
        self.rgb_enabled = True
        self.depth_enabled = True
        self.tracking_enabled = True
        
        # 最新帧缓存
        self.latest_rgb_frame = None
        self.latest_depth_frame = None
        self.latest_tracked_frame = None

        self.test_mode = test_mode
        self.test_post = test_post
        self.server_url = self.config.get("server_url", "http://localhost:5000")
        self.required_stable_frames = self.config.get("required_stable_frames", 3)

    def _setup_logging(self) -> logging.Logger:
        """配置日志系统"""
        logger = logging.getLogger('BerxelTracker')
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
            'mjpg_quality': 95
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
        """启动选定的数据流"""
        if self.__device is None:
            return False

        self.__device.setDenoiseStatus(False)
        
        # 准备stream标志
        stream_flags = 0
        if self.rgb_enabled:
            stream_flags |= BerxelHawkStreamType.forward_dict['BERXEL_HAWK_COLOR_STREAM']
        if self.depth_enabled:
            stream_flags |= BerxelHawkStreamType.forward_dict['BERXEL_HAWK_DEPTH_STREAM']
            
            # 设置深度流模式
            frameMode = self.__device.getCurrentFrameMode(
                BerxelHawkStreamType.forward_dict['BERXEL_HAWK_DEPTH_STREAM'])
            self.__device.setFrameMode(
                BerxelHawkStreamType.forward_dict['BERXEL_HAWK_DEPTH_STREAM'], 
                frameMode)

        return self.__device.startStreams(stream_flags) == 0
    
    
    
    
    
    
    def capture_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """捕获一帧数据"""
        rgb_frame = None
        depth_frame = None
        
        if self.rgb_enabled:
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
        
        if self.depth_enabled:
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
        """处理帧并进行目标检测"""
        if not self.tracking_enabled or self.model is None:
            return frame, None
            
        # 运行YOLO追踪
        results = self.model.track(frame, persist=True)
        annotated_frame = results[0].plot()
        detected_class_name = None

        # 提取检测到的类别
        if len(results[0].boxes) > 0:
            class_tensor = results[0].boxes.cls
            class_id = class_tensor[0].item()
            detected_class_name = self.model.names[class_id]

        # 稳定性过滤
        if detected_class_name == self.previous_class_name:
            self.stable_frame_count += 1
        else:
            self.stable_frame_count = 0

        filtered_class_name = None
        if self.stable_frame_count >= self.config['required_stable_frames']:
            filtered_class_name = detected_class_name
            
        self.previous_class_name = detected_class_name
        return annotated_frame, filtered_class_name
    
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

    def start_tracking(self):
        """开始追踪主循环"""
        if not self.open_device():
            self.logger.error("Failed to open device")
            return
            
        if not self.start_streams():
            self.logger.error("Failed to start streams")
            return
            
        self.logger.info("Starting tracking...")
        
        try:
            while True:
                # 捕获帧
                rgb_frame, depth_frame = self.capture_frame()
                # rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                
                if rgb_frame is not None:
                    # 处理RGB帧
                    if self.tracking_enabled:
                        tracked_frame, class_name = self.process_frame(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
                        self.latest_tracked_frame = tracked_frame
                        if class_name:
                            self.logger.info(f"Detected: {class_name}")
                            self.post_class_name(class_name)
                    else:
                        tracked_frame = rgb_frame
                        
                    # 显示RGB结果
                    if self.config['display_window']:
                        cv2.imshow("RGB View", tracked_frame)
                
                if depth_frame is not None:
                    # 显示深度图
                    if self.config['display_window']:
                        depth_display = ((depth_frame / 10000.) * 255).astype(np.uint8)
                        cv2.imshow("Depth View", depth_display)
                
                # 检查退出条件
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            self.cleanup()

    def get_latest_frame(self) -> Any:
        return self.latest_tracked_frame

    def cleanup(self):
        """清理资源"""
        if self.__device:
            stream_flags = 0
            if self.rgb_enabled:
                stream_flags |= BerxelHawkStreamType.forward_dict['BERXEL_HAWK_COLOR_STREAM']
            if self.depth_enabled:
                stream_flags |= BerxelHawkStreamType.forward_dict['BERXEL_HAWK_DEPTH_STREAM']
            self.__device.stopStream(stream_flags)

        if self.__context and self.__device:
            self.__context.clsoeDevice(self.__device)
            self.__context.destroyCamera()

        cv2.destroyAllWindows()

    def toggle_rgb(self, enabled: bool):
        """切换RGB通道状态"""
        self.rgb_enabled = enabled

    def toggle_depth(self, enabled: bool):
        """切换深度通道状态"""
        self.depth_enabled = enabled

    def toggle_tracking(self, enabled: bool):
        """切换追踪功能状态"""
        self.tracking_enabled = enabled




if __name__ == "__main__":
    # 使用示例
    tracker = BerxelTracker(
        model_path=ROOT_DIR / "runs/detect/train8/weights/best.pt",
        config_path= ROOT_DIR / "configs/tracker_config.yaml"
    )
    
    # 可以通过这些方法控制功能
    # tracker.toggle_depth(False)  # 关闭深度通道
    # tracker.toggle_tracking(False)  # 关闭追踪功能
    
    try:
        tracker.start_tracking()
    except KeyboardInterrupt:
        print("Stopping tracker...")
    finally:
        tracker.cleanup()
    
