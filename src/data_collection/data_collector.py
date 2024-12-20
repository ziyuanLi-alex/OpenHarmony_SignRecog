import struct
import time
import threading
import numpy as np
import cv2
from typing import Optional, Tuple
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]  # 获取项目根目录
sys.path.append(str(ROOT_DIR))

from src.devices.BerxelSdkDriver.BerxelHawkFrame import *
from src.devices.BerxelSdkDriver.BerxelHawkDevice import *
from src.devices.BerxelSdkDriver.BerxelHawkContext import *
from src.devices.BerxelSdkDriver.BerxelHawkDefines import *


class DataCollector:
    def __init__(self, save_dir: str = "dataset"):
        self.__context = None
        self.__device = None
        self.__deviceList = []
        self.is_collecting = False
        self.current_sign = None
        self.save_dir = ROOT_DIR / save_dir / "raw"
        self._setup_directories()
        self._setup_logging()

    def _setup_logging(self):
        """设置日志"""
        self.logger = logging.getLogger('DataCollector')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _setup_directories(self):
        """创建必要的目录结构"""
        # 创建训练集目录
        (self.save_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (self.save_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
        # 创建验证集目录
        (self.save_dir / "valid" / "images").mkdir(parents=True, exist_ok=True)
        (self.save_dir / "valid" / "labels").mkdir(parents=True, exist_ok=True)

    def openDevice(self) -> bool:
        """打开设备"""
        """打开设备"""
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
        if self.__device is None:
            return False

        return True

    def startStream(self) -> bool:
        """启动数据流"""
        if self.__device is None:
            return False

        self.__device.setDenoiseStatus(False)

        # 获取相机参数
        intrinsicParams = self.__device.getDeviceIntriscParams()
        self.logger.info(f"Color Camera Parameters: fx={intrinsicParams.colorIntrinsicParams.fx}, "
                         f"fy={intrinsicParams.colorIntrinsicParams.fy}")

        frameMode = self.__device.getCurrentFrameMode(
            BerxelHawkStreamType.forward_dict['BERXEL_HAWK_DEPTH_STREAM'])
        self.__device.setFrameMode(
            BerxelHawkStreamType.forward_dict['BERXEL_HAWK_DEPTH_STREAM'], frameMode)

        ret = self.__device.startStreams(
            BerxelHawkStreamType.forward_dict['BERXEL_HAWK_DEPTH_STREAM'] |
            BerxelHawkStreamType.forward_dict['BERXEL_HAWK_COLOR_STREAM'])

        return ret == 0

    def captureFrame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """捕获一帧数据，返回RGB图和深度图"""
        hawkDepthFrame = self.__device.readDepthFrame(30)
        if hawkDepthFrame is None:
            return None, None

        hawkColorFrame = self.__device.readColorFrame(30)
        if hawkColorFrame is None:
            self.__device.releaseFrame(hawkDepthFrame)
            return None, None

        try:
            # 获取深度图数据
            depthWidth = hawkDepthFrame.getWidth()
            depthHeight = hawkDepthFrame.getHeight()
            depthFrameBuffer = hawkDepthFrame.getDataAsUint16()
            depth_array = np.ndarray(shape=(depthHeight, depthWidth),
                                     dtype=np.uint16, buffer=depthFrameBuffer)

            # 获取彩色图数据
            colorWidth = hawkColorFrame.getWidth()
            colorHeight = hawkColorFrame.getHeight()
            colorFrameBuffer = hawkColorFrame.getDataAsUint8()
            color_array = np.ndarray(shape=(colorHeight, colorWidth, 3),
                                     dtype=np.uint8, buffer=colorFrameBuffer)

            return color_array, depth_array
        finally:
            self.__device.releaseFrame(hawkColorFrame)
            self.__device.releaseFrame(hawkDepthFrame)

    def save_data(self, rgb_img: np.ndarray, depth_img: np.ndarray,
                  sign: str, counter: int, split: str = "train"):
        """保存数据到指定目录"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"{sign}_{timestamp}_{counter}"

        # 保存RGB图像
        rgb_path = self.save_dir / split / "images" / f"{base_name}_rgb.jpg"
        cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

        # 保存深度图像
        depth_path = self.save_dir / split / "images" / f"{base_name}_depth.png"
        cv2.imwrite(str(depth_path), depth_img)

        self.logger.info(f"Saved {base_name} to {split} set")

    def collect_data(self):
        """主数据收集循环"""
        counter = 0
        while True:
            rgb_img, depth_img = self.captureFrame()
            if rgb_img is None or depth_img is None:
                continue

            # 显示实时画面
            cv2.imshow('RGB View', rgb_img)
            depth_display = ((depth_img / 10000.) * 255).astype(np.uint8)
            cv2.imshow('Depth View', depth_display)

            key = cv2.waitKey(1) & 0xFF

            # 空格键捕获当前手势
            if key == ord(' ') and self.current_sign:
                self.save_data(rgb_img, depth_img, self.current_sign, counter)
                counter += 1
                self.logger.info(f"Captured {self.current_sign} - {counter}")

            # ESC键退出
            elif key == 27:
                break

            # 其他键设置当前手势标签
            elif key in range(ord('a'), ord('z') + 1):
                self.current_sign = chr(key)
                self.logger.info(f"Current sign set to: {self.current_sign}")

    def start_collection(self):
        """开始数据收集"""
        if not self.openDevice():
            self.logger.error("Failed to open device")
            return

        if not self.startStream():
            self.logger.error("Failed to start stream")
            return

        self.logger.info("Starting data collection...")
        self.logger.info("Press space to capture current gesture")
        self.logger.info("Press a-z to set current gesture label")
        self.logger.info("Press ESC to exit")

        try:
            self.collect_data()
        finally:
            self.cleanup()

    def cleanup(self):
        """清理资源"""
        if self.__device:
            self.__device.stopStream(
                BerxelHawkStreamType.forward_dict['BERXEL_HAWK_DEPTH_STREAM'] |
                BerxelHawkStreamType.forward_dict['BERXEL_HAWK_COLOR_STREAM'])

        if self.__context and self.__device:
            self.__context.clsoeDevice(self.__device)
            self.__context.destroyCamera()

        cv2.destroyAllWindows()

if __name__ == '__main__':
    collector = DataCollector()
    collector.start_collection()

