import cv2
from ultralytics import YOLO
import requests
import logging
from typing import Optional, Any
import yaml
import os

logger = logging.getLogger(__name__)

class YOLOTracker:
    def __init__(self, model_path:str,
                 video_source=0,
                 test_mode=False,
                 test_post=False,
                 config_path: Optional[str] = None):

        self.config = self._load_config(config_path)
        # self.config = load_config(config_path)

        # Load the YOLO model
        self.model = YOLO(model_path)
        self.test_mode = test_mode
        self.test_post = test_post
        self.video_source = video_source
        self.latest_frame = None

        self.server_url = self.config.get("server_url", "http://localhost:5000")
        self.required_stable_frames = self.config.get("required_stable_frames", 3)

        if not self.test_mode:
            self._initialize_capture()

        # Parameters for filtering
        self.previous_class_name = None
        self.stable_frame_count = 0


    def _load_config(self, config_path: Optional[str] ) -> dict:
        """加载配置文件"""

        default_config = {
            'server_url': 'http://localhost:5000',
            'required_stable_frames': 3,
            'mjpg_quality': 95,
            'display_window': True
        }

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.error(f"Error loading config: {e}")

        return default_config

    def _initialize_capture(self):
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            raise ValueError(f"Error: Unable to open video source {self.video_source}")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))


    def process_frame(self, frame):
        # Run YOLO tracking on the frame
        results = self.model.track(frame, persist=True, verbose=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        detected_class_name = None

        # If any object is detected, extract the class name
        if len(results[0].boxes) > 0:
            class_tensor = results[0].boxes.cls
            class_id = class_tensor[0].item()
            detected_class_name = self.model.names[class_id]

        # Filtering logic to prevent rapid jumps
        if detected_class_name == self.previous_class_name:
            self.stable_frame_count += 1
        else:
            self.stable_frame_count = 0

        if self.stable_frame_count >= self.required_stable_frames:
            filtered_class_name = detected_class_name
        else:
            filtered_class_name = self.previous_class_name

        # Update previous class name
        self.previous_class_name = detected_class_name

        if filtered_class_name is not None:
            logger.info(f"Detected class: {filtered_class_name}")

        self.latest_frame = annotated_frame
        return annotated_frame, filtered_class_name


    def start_tracking(self):
        """开始追踪"""
        try:
            if self.test_mode:
                self._run_test_mode()
            else:
                self._run_normal_mode()
        finally:
            self.cleanup()

    def _run_test_mode(self) -> None:
        """运行测试模式"""
        frame = cv2.imread("test2.jpg")
        if frame is None:
            logger.error("Error: test.jpg not found.")
            return

        annotated_frame, class_name = self.process_frame(frame)
        if class_name:
            self.post_class_name(class_name)

        if self.config['display_window']:
            cv2.imshow("YOLO Tracking", annotated_frame)
            cv2.waitKey(0)

    def _run_normal_mode(self) -> None:
        """运行正常模式"""
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break

            annotated_frame, class_name = self.process_frame(frame)
            if class_name:
                self.post_class_name(class_name)

            if self.config['display_window']:
                cv2.imshow("YOLO Tracking", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    def get_latest_frame(self) -> Any:
        return self.latest_frame

    def post_class_name(self, class_name) -> None:
        url = f"{self.server_url}/recognize"
        data = {"class_name": class_name}

        if self.test_post:
            logger.info(f"Pseudo-posting data: {data}")
            return

        try:
            response = requests.post(url, json=data, timeout=1.0)
            if response.status_code != 200:
                logger.error(f"Failed to post: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error posting data: {e}")


    def cleanup(self) -> None:
        if not self.test_mode and hasattr(self, "cap"):
            self.cap.release()
        cv2.destroyAllWindows()


example_config = """
server_url: 'http://localhost:5000'
required_stable_frames: 3
mjpg_quality: 95
display_window: true
camera_settings:
  CAP_PROP_FRAME_WIDTH: 640
  CAP_PROP_FRAME_HEIGHT: 480
  CAP_PROP_FPS: 30
"""


if __name__ == "__main__":
    # [MODIFIED] 添加配置文件支持
    tracker = YOLOTracker(
        model_path="runs/detect/train1/weights/best.pt",
        config_path="configs/tracker_config.yaml",
        test_mode=False
    )
    try:
        tracker.start_tracking()
    finally:
        tracker.cleanup()