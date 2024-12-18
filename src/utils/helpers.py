import cv2
import time
from typing import Optional, Dict, Any
import yaml

def generate_html_response(class_name: Optional[str]) -> tuple:
    """生成 HTML 响应"""
    html = f"""
    <html>
        <head>
            <style>
                pre {{
                    font-size: 100px;
                    background-color: #f5f5f5;
                    padding: 20px;
                    border-radius: 5px;
                }}
            </style>
        </head>
        <body>
            <pre>{class_name if class_name else 'No class recognized'}</pre>
        </body>
    </html>
    """
    return html, 200

def generate_mjpg_stream(tracker):
    """生成 MJPEG 视频流"""
    while True:
        frame = tracker.get_latest_frame()
        if frame is not None:
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' +
                      jpeg.tobytes() + b'\r\n')
            time.sleep(1 / 30)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        return {}
