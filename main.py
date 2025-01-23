import threading
from src.web.server import create_app
from src.web.routes import register_routes
from src.core.yolo_tracker import YOLOTracker
from src.utils.helpers import load_config
from src.core.BerxelTracker import BerxelTracker
from src.core.DualModelTracker import DualModelTracker


def main():
    # 加载配置
    config = load_config('configs/model_config.yaml')

    # 创建应用实例
    app, socketio = create_app()

    # 初始化追踪器
    tracker_type = config.get('tracker_type', 'yolo')  # 默认使用YOLO追踪器
    
    if tracker_type == 'yolo':
        tracker = YOLOTracker(
            model_path="runs/detect/train8/weights/best.pt",
            config_path='configs/tracker_config.yaml'
        )
    elif tracker_type == 'berxel':
        tracker = BerxelTracker(
            model_path="runs/detect/train8/weights/best.pt",
            config_path="configs/tracker_config.yaml"
        )
    elif tracker_type == 'dual':
        # 使用双模型追踪器
        tracker = DualModelTracker(
            model_path="runs/detect/train8/weights/best.pt",  # RGB模型路径
            config_path="configs/dual_tracker_config.yaml"  # 双模型配置文件
        )
    else:
        raise ValueError(f"Unsupported tracker type: {tracker_type}")

    # 注册路由
    register_routes(app, tracker)

    # 启动服务器线程
    def run_server():
        socketio.run(app, host="0.0.0.0", port=5000)

    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()

    # 启动追踪
    tracker.start_tracking()


if __name__ == "__main__":
    main()