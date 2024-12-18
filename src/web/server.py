from flask import Flask
from flask_socketio import SocketIO
import logging


def create_app(config_path=None):
    """初始化 Flask 应用"""
    app = Flask(__name__)

    # 从配置文件加载配置
    if config_path:
        app.config.from_pyfile(config_path)

    # 初始化 SocketIO
    socketio = SocketIO(app, cors_allowed_origins="*")

    # 配置日志
    if not app.debug:
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

    return app, socketio