from flask import request, Response
from ..utils.helpers import generate_html_response, generate_mjpg_stream


class WebState:
    """管理 Web 应用状态"""
    latest_class_name = None


def register_routes(app, tracker):
    """注册所有 Web 路由"""

    @app.route('/recognize', methods=['POST'])
    def recognize():
        data = request.get_json()
        if 'class_name' not in data:
            return {"error": "Missing class_name in request"}, 400

        WebState.latest_class_name = data['class_name']
        return {
            "message": f"Class '{data['class_name']}' recognized successfully"
        }, 200

    @app.route('/recognized_class', methods=['GET'])
    def get_recognized_class():
        return generate_html_response(WebState.latest_class_name)

    @app.route('/mjpg_stream', methods=['GET'])
    def mjpg_stream():
        return Response(
            generate_mjpg_stream(tracker),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    return app