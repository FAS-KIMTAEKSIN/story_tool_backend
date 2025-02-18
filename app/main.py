from flask import Flask, send_from_directory, render_template
from flask_cors import CORS
from app.api.routes import api_bp
import os

def create_app():
    # 빌드 파일 경로를 현재 디렉토리('.')로 설정
    build_path = os.getenv('REACT_BUILD_PATH', '.')
    
    app = Flask(__name__,
                static_folder="static",  # 상대 경로로 변경
                template_folder=build_path)
    CORS(app)
    
    # React의 index.html 반환을 위한 라우트
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve(path):
        if path != "" and path.startswith("static"):
            return send_from_directory("static", path.replace("static/", ""))
        elif path == "asset-manifest.json":
            return send_from_directory(build_path, path)
        return render_template("index.html")
    
    # API 블루프린트 등록
    app.register_blueprint(api_bp, url_prefix='/api')
    
    return app