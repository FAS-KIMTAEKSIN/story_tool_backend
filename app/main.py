from flask import Flask, send_from_directory, render_template
from flask_cors import CORS
from app.api.routes import api_bp

def create_app():
    app = Flask(__name__,
                static_folder="static",
                template_folder="templates")
    CORS(app)
    
    # React의 index.html 반환을 위한 라우트
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve(path):
        if path != "" and path.startswith("static"):
            return send_from_directory(app.static_folder, path)
        return render_template("index.html")
    
    # API 블루프린트 등록
    app.register_blueprint(api_bp, url_prefix='/api')
    
    return app