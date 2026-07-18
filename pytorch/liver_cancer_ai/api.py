import os
import tempfile

from flask import Flask, jsonify, request

from .inference import predict_ct


def create_app():
    app = Flask(__name__, static_folder="../../web", static_url_path="")
    weights_path = os.environ.get("LIVER_MODEL_WEIGHTS")
    device = os.environ.get("LIVER_MODEL_DEVICE")

    @app.get("/")
    def index():
        return app.send_static_file("index.html")

    @app.post("/api/predict")
    def predict():
        uploaded = request.files.get("file")
        if uploaded is None or uploaded.filename == "":
            return jsonify({"error": "请上传 CT 影像文件"}), 400

        suffix = os.path.splitext(uploaded.filename)[1] or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            uploaded.save(temp_file.name)
            temp_path = temp_file.name

        try:
            result = predict_ct(temp_path, weights_path=weights_path, device=device)
            result["filename"] = uploaded.filename
            return jsonify(result)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
