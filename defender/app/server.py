from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request

from defender.app.predictor import Predictor

load_dotenv()


def _expand(p: str) -> Path:
    return Path(os.path.expandvars(p)).expanduser().resolve()


# Model under ARTIFACTS_DIR unless MODEL_PATH is set
ARTIFACTS_DIR = _expand(os.getenv("ARTIFACTS_DIR", "./artifacts"))
MODEL_PATH = _expand(os.getenv("MODEL_PATH", str(ARTIFACTS_DIR / "baseline.joblib")))
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["predictor"] = Predictor(MODEL_PATH, threshold=THRESHOLD)

    @app.post("/")
    def classify():
        if request.headers.get("Content-Type", "") != "application/octet-stream":
            return jsonify({"error": "expecting application/octet-stream"}), 400
        bytez = request.data or b""
        try:
            pred = app.config["predictor"].predict(bytez)
            if pred not in (0, 1):
                return jsonify({"error": "unexpected model result (not in [0,1])"}), 500
            return jsonify({"result": int(pred)}), 200
        except Exception as e:
            app.logger.exception("prediction failed: %s", e)
            # Count parse/other errors as malicious
            return jsonify({"result": 1}), 200

    @app.get("/model")
    def model():
        return jsonify(app.config["predictor"].model_info()), 200

    return app


app = create_app()

if __name__ == "__main__":
    # Dev server
    app.run(host="0.0.0.0", port=8080, debug=True)
