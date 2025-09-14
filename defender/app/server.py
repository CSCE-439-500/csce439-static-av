from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request

load_dotenv()


def _expand(p: str) -> Path:
    return Path(os.path.expandvars(p)).expanduser().resolve()


# Directories / paths
ARTIFACTS_DIR = _expand(os.getenv("ARTIFACTS_DIR", "./artifacts"))

# Which runtime to use: "demo" (old) or "ember" (new)
MODEL_KIND = os.getenv("MODEL_KIND", "ember").strip().lower()

# Demo model defaults (kept for backwards-compat)
MODEL_PATH_DEMO = _expand(os.getenv("MODEL_PATH", str(ARTIFACTS_DIR / "baseline.joblib")))
THRESHOLD_DEMO = float(os.getenv("THRESHOLD", "0.5"))

# Ember model artifacts
MODEL_PATH_EMBER = _expand(
    os.getenv("MODEL_PATH", str(ARTIFACTS_DIR / "ember_model_calibrated.joblib"))
)
FEATURE_SPEC_PATH = _expand(
    os.getenv("FEATURE_SPEC", str(ARTIFACTS_DIR / "ember_feature_spec.json"))
)
THRESHOLD_JSON_PATH = _expand(os.getenv("THRESHOLD_JSON", str(ARTIFACTS_DIR / "threshold.json")))


def create_app() -> Flask:
    app = Flask(__name__)

    # Select predictor lazily to avoid importing unused dependencies
    if MODEL_KIND == "ember":
        from defender.app.predictor_ember import PredictorEmber  # local import

        app.config["predictor"] = PredictorEmber(
            model_path=MODEL_PATH_EMBER,
            feature_spec_path=FEATURE_SPEC_PATH,
            threshold_json_path=THRESHOLD_JSON_PATH,
        )
    else:
        from defender.app.predictor import Predictor as DemoPredictor  # local import

        app.config["predictor"] = DemoPredictor(MODEL_PATH_DEMO, threshold=THRESHOLD_DEMO)

    @app.post("/")
    def classify():
        # Expect raw bytes
        if request.headers.get("Content-Type", "") != "application/octet-stream":
            return jsonify({"error": "expecting application/octet-stream"}), 400

        # Optional lightweight debug
        debug_enabled = (
            request.headers.get("X-Debug", "").strip() == "1"
            or request.args.get("debug", "").strip() == "1"
        )

        bytez = request.data or b""
        try:
            if debug_enabled and hasattr(app.config["predictor"], "predict_debug"):
                pred, dbg = app.config["predictor"].predict_debug(bytez)
                if pred not in (0, 1):
                    return jsonify({"error": "unexpected model result (not in [0,1])"}), 500
                return jsonify({"result": int(pred), "debug": dbg}), 200
            else:
                pred = app.config["predictor"].predict(bytez)
                if pred not in (0, 1):
                    return jsonify({"error": "unexpected model result (not in [0,1])"}), 500
                return jsonify({"result": int(pred)}), 200

        except Exception as e:
            app.logger.exception("prediction failed: %s", e)
            resp = {"result": 1}
            if debug_enabled:
                resp["debug"] = {"error": "exception", "reason": str(e)}
            return jsonify(resp), 200

    @app.get("/model")
    def model():
        return jsonify(app.config["predictor"].model_info()), 200

    return app


app = create_app()

if __name__ == "__main__":
    # Dev server
    app.run(host="0.0.0.0", port=8080, debug=True)
