from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from defender.app.ember_adapter import bytes_to_numeric_features


@dataclass
class PredictorEmber:
    model_path: Path
    feature_spec_path: Path
    threshold_json_path: Path

    def __post_init__(self) -> None:
        self.model = joblib.load(self.model_path)
        spec = json.loads(Path(self.feature_spec_path).read_text())
        self.feature_columns: list[str] = list(spec.get("feature_columns", []))
        if not self.feature_columns:
            raise RuntimeError("feature_spec missing 'feature_columns'")

        t = json.loads(Path(self.threshold_json_path).read_text()).get("tau", None)
        if t is None:
            raise RuntimeError("threshold.json missing 'tau'")
        self.tau: float = float(t)

    def _vectorize(self, feats: dict[str, Any]) -> np.ndarray:
        """Order features exactly as in training; fill missing with 0."""
        row = [feats.get(col, 0.0) for col in self.feature_columns]
        X = np.asarray([row], dtype=np.float32)
        return X

    def score(self, bytez: bytes) -> float:
        feats = bytes_to_numeric_features(bytez)
        X = self._vectorize(feats)
        if hasattr(self.model, "decision_function"):
            s = float(self.model.decision_function(X)[0])
        else:
            # calibrated models expose predict_proba
            s = float(self.model.predict_proba(X)[0, 1])
        return s

    def predict(self, bytez: bytes) -> int:
        s = self.score(bytez)
        return int(s >= self.tau)

    def predict_debug(self, bytez: bytes) -> tuple[int, dict[str, Any]]:
        feats = bytes_to_numeric_features(bytez)
        X = self._vectorize(feats)
        if hasattr(self.model, "decision_function"):
            s = float(self.model.decision_function(X)[0])
            conf = None
            rule = "decision_function"
        else:
            proba = self.model.predict_proba(X)[0]
            s = float(proba[1])
            conf = [float(proba[0]), float(proba[1])]
            rule = "predict_proba"

        yhat = int(s >= self.tau)
        dbg: dict[str, Any] = {
            "score": s,
            "tau": self.tau,
            "decision_rule": rule,
            "computed_keys": len(feats),
            "missing_keys": int(sum(1 for c in self.feature_columns if c not in feats)),
        }
        if conf is not None:
            dbg["proba"] = conf
        # (Optional: include a tiny subset of feature signals)
        keep = [k for k in self.feature_columns if k.startswith("histogram.")][:8]
        dbg["hist_head"] = {k: feats.get(k, 0.0) for k in keep}
        return yhat, dbg

    def model_info(self) -> dict:
        base = {
            "runtime": "ember-blackbox",
            "features": {
                "families": ["histogram", "byteentropy", "general", "header", "strings"],
                "using_histogram": True,
                "using_byteentropy": False,
                "using_general_header": False,
                "using_strings": False,
            },
            "threshold": self.tau,
            "model_path": str(self.model_path),
            "feature_spec": str(self.feature_spec_path),
        }
        # Allow model to contribute metadata if present
        info = getattr(self.model, "model_info", None)
        if callable(info):
            base.update(info())
        return base
