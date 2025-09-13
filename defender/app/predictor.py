from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import joblib
import pandas as pd

from avlab.features.pe_static import extract_pe_features
from avlab.serve.pe_bytes_adapter import bytes_to_features


@dataclass
class Predictor:
    model_path: Path
    threshold: float | None = None  # use later for calibrated thresholds

    def __post_init__(self) -> None:
        self.model = joblib.load(self.model_path)

    def predict(self, bytez: bytes) -> int:
        df: pd.DataFrame = bytes_to_features(bytez)
        yhat = int(self.model.predict(df[["n_imports"]].values)[0])
        # If change threshold, put predict_proba here
        return yhat

    def predict_debug(self, bytez: bytes) -> tuple[int, dict[str, Any]]:
        """
        Returns (label, debug_dict) with minimal signals to understand the decision.
        debug_dict keys:
          - parsed: bool  (did PE parse succeed)
          - n_imports: int
          - proba: [p0, p1]  (if available)
          - decision_rule: str  ('svc.predict' or 'svc.predict_proba' or 'fallback_exception')
        """
        try:
            parsed, n_imports = self._extract_feature_with_status(bytez)
            X = pd.DataFrame([{"n_imports": n_imports}])[["n_imports"]].values

            proba: list[float] | None = None
            decision_rule = "svc.predict"

            # If the underlying model exposes probabilities, show them
            if hasattr(self.model, "predict_proba"):
                try:
                    pp = self.model.predict_proba(X)[0]
                    proba = [float(pp[0]), float(pp[1])]
                    decision_rule = "svc.predict_proba"
                except Exception:
                    # Fall back silently if proba fails for any reason
                    proba = None
                    decision_rule = "svc.predict"

            yhat = int(self.model.predict(X)[0])

            dbg: dict[str, Any] = {
                "parsed": bool(parsed),
                "n_imports": int(n_imports),
                "decision_rule": decision_rule,
            }
            if proba is not None:
                dbg["proba"] = proba

            return yhat, dbg

        except Exception as e:
            # In case anything blows up before the model call, mirror server fail-safe
            return 1, {
                "parsed": False,
                "n_imports": 0,
                "decision_rule": "fallback_exception",
                "error": str(e),
            }

    # Extract n_imports and whether parsing succeeded, without altering training code paths.
    def _extract_feature_with_status(self, bytez: bytes) -> tuple[bool, int]:
        """
        Writes bytes to a temp file and calls the same extractor used elsewhere,
        but captures a 'parsed' flag separately from the 'n_imports=0' fallback.
        """
        with NamedTemporaryFile(suffix=".bin", delete=True) as tf:
            tf.write(bytez)
            tf.flush()
            try:
                feats = extract_pe_features(
                    Path(tf.name)
                )  # includes {"parsed": bool, "n_imports": int}
                parsed = bool(feats.get("parsed", True))
                n_imports = int(feats.get("n_imports", 0))
            except Exception:
                parsed = False
                n_imports = 0
        return parsed, n_imports

    def model_info(self) -> dict:
        info = getattr(self.model, "model_info", None)
        base = info() if callable(info) else {}
        base.update(
            {
                "runtime": "defender-blackbox",
                "features": base.get("features", ["n_imports"]),
                "threshold": self.threshold,
            }
        )
        return base
