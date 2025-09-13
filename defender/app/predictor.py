from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

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
        # If change threshold, put predict_prba here
        return yhat

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
