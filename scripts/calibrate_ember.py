from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import check_random_state


def _iter_shards(cache_dir: Path, pattern: str) -> Iterable[Path]:
    yield from sorted(cache_dir.glob(pattern))


def _select_feature_columns(all_cols: Sequence[str], prefixes: Sequence[str]) -> list[str]:
    feat_cols: list[str] = []
    for c in all_cols:
        if c in ("label", "subset"):
            continue
        for p in prefixes:
            if p.endswith(".*"):
                pref = p[:-2]
                if c.startswith(pref):
                    feat_cols.append(c)
                    break
            else:
                if c == p:
                    feat_cols.append(c)
                    break
    return feat_cols


def _load_matrix(
    shards: Iterable[Path],
    *,
    spec_cols: Sequence[str] | None,
    prefixes: Sequence[str],
    limit_rows: int,
    seed: int = 1337,
) -> tuple[np.ndarray, np.ndarray]:
    rng = check_random_state(seed)
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    total = 0
    for p in shards:
        df = pd.read_parquet(p)
        if "label" not in df.columns:
            continue
        y = df["label"].astype("int8").values

        feat_cols = list(spec_cols) if spec_cols else _select_feature_columns(df.columns, prefixes)
        if not feat_cols:
            continue
        missing = [c for c in feat_cols if c not in df.columns]
        for m in missing:
            df[m] = 0
        X = df[feat_cols].fillna(0).astype("float32").values

        X_parts.append(X)
        y_parts.append(y)
        total += len(y)
        if total >= limit_rows:
            break

    if not X_parts:
        return np.empty((0, 0), dtype="float32"), np.empty((0,), dtype="int8")

    X_all = np.vstack(X_parts)
    y_all = np.concatenate(y_parts)

    # Downsample exactly to limit_rows if we overshot
    if len(y_all) > limit_rows:
        idx = rng.choice(len(y_all), size=limit_rows, replace=False)
        X_all, y_all = X_all[idx], y_all[idx]
    return X_all, y_all


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Calibrate a prefit classifier (isotonic) on EMBER shards."
    )
    ap.add_argument(
        "--cache", type=Path, default=os.getenv("EMBER_CACHE"), help="Parquet cache directory."
    )
    ap.add_argument("--model-in", type=Path, required=True, help="Prefit model (joblib).")
    ap.add_argument(
        "--spec", type=Path, required=True, help="Feature spec JSON with 'feature_columns'."
    )
    ap.add_argument(
        "--model-out", type=Path, default=Path("artifacts/ember_model_calibrated.joblib")
    )
    ap.add_argument(
        "--feature-prefixes",
        type=str,
        default="histogram.* , byteentropy.* , general.* , header.* , strings.*",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="train_features_*.parquet",
        help="Shards to use for calibration.",
    )
    ap.add_argument("--cal-rows", type=int, default=200_000, help="Rows to use for calibration.")
    args = ap.parse_args()

    if not args.cache or not args.cache.exists():
        raise SystemExit("Cache directory not found. Set --cache or EMBER_CACHE.")

    spec = json.loads(args.spec.read_text())
    spec_cols = spec.get("feature_columns", None)
    if not spec_cols:
        raise SystemExit("Spec missing 'feature_columns'.")

    prefixes = [p.strip() for p in args.feature_prefixes.split(",") if p.strip()]

    model = joblib.load(args.model_in)

    X_cal, y_cal = _load_matrix(
        _iter_shards(args.cache, args.pattern),
        spec_cols=spec_cols,
        prefixes=prefixes,
        limit_rows=args.cal_rows,
    )
    if y_cal.size == 0:
        raise SystemExit("No calibration data found.")

    # Fit isotonic calibration on the prefit model
    calib = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calib.fit(X_cal, y_cal)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calib, args.model_out)
    print(f"[calibrate] wrote calibrated model -> {args.model_out}")


if __name__ == "__main__":
    main()
