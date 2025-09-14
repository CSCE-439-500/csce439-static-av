from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
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


def _union_feature_columns(
    shards: Iterable[Path], prefixes: Sequence[str], scan_limit: int = 3
) -> list[str]:
    """
    Scan up to scan_limit shards to build a stable union of feature columns
    matching the provided prefixes. Keeps deterministic order (sorted).
    """
    cols: set[str] = set()
    for i, p in enumerate(shards):
        df = pd.read_parquet(p, engine="pyarrow", columns=None)  # read header to get columns
        cols.update(_select_feature_columns(df.columns, prefixes))
        if i + 1 >= scan_limit:
            break
    return sorted(cols)


def _load_train_matrix(
    shards: Iterable[Path],
    feat_cols: Sequence[str],
    limit_rows: int | None = None,
    rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load up to limit_rows labeled rows from shards into (X, y).
    Missing columns are filled with 0. Casts features to float32, labels to int8.
    If limit_rows is set and total > limit_rows, down-samples uniformly.
    """
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    total = 0

    for p in shards:
        df = pd.read_parquet(p, engine="pyarrow")
        # Keep labeled rows only (already true in cache, but safe)
        df = df[df["label"] != -1]
        # Ensure all requested columns exist; fill missing with 0
        missing = [c for c in feat_cols if c not in df.columns]
        for m in missing:
            df[m] = 0
        X = df[feat_cols].fillna(0).astype("float32").values
        y = df["label"].astype("int8").values
        X_parts.append(X)
        y_parts.append(y)
        total += len(y)

    if not X_parts:
        return np.empty((0, len(feat_cols)), dtype="float32"), np.empty((0,), dtype="int8")

    X_all = np.vstack(X_parts)
    y_all = np.concatenate(y_parts)

    if limit_rows is not None and len(y_all) > limit_rows:
        rng = rng or check_random_state(1337)
        idx = rng.choice(len(y_all), size=limit_rows, replace=False)
        X_all = X_all[idx]
        y_all = y_all[idx]

    return X_all, y_all


def _class_balance_weights(y: np.ndarray) -> np.ndarray:
    """
    Create sample weights to balance classes roughly equally.
    w_c = N / (2 * count_c)
    """
    N = len(y)
    w = np.empty(N, dtype="float32")
    counts = np.bincount(y, minlength=2).astype(np.float64)
    for c in (0, 1):
        mask = y == c
        denom = max(counts[c], 1.0)
        w[mask] = N / (2.0 * denom)
    return w


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train a baseline EMBER model (HGBT) on cached Parquet shards."
    )
    ap.add_argument(
        "--cache", type=Path, default=os.getenv("EMBER_CACHE"), help="Parquet cache directory."
    )
    ap.add_argument(
        "--out-model",
        type=Path,
        default=Path("artifacts/ember_model.joblib"),
        help="Where to save the model.",
    )
    ap.add_argument(
        "--out-spec",
        type=Path,
        default=Path("artifacts/ember_feature_spec.json"),
        help="Where to save feature spec.",
    )
    ap.add_argument(
        "--feature-prefixes",
        type=str,
        default="histogram.* , general.* , header.* , strings.*",
        help="Comma-separated feature prefixes to use.",
    )
    ap.add_argument(
        "--scan-shards",
        type=int,
        default=3,
        help="How many shards to scan to build the feature union.",
    )
    ap.add_argument(
        "--limit-rows",
        type=int,
        default=600_000,
        help="Max rows to load for training (RAM safety).",
    )
    ap.add_argument("--random-state", type=int, default=1337, help="Random seed.")
    ap.add_argument(
        "--max-leaf-nodes", type=int, default=31, help="HGBT max leaf nodes (complexity control)."
    )
    ap.add_argument("--learning-rate", type=float, default=0.1, help="HGBT learning rate.")
    ap.add_argument("--max-iter", type=int, default=500, help="HGBT max iterations.")
    ap.add_argument(
        "--early-stopping",
        action="store_true",
        default=True,
        help="Use early stopping on 10%% validation.",
    )
    ap.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Validation fraction for early stopping.",
    )
    args = ap.parse_args()

    if not args.cache or not args.cache.exists():
        raise SystemExit("Cache directory not found. Set --cache or EMBER_CACHE.")

    feature_prefixes = [p.strip() for p in args.feature_prefixes.split(",") if p.strip()]

    print(f"[train-ember] cache={args.cache}")
    print(f"[train-ember] prefixes={feature_prefixes}")

    # Build feature union from first N shards
    feat_cols = _union_feature_columns(
        _iter_shards(args.cache, "train_*.parquet"), feature_prefixes, scan_limit=args.scan_shards
    )
    if not feat_cols:
        raise SystemExit("No feature columns matched the given prefixes.")
    print(f"[train-ember] feature columns: {len(feat_cols)}")

    # 2 Load training matrix
    rng = check_random_state(args.random_state)
    X, y = _load_train_matrix(
        _iter_shards(args.cache, "train_*.parquet"), feat_cols, limit_rows=args.limit_rows, rng=rng
    )
    if y.size == 0:
        raise SystemExit("No training data found.")
    print(f"[train-ember] loaded rows: {len(y)}  (features: {X.shape[1]})")

    # Sample weights for class balance
    sw = _class_balance_weights(y)

    # Train HGBT with early stopping
    clf = HistGradientBoostingClassifier(
        max_leaf_nodes=args.max_leaf_nodes,
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        early_stopping=args.early_stopping,
        validation_fraction=args.validation_fraction,
        random_state=args.random_state,
    )
    clf.fit(X, y, sample_weight=sw)

    # Save artifacts
    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, args.out_model)
    print(f"[train-ember] saved model -> {args.out_model}")

    spec = {
        "feature_prefixes": feature_prefixes,
        "feature_columns": feat_cols,  # ordered columns used at train time
        "rows": int(len(y)),
        "sklearn": getattr(clf, "__module__", "sklearn"),
        "model_cls": clf.__class__.__name__,
        "params": {
            "max_leaf_nodes": args.max_leaf_nodes,
            "learning_rate": args.learning_rate,
            "max_iter": args.max_iter,
            "early_stopping": args.early_stopping,
            "validation_fraction": args.validation_fraction,
            "random_state": args.random_state,
        },
    }
    args.out_spec.parent.mkdir(parents=True, exist_ok=True)
    args.out_spec.write_text(json.dumps(spec, indent=2))
    print(f"[train-ember] saved feature spec -> {args.out_spec}")


if __name__ == "__main__":
    main()
