#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from avlab.metrics.thresholding import report_at_threshold, roc, threshold_at_fpr


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


def _iter_shards(cache_dir: Path, pattern: str) -> Iterable[Path]:
    yield from sorted(cache_dir.glob(pattern))


def _scores_for_shards(
    model,
    shards: Iterable[Path],
    feature_prefixes: Sequence[str],
    *,
    spec_cols: Sequence[str] | None = None,
    limit_rows: int | None = None,
    batch_rows: int = 200_000,  # reserved for future chunked scoring
) -> tuple[np.ndarray, np.ndarray]:
    ys: list[np.ndarray] = []
    ss: list[np.ndarray] = []
    seen = 0

    for shard in shards:
        df = pd.read_parquet(shard)

        # Grab labels FIRST (before we slice away columns)
        if "label" not in df.columns:
            # shard without labels? skip defensively
            continue
        y = df["label"].astype("int8").values

        # Decide feature columns: prefer spec (exact order) else by prefixes
        feat_cols = (
            list(spec_cols) if spec_cols else _select_feature_columns(df.columns, feature_prefixes)
        )
        if not feat_cols:
            continue

        # Ensure missing feature columns exist, then enforce exact ordering
        missing = [c for c in feat_cols if c not in df.columns]
        for m in missing:
            df[m] = 0
        X_df = df[feat_cols].fillna(0).astype("float32")
        X = X_df.values

        # Score
        if hasattr(model, "decision_function"):
            s = model.decision_function(X)
        else:
            s = model.predict_proba(X)[:, 1]

        ys.append(y)
        ss.append(s.astype("float32"))

        seen += len(y)
        if limit_rows is not None and seen >= limit_rows:
            break

    y_all = np.concatenate(ys) if ys else np.empty((0,), dtype="int8")
    s_all = np.concatenate(ss) if ss else np.empty((0,), dtype="float32")
    return y_all, s_all


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate EMBER model with ROC and FPR@1%% / TPR@tau.")
    ap.add_argument(
        "--cache", type=Path, default=os.getenv("EMBER_CACHE"), help="Parquet cache directory."
    )
    ap.add_argument("--model", type=Path, required=True, help="Path to trained model (joblib).")
    ap.add_argument(
        "--feature-prefixes",
        type=str,
        default="histogram.* , general.* , header.* , strings.*",
        help="Comma-separated feature prefixes (e.g., 'histogram.* , general.* , header.* , strings.*').",
    )
    ap.add_argument(
        "--val-source",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Which subset to use for threshold selection.",
    )
    ap.add_argument(
        "--val-max-rows", type=int, default=500_000, help="Max rows from validation to compute tau."
    )
    ap.add_argument(
        "--target-fpr",
        type=float,
        default=0.01,
        help="Target FPR for thresholding (e.g., 0.01 = 1%%).",
    )
    ap.add_argument(
        "--threshold-in", type=Path, default=None, help="Optional JSON file with {'tau': <float>}."
    )
    ap.add_argument(
        "--threshold-out", type=Path, default=None, help="Where to save picked threshold JSON."
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/ember_eval.csv"),
        help="CSV to write summary metrics.",
    )
    ap.add_argument(
        "--spec",
        type=Path,
        default=None,
        help="Feature spec JSON with 'feature_columns' from training.",
    )
    args = ap.parse_args()

    # Resolve cache dir carefully; fail fast if it's the CWD because env var was empty
    if not args.cache:
        raise SystemExit("Cache directory not provided. Set --cache or EMBER_CACHE.")
    cache_dir = args.cache.resolve()
    if not cache_dir.exists():
        raise SystemExit(f"Cache directory not found: {cache_dir}")

    feature_prefixes = [p.strip() for p in args.feature_prefixes.split(",") if p.strip()]
    model = joblib.load(args.model)

    # Load spec columns if provided
    spec_cols: list[str] | None = None
    if args.spec is not None and args.spec.exists():
        spec = json.loads(args.spec.read_text())
        sc = spec.get("feature_columns", None)
        if sc:
            spec_cols = list(sc)

    # -------- 1) Pick tau on validation --------
    if args.threshold_in and args.threshold_in.exists():
        tau = float(json.loads(args.threshold_in.read_text())["tau"])
        source_for_tau = "file"
    else:
        # shard patterns based on your cache naming:
        #   train: train_features_0.000.parquet, train_features_1.000.parquet, ...
        #   test:  test_features.000.parquet
        pat = "train_features_*.parquet" if args.val_source == "train" else "test_features*.parquet"
        # the simpler patterns above already match your files because '*' spans the '.000';
        # if you want to be stricter, you could use 'train_features_*.???.parquet' / 'test_features.???.parquet'.
        shards = list(_iter_shards(cache_dir, pat))
        print(f"[eval-ember] cache={cache_dir}")
        print(f"[eval-ember] using pattern: {pat}")
        print(f"[eval-ember] found shards: {len(shards)}")
        if not shards:
            raise SystemExit("No validation shards matched. Check --cache and filename pattern.")

        y_val, s_val = _scores_for_shards(
            model,
            shards,
            feature_prefixes,
            spec_cols=spec_cols,
            limit_rows=args.val_max_rows,
        )
        if y_val.size == 0:
            raise SystemExit("No validation data found to compute threshold.")
        tau = threshold_at_fpr(y_val, s_val, target_fpr=args.target_fpr)
        source_for_tau = args.val_source
        if args.threshold_out:
            args.threshold_out.parent.mkdir(parents=True, exist_ok=True)
            args.threshold_out.write_text(json.dumps({"tau": float(tau)}, indent=2))

    # -------- 2) Evaluate on test --------
    test_pat = "test_features*.parquet"
    test_shards = list(_iter_shards(cache_dir, test_pat))
    if not test_shards:
        raise SystemExit(f"No test shards matched pattern '{test_pat}' under {cache_dir}")

    y_te, s_te = _scores_for_shards(
        model,
        test_shards,
        feature_prefixes,
        spec_cols=spec_cols,
        limit_rows=None,
    )
    if y_te.size == 0:
        raise SystemExit("No test data found.")

    roc_te = roc(y_te, s_te)
    rep_te = report_at_threshold(y_te, s_te, tau)

    # -------- 3) Summary --------
    print(f"[eval-ember] tau source: {source_for_tau}")
    print(f"[eval-ember] AUC(test): {roc_te.auc:.4f}")
    print(f"[eval-ember] target_fpr: {args.target_fpr:.4f}")
    print(
        f"[eval-ember] at tau={rep_te['tau']:.6f} -> FPR={rep_te['fpr']:.4f}, "
        f"TPR={rep_te['tpr']:.4f}, Precision={rep_te['precision']:.4f}, Recall={rep_te['recall']:.4f}"
    )
    print(
        f"[eval-ember] confusion (tn, fp, fn, tp): {rep_te['tn']} {rep_te['fp']} {rep_te['fn']} {rep_te['tp']}"
    )

    # -------- 4) Save CSV row --------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {
        "model": str(args.model),
        "tau_source": source_for_tau,
        "tau": rep_te["tau"],
        "auc_test": roc_te.auc,
        "fpr_at_tau": rep_te["fpr"],
        "tpr_at_tau": rep_te["tpr"],
        "precision": rep_te["precision"],
        "recall": rep_te["recall"],
        "tn": rep_te["tn"],
        "fp": rep_te["fp"],
        "fn": rep_te["fn"],
        "tp": rep_te["tp"],
        "feature_prefixes": ";".join(feature_prefixes),
        "target_fpr": args.target_fpr,
        "used_spec": bool(spec_cols),
    }
    pd.DataFrame([row]).to_csv(args.out, index=False)
    print(f"[eval-ember] wrote {args.out}")


if __name__ == "__main__":
    main()
