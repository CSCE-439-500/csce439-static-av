from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from avlab.models.sklearn_baseline import SklearnBaselineModel

load_dotenv()


def _expand(s: str) -> Path:
    return Path(os.path.expandvars(s)).expanduser().resolve()


def main():
    parser = argparse.ArgumentParser(description="Train model from features.")
    parser.add_argument("--config", type=Path, required=True, help="YAML config file.")
    parser.add_argument("--out", type=Path, help="Override model output path.")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    feat_path = _expand(cfg["data"]["features_out"])
    out_model = (
        _expand(cfg["train"]["out_model_path"]) if not args.out else Path(args.out).resolve()
    )

    df = read_features(feat_path=feat_path)
    X = df[["n_imports"]].values
    y = df["label"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=cfg["train"]["test_size"],
        stratify=y if cfg["train"]["stratify"] else None,
        random_state=cfg["random_seed"],
    )

    model = SklearnBaselineModel(random_state=cfg["random_seed"])
    model.fit(X_tr, y_tr)

    # quick report
    from sklearn.metrics import accuracy_score

    acc = accuracy_score(y_te, model.predict(X_te))
    print(f"[train] held-out accuracy: {acc:.4f}")

    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_model)
    print(f"[train] saved model to: {out_model}")


def read_features(feat_path: Path) -> pd.DataFrame:
    if feat_path.suffix.lower() == ".parquet":
        return pd.read_parquet(feat_path)
    elif feat_path.suffix.lower() == ".csv":
        return pd.read_csv(feat_path)
    else:
        return pd.read_parquet(feat_path)


if __name__ == "__main__":
    main()
