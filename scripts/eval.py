from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

load_dotenv()


def _expand(s: str) -> Path:
    return Path(os.path.expandvars(s)).expanduser().resolve()


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on features.")
    parser.add_argument("--model", type=str, default="${ARTIFACTS_DIR}/baseline.joblib")
    parser.add_argument("--features", type=str, default="${OUTPUTS_DIR}/features.parquet")
    args = parser.parse_args()

    model_path = _expand(args.model)
    feat_path = _expand(args.features)

    model = joblib.load(model_path)
    df = read_features(feat_path)

    X = df[["n_imports"]].values
    y = df["label"].values

    yhat = model.predict(X)
    acc = accuracy_score(y, yhat)
    print(f"[eval] accuracy: {acc:.4f}")
    print("[eval] confusion matrix (tn, fp, fn, tp):")
    print(confusion_matrix(y, yhat).ravel())
    print("[eval] report:")
    print(classification_report(y, yhat, digits=4))


def read_features(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    else:
        return pd.read_parquet(path)


if __name__ == "__main__":
    main()
