from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on features.")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained model (joblib).")
    parser.add_argument(
        "--features", type=Path, required=True, help="Path to features parquet/csv."
    )
    args = parser.parse_args()

    model = joblib.load(args.model)
    df = read_features(args.features)

    X = df[["n_imports"]].values
    y = df["label"].values

    yhat = model.predict(X)
    acc = accuracy_score(y, yhat)
    print(f"[eval] accuracy: {acc:.4f}")
    print("[eval] confusion matrix (tn, fp, fn, tp):")
    print(confusion_matrix(y, yhat).ravel())
    print("[eval] report:")
    print(classification_report(y, yhat, digits=4))


def read_features(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    else:
        return pd.read_parquet(path)


if __name__ == "__main__":
    main()
