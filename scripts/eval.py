import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on features (stub).")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained model (joblib).")
    parser.add_argument(
        "--features", type=Path, required=True, help="Path to features parquet/csv."
    )
    args = parser.parse_args()

    # TODO: load model + features, compute metrics, print report.
    print("[eval] (stub) model:", args.model)
    print("[eval] (stub) features:", args.features)


if __name__ == "__main__":
    main()
