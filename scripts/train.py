import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train model from features (stub).")
    parser.add_argument("--config", type=Path, required=True, help="YAML config file.")
    parser.add_argument("--out", type=Path, help="Override model output path.")
    args = parser.parse_args()

    # TODO: load YAML, read features, split, train sklearn model, write joblib.
    print("[train] (stub) using config:", args.config)
    if args.out:
        print("[train] (stub) override model path:", args.out)


if __name__ == "__main__":
    main()
