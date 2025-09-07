import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Featurize PE files (stub).")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory of PE files.")
    parser.add_argument(
        "--out", type=Path, required=True, help="Output features path (parquet/csv)."
    )
    args = parser.parse_args()

    # TODO: import avlab.features.pe_static and write a dataframe here.
    print("[featurize] (stub) scanning:", args.input_dir)
    print("[featurize] (stub) writing features to:", args.out)


if __name__ == "__main__":
    main()
