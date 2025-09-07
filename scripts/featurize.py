from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from avlab.dataio.dataset import discover_files, write_features
from avlab.features.pe_static import extract_pe_features


def main():
    parser = argparse.ArgumentParser(description="Featurize PE files (demo-equivalent).")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory of PE files (expects subfolders named e.g. goodware/, malware/).",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output features path (.parquet or .csv)."
    )
    args = parser.parse_args()

    pairs = discover_files(args.input_dir)
    rows = []
    for path, label in pairs:
        feats = extract_pe_features(path)
        feats["label"] = label
        rows.append(feats)

    if not rows:
        print(
            "[featurize] no labeled files found; expected subfolders like 'goodware' and 'malware'."
        )
        return

    df = pd.DataFrame(rows)
    write_features(df, args.out)
    print(f"[featurize] wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
