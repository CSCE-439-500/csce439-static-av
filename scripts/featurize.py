from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from avlab.dataio.dataset import discover_files, write_features
from avlab.features.pe_static import extract_pe_features
from avlab.utils.paths import ensure_dirs

load_dotenv()


def _expand(s: str) -> Path:
    return Path(os.path.expandvars(s)).expanduser().resolve()


def main():
    parser = argparse.ArgumentParser(description="Featurize PE files (using env).")
    parser.add_argument(
        "--input-dir", type=str, default="${SAMPLES_DIR}", help="Directory of PE files."
    )
    parser.add_argument(
        "--out", type=str, default="${OUTPUTS_DIR}/features.parquet", help="Output path."
    )
    args = parser.parse_args()

    ensure_dirs()
    input_dir = _expand(args.input_dir)
    out_path = _expand(args.out)

    pairs = discover_files(input_dir)
    rows = []
    for path, label in pairs:
        feats = extract_pe_features(path)
        feats["label"] = label
        rows.append(feats)

    if not rows:
        print(f"[featurize] no labeled files found under {input_dir}")
        return

    df = pd.DataFrame(rows)
    write_features(df, out_path)
    print(f"[featurize] wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
