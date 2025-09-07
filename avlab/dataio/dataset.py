from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class DataPaths:
    input_dir: Path
    features_out: Path


LABEL_MAP = {
    "goodware": 0,
    "benign": 0,
    "malware": 1,
    "mal": 1,
}


def discover_files(input_dir: Path) -> list[tuple[Path, int]]:
    """
    Recursively find files under input_dir.
    Label is inferred from immediate parent directory name, ex:
      data/samples/goodware/*.exe -> 0
      data/samples/malware/*.exe  -> 1
    """
    pairs: list[tuple[Path, int]] = []
    for p in input_dir.rglob("*"):
        if p.is_file():
            parent = p.parent.name.lower()
            label = LABEL_MAP.get(parent)
            if label is not None:
                pairs.append((p, label))
    return pairs


def write_features(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        df.to_parquet(out_path, index=False)
    elif out_path.suffix.lower() == ".csv":
        df.to_csv(out_path, index=False)
    else:
        # default to parquet
        df.to_parquet(out_path, index=False)


def read_features(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    else:
        return pd.read_parquet(path)
