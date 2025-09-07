from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataPaths:
    input_dir: Path
    features_out: Path


# TODO: Read/write dataframes and labels.
