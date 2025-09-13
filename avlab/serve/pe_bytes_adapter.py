from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from avlab.features.pe_static import extract_pe_features


def bytes_to_features(bytez: bytes) -> pd.DataFrame:
    """
    Write bytes to a temp file, reuse the PE extractor.
    """
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(suffix=".bin", delete=True) as tf:
        tf.write(bytez)
        tf.flush()
        feats: dict[str, Any] = extract_pe_features(Path(tf.name))
    return pd.DataFrame([feats])
