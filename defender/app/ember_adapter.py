from __future__ import annotations

from typing import Any


def _byte_histogram(bytez: bytes) -> dict[str, float]:
    """256-bin histogram normalized to probabilities; keys: histogram.000..255"""
    counts = [0] * 256
    for b in bytez:
        counts[b] += 1
    n = float(len(bytez)) if bytez else 1.0
    width = 3
    return {f"histogram.{i:0{width}d}": (counts[i] / n) for i in range(256)}


def _strings_summary(bytez: bytes) -> dict[str, float]:
    """
    Placeholder numeric summaries for strings.* family.
    DOES NOT emit per-string features; just a few scalars to avoid empty dict.
    """
    runs = []
    cur = 0
    for b in bytez:
        if 32 <= b <= 126:  # printable ASCII
            cur += 1
        else:
            if cur >= 5:
                runs.append(cur)
            cur = 0
    if cur >= 5:
        runs.append(cur)

    total = len(runs)
    avg_len = (sum(runs) / total) if total else 0.0
    max_len = max(runs) if runs else 0.0
    return {
        "strings.count": float(total),
        "strings.avg_len": float(avg_len),
        "strings.max_len": float(max_len),
    }


def bytes_to_numeric_features(bytez: bytes) -> dict[str, Any]:
    """
    Build a numeric feature dict whose keys match the training column names.
    Currently:
      - histogram.000..255 : real values
      - strings.*          : a few summary scalars (names may not exist in spec, harmless)
    For all other training columns, the predictor will fill zeros.
    """
    feats: dict[str, Any] = {}
    feats.update(_byte_histogram(bytez))
    feats.update(_strings_summary(bytez))
    # Placeholders for other families; predictor will insert zeros for any missing columns.
    return feats
