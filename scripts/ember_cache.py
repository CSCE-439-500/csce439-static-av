from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

# ---- Helpers --------------------------------------------------------------

NUMERIC = (int, float)


def flatten_numeric(record: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten ONLY numeric content of an EMBER JSON object.
    - Keeps scalar numbers (e.g., general.size)
    - Expands short numeric lists (<=512) into fixed columns (e.g., bytehist.000..255)
    - Skips strings and lists of strings (e.g., imports)
    """
    out: dict[str, Any] = {}

    def put(prefix: str, key: str, val: Any) -> None:
        out[f"{prefix}.{key}" if prefix else key] = val

    def walk(prefix: str, obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(obj, list):
            if len(obj) <= 512 and all(isinstance(x, NUMERIC) for x in obj):
                width = max(3, len(str(len(obj) - 1)))
                for i, x in enumerate(obj):
                    put(prefix, f"{i:0{width}d}", x)
            # else: skip lists of strings or long/unknown lists
        elif isinstance(obj, NUMERIC):
            put("", prefix, obj)
        # else: skip strings/None/etc.

    walk("", record)
    return out


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def shard_writer(
    df_iter: Iterable[dict[str, Any]], out_dir: Path, base: str, rows_per_file: int = 200_000
) -> list[Path]:
    """
    Write an iterator of dict rows to multiple Parquet files with up to rows_per_file rows each.
    Returns list of parquet paths written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    buf: list[dict[str, Any]] = []
    shard_idx = 0

    def flush() -> None:
        nonlocal shard_idx, buf
        if not buf:
            return
        df = pd.DataFrame.from_records(buf)
        p = out_dir / f"{base}.{shard_idx:03d}.parquet"
        df.to_parquet(p, index=False)
        paths.append(p)
        buf = []
        shard_idx += 1

    for row in df_iter:
        buf.append(row)
        if len(buf) >= rows_per_file:
            flush()
    flush()
    return paths


def build_cache_for_file(jsonl_path: Path, subset: str, out_dir: Path) -> list[Path]:
    """
    Stream a single JSONL file and write numeric-only parquet shards.
    Skips unlabeled rows (label == -1).
    """
    rows = (_row for rec in iter_jsonl(jsonl_path) if (_row := _to_row(rec, subset)) is not None)
    base = jsonl_path.stem  # e.g., train_features_0
    return shard_writer(rows, out_dir, base=base)


def _to_row(rec: dict[str, Any], subset: str) -> dict[str, Any] | None:
    """
    Convert a raw EMBER JSON record to a flat numeric row with label+subset.
    Returns None for unlabeled rows (label == -1).
    """
    label = rec.get("label", None)
    if label is None or int(label) == -1:
        return None
    flat = flatten_numeric(rec)
    flat["label"] = int(label)
    flat["subset"] = subset
    return flat


# ---- CLI ------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache EMBER JSONL to Parquet shards (numeric features only)."
    )
    parser.add_argument(
        "--in_dir",
        type=Path,
        default=os.getenv("EMBER_DIR"),
        help="Directory with EMBER JSONL shards.",
    )
    parser.add_argument(
        "--out_dir", type=Path, default=os.getenv("EMBER_CACHE"), help="Output cache directory."
    )
    parser.add_argument(
        "--rows-per-file", type=int, default=200_000, help="Max rows per Parquet shard."
    )
    args = parser.parse_args()

    if not args.in_dir or not args.out_dir:
        raise SystemExit("Set --in_dir and --out_dir or EMBER_DIR / EMBER_CACHE env vars.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Train shards (train_features_*.jsonl)
    train_paths = sorted(args.in_dir.glob("train_features_*.jsonl"))
    # Test shard (test_features.jsonl)
    test_paths = sorted(args.in_dir.glob("test_features.jsonl"))

    if not train_paths and not test_paths:
        raise SystemExit(f"No EMBER JSONL shards found in {args.in_dir}")

    print(f"[ember-cache] in_dir={args.in_dir}")
    print(f"[ember-cache] out_dir={args.out_dir}")

    written: list[Path] = []
    for jp in train_paths:
        print(f"[ember-cache] train: {jp.name}")
        written += build_cache_for_file(jp, subset="train", out_dir=args.out_dir)

    for jp in test_paths:
        print(f"[ember-cache] test:  {jp.name}")
        written += build_cache_for_file(jp, subset="test", out_dir=args.out_dir)

    # Quick summary
    total_rows = 0
    for p in written:
        try:
            total_rows += len(pd.read_parquet(p))
        except Exception:
            pass
    print(f"[ember-cache] wrote {len(written)} shards, ~{total_rows} labeled rows.")


if __name__ == "__main__":
    main()
