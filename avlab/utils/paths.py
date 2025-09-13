from __future__ import annotations

import os
from pathlib import Path

try:
    #  Load .env if python-dotenv is available, otherwise, rely on OS env
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


def _p(val: str | None, default: Path) -> Path:
    return Path(os.path.expandvars(val)).expanduser().resolve() if val else default.resolve()


# If no .env present put data root next to the repo (../csce439-av-data)
_DEFAULT_DATA_ROOT = Path.cwd() / ".." / "csce439-av-data"

DATA_ROOT = _p(os.getenv("DATA_ROOT"), _DEFAULT_DATA_ROOT)
SAMPLES_DIR = _p(os.getenv("SAMPLES_DIR"), DATA_ROOT / "samples")
ARTIFACTS_DIR = _p(os.getenv("ARTIFACTS_DIR"), DATA_ROOT / "artifacts")
OUTPUTS_DIR = _p(os.getenv("OUTPUTS_DIR"), DATA_ROOT / "outputs")


def ensure_dirs() -> None:
    for d in (DATA_ROOT, SAMPLES_DIR, ARTIFACTS_DIR, OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
