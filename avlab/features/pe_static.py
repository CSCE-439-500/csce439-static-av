from __future__ import annotations

from pathlib import Path
from typing import Any

import pefile


def extract_pe_features(path: Path) -> dict[str, Any]:
    """
    Minimal static features matching demo.py exactly:
    - n_imports: len(pe.DIRECTORY_ENTRY_IMPORT)
    """
    try:
        pe = pefile.PE(str(path))
        n_imports = len(getattr(pe, "DIRECTORY_ENTRY_IMPORT", []))
    except Exception:
        # TODO: Consider proper exception handling for parse failures.
        # If the file cannot be parsed as PE, mimic "worst case" and set imports to 0.
        n_imports = 0
    return {
        "path": str(path),
        "n_imports": int(n_imports),
    }
