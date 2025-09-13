from __future__ import annotations

from pathlib import Path
from typing import Any

import pefile


def extract_pe_features(path: Path) -> dict[str, Any]:
    """
    Minimal static features matching demo.py exactly:
    - n_imports: len(pe.DIRECTORY_ENTRY_IMPORT)
    Also returns:
    - parsed: whether PE parsing succeeded
    """
    parsed = True
    try:
        pe = pefile.PE(str(path))
        n_imports = len(getattr(pe, "DIRECTORY_ENTRY_IMPORT", []))
    except Exception:
        # If the file cannot be parsed as PE, set imports to 0.
        parsed = False
        n_imports = 0
    return {
        "path": str(path),
        "n_imports": int(n_imports),
        "parsed": bool(parsed),
    }
