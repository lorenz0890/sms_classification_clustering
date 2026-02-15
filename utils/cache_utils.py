from __future__ import annotations

import shutil
from pathlib import Path


def clear_dir(path: Path) -> None:
    """Remove all entries inside the directory."""
    if not path.exists():
        print(f"cache directory not found: {path}")
        return
    if not path.is_dir():
        print(f"not a directory: {path}")
        return
    for entry in path.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    print(f"cleared {path}")
