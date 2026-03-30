from __future__ import annotations

from pathlib import Path
from typing import Dict
import os


def load_env_file(path: Path) -> Dict[str, str]:
    loaded: Dict[str, str] = {}

    if not path.exists():
        return loaded

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_matching_quotes(value.strip())
        if not key:
            continue

        os.environ.setdefault(key, value)
        loaded[key] = value

    return loaded


def _strip_matching_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value

