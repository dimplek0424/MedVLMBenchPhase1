"""
medvlm_core.logging
-------------------
CSV/JSON helpers and run manifest utilities (git commit, config hash).
"""

import csv, json, hashlib, subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

def write_csv(rows: Sequence[Sequence[Any]], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

def write_json(obj: Dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def git_commit_short() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"

def config_hash(cfg: Dict[str, Any]) -> str:
    """
    Stable, short hash of a config dict (sorted keys).
    """
    payload = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:8]
