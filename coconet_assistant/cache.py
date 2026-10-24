"""On-disk cache of completed ensemble runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from coconet_assistant.scenario import scenario_fingerprint

DEFAULT_CACHE_DIRNAME = ".coconet_assistant_cache"


def default_cache_dir(root: str | Path | None = None) -> Path:
    base = Path(root) if root is not None else Path.cwd()
    return base / DEFAULT_CACHE_DIRNAME


def _file_stamp(path_value: Any) -> dict[str, Any]:
    path = Path(str(path_value))
    if not path.is_file():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def run_cache_key(config: dict[str, Any]) -> str:
    """Identity of a run: scenario fields plus reefs/coastline file stamps."""
    digest = scenario_fingerprint(config)
    stamps = {
        "reefs": _file_stamp(config.get("reefs_file")),
        "coastline": _file_stamp(config.get("coastline_file")),
    }
    extra = json.dumps(stamps, sort_keys=True)
    stamp_digest = hashlib.sha256(extra.encode("utf-8")).hexdigest()[:10]
    return f"{digest[:16]}_{stamp_digest}"


def run_dir(cache_dir: Path, key: str) -> Path:
    return Path(cache_dir) / key


def load_cached_run(cache_dir: Path, key: str) -> dict[str, Any] | None:
    folder = run_dir(cache_dir, key)
    marker = folder / "run.json"
    output = folder / "output.csv"
    if not marker.is_file() or not output.is_file():
        return None
    loaded = json.loads(marker.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return None
    payload: dict[str, Any] = dict(loaded)
    payload["output_file"] = str(output)
    payload["cached"] = True
    return payload


def save_cached_run(cache_dir: Path, key: str, payload: dict[str, Any]) -> Path:
    folder = run_dir(cache_dir, key)
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / "run.json"
    marker.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return folder
