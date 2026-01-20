"""
CANN Experience Manager.

Simple file-based storage:
- errors/: CLI records raw errors (JSON)
- tips/: Claude generates analysis (Markdown)
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


# Output directory for current run
_output_dir: Optional[Path] = None


def set_output_dir(path: Optional[Path]) -> None:
    """Set output directory for experience files."""
    global _output_dir
    _output_dir = path


def get_experience_dir() -> Path:
    """Get global experience directory."""
    cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_dir / "cann-claude" / "experience"


def _get_errors_dir() -> Path:
    """Get errors directory, create if needed."""
    d = get_experience_dir() / "errors"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_next_error_id() -> int:
    """Get next error ID based on existing files."""
    errors_dir = _get_errors_dir()
    existing = list(errors_dir.glob("*.json"))
    if not existing:
        return 1
    ids = []
    for f in existing:
        try:
            ids.append(int(f.stem.split("_")[0]))
        except (ValueError, IndexError):
            pass
    return max(ids, default=0) + 1


def record_error(op_name: str, stage: str, error_msg: str) -> Path:
    """Record error to file. Returns the file path."""
    error_id = _get_next_error_id()
    filename = f"{error_id:03d}_{op_name}_{stage}.json"

    data = {
        "id": error_id,
        "op": op_name,
        "stage": stage,
        "error": error_msg,
        "time": datetime.now().isoformat(),
    }

    # Save to global experience dir
    global_path = _get_errors_dir() / filename
    global_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    # Also save to output dir if set
    if _output_dir:
        out_errors = _output_dir / "experience" / "errors"
        out_errors.mkdir(parents=True, exist_ok=True)
        (out_errors / filename).write_text(json.dumps(data, indent=2, ensure_ascii=False))

    return global_path


def record_optimization(
    op_name: str,
    before_ms: float,
    after_ms: float,
    description: str = "",
) -> None:
    """Record optimization result."""
    tips_dir = get_experience_dir() / "tips"
    tips_dir.mkdir(parents=True, exist_ok=True)

    speedup = before_ms / after_ms if after_ms > 0 else 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"opt_{op_name}_{timestamp}.json"

    data = {
        "op": op_name,
        "before_ms": before_ms,
        "after_ms": after_ms,
        "speedup": round(speedup, 2),
        "description": description,
        "time": datetime.now().isoformat(),
    }

    (tips_dir / filename).write_text(json.dumps(data, indent=2, ensure_ascii=False))

    # Also save to output dir
    if _output_dir:
        out_tips = _output_dir / "experience" / "tips"
        out_tips.mkdir(parents=True, exist_ok=True)
        (out_tips / filename).write_text(json.dumps(data, indent=2, ensure_ascii=False))


def sync_tips_to_global() -> None:
    """Sync tips from output directory to global cache.

    This ensures tips created by Claude (markdown files) are persisted
    to the global cache for future runs.
    """
    if _output_dir is None:
        return

    out_tips = _output_dir / "experience" / "tips"
    if not out_tips.exists():
        return

    global_tips = get_experience_dir() / "tips"
    global_tips.mkdir(parents=True, exist_ok=True)

    for f in out_tips.glob("*"):
        if f.is_file():
            dest = global_tips / f.name
            if not dest.exists():
                dest.write_text(f.read_text())
