"""
CANN Iteration History Management.

Utilities for managing iteration history and solution files.
"""

import json
import shutil
from pathlib import Path
from typing import Dict

from .evaluator import save_solution_files


def load_history(output_dir: Path) -> Dict:
    """Load iteration history from file.

    Returns a dict with structure:
    {
        "config": {...},
        "summary": {"total": 0, "successful": 0, "best_iteration": None},
        "iterations": [...]
    }
    """
    history_path = output_dir / "iteration_history.json"
    if history_path.exists():
        try:
            return json.loads(history_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "config": {},
        "summary": {"total": 0, "successful": 0, "best_iteration": None},
        "iterations": []
    }


def save_history(output_dir: Path, history: Dict):
    """Save iteration history to file."""
    history_path = output_dir / "iteration_history.json"
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False))


def save_iteration_solution(output_dir: Path, solution: Dict, iteration: int) -> Path:
    """Save solution to iteration-specific directory.

    Creates:
        output_dir/solution-{iteration}/
        ├── solution.json
        ├── kernel_impl.cpp
        ├── kernel_entry.cpp
        ├── tiling_fields.json
        ├── tiling_func.cpp
        ├── infer_shape.cpp
        └── output_alloc.cpp

    Returns:
        Path to the iteration directory
    """
    iter_dir = output_dir / f"solution-{iteration}"
    save_solution_files(solution, str(iter_dir))
    return iter_dir


def save_best_solution(output_dir: Path, solution: Dict, iteration: int):
    """Save best solution to best_solution directory."""
    best_dir = output_dir / "best_solution"

    # Remove existing best_solution if exists
    if best_dir.exists():
        shutil.rmtree(best_dir)

    # Copy from iteration directory
    iter_dir = output_dir / f"solution-{iteration}"
    if iter_dir.exists():
        # Use symlinks=True to preserve symlinks, ignore_dangling_symlinks=True
        # to handle broken symlinks (common in sandbox compilation)
        shutil.copytree(iter_dir, best_dir, symlinks=True, ignore_dangling_symlinks=True)
    else:
        # Fallback: save directly
        save_solution_files(solution, str(best_dir))
