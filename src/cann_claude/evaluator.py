"""
CANN Solution Evaluator.

Core evaluation logic shared by:
- CLI command (cann-claude generate, cann-claude evaluate)

Uses CANNInitTask from cann_parallel_evaluator to handle compilation, correctness, and
performance testing in isolated subprocess, preventing environment pollution and segfaults.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class EvaluationResult:
    """Result of evaluating a CANN solution."""
    success: bool
    stage: str  # "success", "compile", "correctness", "performance", etc.
    error: Optional[str] = None
    runtime_ms: Optional[float] = None
    speedup: Optional[float] = None
    score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "success": self.success,
            "stage": self.stage,
        }
        if self.error:
            result["error"] = self.error
        if self.runtime_ms is not None:
            result["runtime_ms"] = self.runtime_ms
        if self.speedup is not None:
            result["speedup"] = self.speedup
        if self.score is not None:
            result["score"] = self.score
        return result



def evaluate_solution(
    solution: Dict[str, Any],
    op_name: str,
    python_ref: str,
    npu_type: str = "Ascend910B",
    project_path: Optional[str] = None,
    fake_mode: bool = False,
    save_to: Optional[str] = None,
) -> EvaluationResult:
    """Evaluate a CANN solution using CANNInitTask.

    Args:
        solution: Solution dict with 4 complete source files
        op_name: Operator name
        python_ref: Python reference code
        npu_type: NPU type (default: Ascend910B)
        project_path: Path for compilation artifacts
        fake_mode: Skip actual compilation
        save_to: Save solution files to this directory

    Returns:
        EvaluationResult with success status and metrics
    """
    # Save solution files if requested
    if save_to:
        save_solution_files(solution, save_to)

    # Fake mode - return mock success
    if fake_mode:
        return EvaluationResult(
            success=True,
            stage="success",
            runtime_ms=0.1,
            speedup=10.0,
            score=-0.1,
        )

    try:
        from cann_parallel_evaluator import CANNInitTask, CANNSolutionConfig, Solution

        # Set restrictive umask - msopgen requires files not writable by group/others
        old_umask = os.umask(0o022)

        try:
            # Ensure project_path exists
            if project_path is None:
                import tempfile
                project_path = tempfile.mkdtemp(prefix=f"cann_{op_name}_")
            Path(project_path).mkdir(parents=True, exist_ok=True)

            # Create task data
            task_data = {
                "op_name": op_name,
                "npu_type": npu_type,
                "python_reference": python_ref,
            }

            # Create CANNInitTask
            task = CANNInitTask(data=task_data, fake_mode=False)

            # Create solution config
            config = CANNSolutionConfig(
                project_path=project_path,
                op_kernel=solution.get("op_kernel", ""),
                op_host_tiling=solution.get("op_host_tiling", ""),
                op_host=solution.get("op_host", ""),
                pybinding=solution.get("pybinding", ""),
            )

            # Wrap in Solution object
            sol = Solution(sol_string="", other_info=config.to_dict())

            # Evaluate using task
            result = task.evaluate_solution(sol)

            # Convert to EvaluationResult
            if result.valid:
                runtime = result.additional_info.get("runtime")
                speedup = result.additional_info.get("speedup")

                return EvaluationResult(
                    success=True,
                    stage=result.additional_info.get("stage", "success"),
                    runtime_ms=runtime,
                    speedup=speedup,
                    score=result.score,
                )
            else:
                return EvaluationResult(
                    success=False,
                    stage=result.additional_info.get("stage", "unknown"),
                    error=result.additional_info.get("error"),
                )

        finally:
            # Restore original umask
            os.umask(old_umask)

    except ImportError as e:
        return EvaluationResult(
            success=False,
            stage="import_error",
            error=f"cann-parallel-evaluator not available: {e}",
        )
    except Exception as e:
        import traceback
        return EvaluationResult(
            success=False,
            stage="exception",
            error=f"{str(e)}\n{traceback.format_exc()}",
        )


def save_solution_files(solution: Dict[str, Any], output_dir: str) -> None:
    """Save solution components to individual files.

    Creates:
        output_dir/
        ├── solution.json       # Full solution
        ├── op_kernel.cpp       # Kernel implementation + entry
        ├── op_host_tiling.h    # TilingData struct
        ├── op_host.cpp         # TilingFunc / InferShape / OpDef
        └── pybinding.cpp       # CppExtension binding
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    # Save full solution
    (path / "solution.json").write_text(
        json.dumps(solution, indent=2, ensure_ascii=False)
    )

    # Save individual components
    components = {
        "op_kernel.cpp": solution.get("op_kernel", ""),
        "op_host_tiling.h": solution.get("op_host_tiling", ""),
        "op_host.cpp": solution.get("op_host", ""),
        "pybinding.cpp": solution.get("pybinding", ""),
    }

    for filename, content in components.items():
        if content:
            (path / filename).write_text(content)


def load_solution(solution_path: str) -> Optional[Dict[str, Any]]:
    """Load solution from JSON file or directory.

    Args:
        solution_path: Path to solution.json or directory containing it

    Returns:
        Solution dict or None if not found
        If parse error, returns {"_error": "..."}
    """
    path = Path(solution_path)

    if path.is_dir():
        path = path / "solution.json"

    if not path.exists():
        return None

    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return {"_error": f"Failed to parse solution.json: {e}"}
