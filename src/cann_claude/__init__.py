"""
CANN Claude Tools - Ascend C operator generation for Claude Code.
"""

__version__ = "0.1.0"

from .config import get_config, CANNConfig


def check_evaluator() -> bool:
    """Check if cann_parallel_evaluator is available."""
    try:
        from cann_parallel_evaluator import CANNInitTask
        return True
    except ImportError:
        return False


# Legacy alias
check_evotoolkit = check_evaluator


def ensure_evaluator():
    """Ensure cann_parallel_evaluator is installed, raise helpful error if not."""
    if not check_evaluator():
        raise ImportError(
            "cann-parallel-evaluator not installed.\n"
            "Please install it with:\n"
            "  pip install cann-parallel-evaluator\n"
        )


# Legacy alias
ensure_evotoolkit = ensure_evaluator


__all__ = [
    "__version__",
    "get_config",
    "CANNConfig",
    "check_evaluator",
    "ensure_evaluator",
    "check_evotoolkit",
    "ensure_evotoolkit",
]
