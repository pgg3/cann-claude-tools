"""
CANN Claude Tools - Ascend C operator generation for Claude Code.
"""

__version__ = "0.1.0"

from .config import get_config, CANNConfig


def check_evotoolkit() -> bool:
    """Check if evotoolkit is available."""
    try:
        from evotoolkit.task.cann_init import CANNInitTask
        return True
    except ImportError:
        return False


def ensure_evotoolkit():
    """Ensure evotoolkit is installed, raise helpful error if not."""
    if not check_evotoolkit():
        raise ImportError(
            "evotoolkit not installed or cann_init module not available.\n"
            "Please install it with:\n"
            "  pip install -e ./evotoolkit[cann_init]\n"
            "Or run the install script:\n"
            "  ./install.sh"
        )


__all__ = [
    "__version__",
    "get_config",
    "CANNConfig",
    "check_evotoolkit",
    "ensure_evotoolkit",
]
