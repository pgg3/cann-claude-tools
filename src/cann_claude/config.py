"""
Configuration management for CANN Claude Tools.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CANNConfig:
    """Configuration for CANN operator generation."""

    op_name: str
    output_dir: Path
    python_ref_path: Path
    max_iterations: int = 10
    npu_type: str = "Ascend910B"
    fake_mode: bool = False

    @classmethod
    def from_env(cls) -> Optional["CANNConfig"]:
        """Load configuration from environment variables."""
        op_name = os.environ.get("CANN_OP_NAME")
        output_dir = os.environ.get("CANN_OUTPUT_DIR")
        python_ref = os.environ.get("CANN_PYTHON_REF")

        if not all([op_name, output_dir, python_ref]):
            return None

        return cls(
            op_name=op_name,
            output_dir=Path(output_dir),
            python_ref_path=Path(python_ref),
            max_iterations=int(os.environ.get("CANN_MAX_ITERATIONS", "10")),
            npu_type=os.environ.get("CANN_NPU_TYPE", "Ascend910B"),
            fake_mode=os.environ.get("CANN_FAKE_MODE", "").lower() == "true",
        )

    def to_env(self) -> dict[str, str]:
        """Export configuration to environment variables dict."""
        return {
            "CANN_OP_NAME": self.op_name,
            "CANN_OUTPUT_DIR": str(self.output_dir),
            "CANN_PYTHON_REF": str(self.python_ref_path),
            "CANN_MAX_ITERATIONS": str(self.max_iterations),
            "CANN_NPU_TYPE": self.npu_type,
            "CANN_FAKE_MODE": str(self.fake_mode).lower(),
        }

    def set_env(self):
        """Set environment variables from this config."""
        for key, value in self.to_env().items():
            os.environ[key] = value


def get_config() -> Optional[CANNConfig]:
    """Get current configuration from environment."""
    return CANNConfig.from_env()


# Default paths
def get_claude_config_dir() -> Path:
    """Get Claude Code config directory."""
    return Path.home() / ".claude"


def get_project_claude_dir() -> Path:
    """Get project-local .claude directory."""
    return Path.cwd() / ".claude"
