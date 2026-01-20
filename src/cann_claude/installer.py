"""
Package utilities for CANN Claude Tools.

Provides path helpers for accessing package resources.
"""

from pathlib import Path


def get_package_dir() -> Path:
    """Get the package installation directory."""
    return Path(__file__).parent


def get_mcp_server_path() -> Path:
    """Get the path to the MCP server script."""
    return get_package_dir() / "mcp_server.py"


def get_skill_template_path() -> Path:
    """Get the path to the skill template."""
    return get_package_dir() / "templates" / "skill.md"
