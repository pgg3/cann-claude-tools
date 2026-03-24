"""
CANN Tools MCP Server.

Provides tools for:
- Searching AscendC APIs
- Getting curated operator examples
- Listing available APIs by category

Run with:
    python -m cann_claude.mcp_server

Or configure in Claude Code MCP settings.
"""

import json
import sys
from typing import Any

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


# Lazy-loaded knowledge provider
_provider = None


def _load_module_from_file(module_name: str, file_path):
    """Load a Python module from file and register it in sys.modules."""
    import importlib.util

    if module_name in sys.modules:
        return sys.modules[module_name]

    file_path = str(file_path)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    # For packages (__init__.py), set __path__ so submodule imports work
    if file_path.endswith("__init__.py"):
        import os
        module.__path__ = [os.path.dirname(file_path)]

    spec.loader.exec_module(module)
    return module


def _import_knowledge_provider():
    """Import CANNKnowledgeProvider without triggering heavy imports.

    Direct ``from cann_parallel_evaluator import CANNKnowledgeProvider``
    would execute __init__.py which pulls in the evaluator, torch, and
    other dependencies that are unnecessary for the MCP server.

    We locate the package on disk via importlib.util.find_spec (no import)
    and load each knowledge submodule directly, completely bypassing the
    package init chain.

    On Windows, also patches Path.read_text to default to UTF-8 encoding
    since the knowledge .md files are UTF-8 but the system locale may not be.
    """
    import importlib.util
    import pathlib
    import platform
    from pathlib import Path

    # On Windows, patch Path.read_text to default to UTF-8.
    if platform.system() == "Windows":
        _orig_read_text = pathlib.Path.read_text

        def _utf8_read_text(self, encoding=None, errors=None):
            return _orig_read_text(self, encoding=encoding or "utf-8", errors=errors)

        pathlib.Path.read_text = _utf8_read_text

    # Find cann_parallel_evaluator package root without importing it
    spec = importlib.util.find_spec("cann_parallel_evaluator")
    if spec is None or spec.origin is None:
        raise ImportError("cann-parallel-evaluator package not found")
    pkg_root = Path(spec.origin).parent
    knowledge_dir = pkg_root / "knowledge"

    if not knowledge_dir.exists():
        raise ImportError(f"Knowledge directory not found: {knowledge_dir}")

    # Register lightweight stub modules for parent packages so that relative
    # imports inside knowledge submodules resolve correctly without triggering
    # the heavy cann_parallel_evaluator/__init__.py (which imports torch etc.)
    import types
    if "cann_parallel_evaluator" not in sys.modules:
        stub = types.ModuleType("cann_parallel_evaluator")
        stub.__path__ = [str(pkg_root)]
        stub.__package__ = "cann_parallel_evaluator"
        stub.__spec__ = spec
        sys.modules["cann_parallel_evaluator"] = stub

    # Register a stub for cann_parallel_evaluator.knowledge (will be replaced below)
    if "cann_parallel_evaluator.knowledge" not in sys.modules:
        k_stub = types.ModuleType("cann_parallel_evaluator.knowledge")
        k_stub.__path__ = [str(knowledge_dir)]
        k_stub.__package__ = "cann_parallel_evaluator.knowledge"
        sys.modules["cann_parallel_evaluator.knowledge"] = k_stub

    # Load internal dependencies first (api_scanner, examples, primers)
    # that provider.py imports via relative imports.

    # 1. api_scanner
    _load_module_from_file(
        "cann_parallel_evaluator.knowledge.api_scanner",
        knowledge_dir / "api_scanner.py",
    )

    # 2. examples sub-package
    _load_module_from_file(
        "cann_parallel_evaluator.knowledge.examples.curated_examples",
        knowledge_dir / "examples" / "curated_examples.py",
    )
    _load_module_from_file(
        "cann_parallel_evaluator.knowledge.examples",
        knowledge_dir / "examples" / "__init__.py",
    )

    # 3. primers sub-package (needs level0 + level1 loaded first)
    _load_module_from_file(
        "cann_parallel_evaluator.knowledge.primers.level0_programming_model",
        knowledge_dir / "primers" / "level0_programming_model.py",
    )
    _load_module_from_file(
        "cann_parallel_evaluator.knowledge.primers.level1_patterns",
        knowledge_dir / "primers" / "level1_patterns.py",
    )
    _load_module_from_file(
        "cann_parallel_evaluator.knowledge.primers",
        knowledge_dir / "primers" / "__init__.py",
    )

    # 4. knowledge __init__ (registers the package namespace)
    _load_module_from_file(
        "cann_parallel_evaluator.knowledge",
        knowledge_dir / "__init__.py",
    )

    # 5. provider module
    mod = _load_module_from_file(
        "cann_parallel_evaluator.knowledge.provider",
        knowledge_dir / "provider.py",
    )

    return mod.CANNKnowledgeProvider


def get_provider():
    """Lazy-load knowledge provider.

    Note: Redirects stdout to stderr during initialization to prevent
    cann_parallel_evaluator's progress output from polluting the MCP STDIO channel.
    """
    global _provider
    if _provider is None:
        try:
            CANNKnowledgeProvider = _import_knowledge_provider()

            # Redirect stdout to stderr during init to prevent progress output
            # from polluting MCP's JSON-RPC channel
            original_stdout = sys.stdout
            sys.stdout = sys.stderr
            try:
                _provider = CANNKnowledgeProvider()
            finally:
                sys.stdout = original_stdout
        except (ImportError, ModuleNotFoundError):
            _provider = _StubProvider()
    return _provider


class _StubProvider:
    """Stub provider when cann-parallel-evaluator is not available."""

    def search_api(self, name: str) -> dict:
        return {
            "status": "stub",
            "message": "cann-parallel-evaluator not installed. Install with: pip install cann-parallel-evaluator",
            "api_info": None,
            "candidates": [],
        }

    def list_apis(self) -> dict:
        return {"error": "cann-parallel-evaluator not installed"}

    def get_example(self, pattern: str):
        return None


def create_server() -> "Server":
    """Create and configure the MCP server."""
    if not MCP_AVAILABLE:
        raise ImportError("MCP SDK not installed. Install with: pip install mcp")

    server = Server("cann-tools")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="cann_search_api",
                description="""Search AscendC API by name.

Returns:
- status: "found", "not_found", or "ambiguous"
- api_info: {name, category, description, header} when found
- candidates: list of similar API names when ambiguous

Example: cann_search_api("DataCopy") -> found, header: "kernel_operator_data_copy_intf.h"
Example: cann_search_api("Reduce") -> ambiguous, candidates: ["ReduceMax", "ReduceMin", "ReduceSum"]""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "API name (e.g., 'Add', 'DataCopy', 'ReduceSum')",
                        }
                    },
                    "required": ["name"],
                },
            ),
            Tool(
                name="cann_get_example",
                description="""Get a curated complete AscendC operator example for a compute pattern.

Available patterns:
- Vector: elementwise, reduction, softmax, broadcast, pooling
- Cube: matmul, convolution, attention
- Mixed: normalization, index, resize

Each example includes all 6 components: kernel_impl, kernel_entry_body,
tiling_fields, tiling_func_body, infer_shape_body, output_alloc_code.

Returns the full example content, or null if no example exists for the pattern.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Compute pattern (e.g., 'elementwise', 'matmul', 'normalization')",
                        }
                    },
                    "required": ["pattern"],
                },
            ),
            Tool(
                name="cann_get_knowledge",
                description="""List all available AscendC APIs grouped by category.

Returns categorized API list:
- Vector Compute: Add, Mul, Max, ReduceSum, ...
- Vector Data: Duplicate, DataCopy, Cast, ...
- Cube/Matrix: Mmad, Gemm, ...
- Data Movement: DataCopy, DataCopyPad, ...
- Pipe & Buffer: TPipe, TQue, TBuf, ...

Use this when you need to find which API to use for an operation.""",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        provider = get_provider()

        if name == "cann_search_api":
            api_name = arguments.get("name", "")
            result = provider.search_api(api_name)
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2, ensure_ascii=False),
            )]

        elif name == "cann_get_example":
            pattern = arguments.get("pattern", "")
            example = provider.get_example(pattern)
            if example:
                return [TextContent(type="text", text=example)]
            else:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "not_found",
                        "message": f"No curated example for pattern '{pattern}'",
                        "available_patterns": [
                            "elementwise", "reduction", "softmax", "broadcast",
                            "pooling", "matmul", "convolution", "attention",
                            "normalization", "index", "resize",
                        ],
                    }, indent=2),
                )]

        elif name == "cann_get_knowledge":
            result = provider.list_apis()
            # Format as readable text
            lines = ["# Available AscendC APIs\n"]
            for group_name, apis in result.items():
                lines.append(f"## {group_name}")
                lines.append(", ".join(apis))
                lines.append("")
            return [TextContent(type="text", text="\n".join(lines))]

        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    return server


async def main():
    """Run the MCP server."""
    if not MCP_AVAILABLE:
        print("MCP SDK not installed. Install with: pip install mcp", file=sys.stderr)
        sys.exit(1)

    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
