"""
CANN Tools MCP Server.

Provides tools for:
- Searching AscendC APIs
- Finding operator examples
- Getting knowledge summaries

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


# Lazy-loaded knowledge base
_knowledge_base = None


class StubKnowledgeBase:
    """Stub knowledge base when evotoolkit is not available."""

    def search_api(self, name: str) -> dict:
        return {
            "status": "stub",
            "message": "evotoolkit not installed. Install with: pip install -e ./evotoolkit",
            "api_info": None,
            "candidates": []
        }

    def search_operator(self, name: str, top_k: int = 3) -> dict:
        return {
            "status": "stub",
            "message": "evotoolkit not installed",
            "primary": None,
            "related": []
        }

    def get_available_knowledge_summary(self) -> str:
        return "Knowledge base not available. Install evotoolkit to enable."


def get_knowledge_base():
    """Lazy-load knowledge base.

    Note: Redirects stdout to stderr during initialization to prevent
    evotoolkit's progress output from polluting the MCP STDIO channel.
    """
    global _knowledge_base
    if _knowledge_base is None:
        try:
            from evotoolkit.evo_method.cann_initer.knowledge import (
                RealKnowledgeBase,
                KnowledgeBaseConfig,
            )
            # Redirect stdout to stderr during init to prevent progress output
            # from polluting MCP's JSON-RPC channel
            original_stdout = sys.stdout
            sys.stdout = sys.stderr
            try:
                config = KnowledgeBaseConfig()
                _knowledge_base = RealKnowledgeBase(config, auto_build=True)
            finally:
                sys.stdout = original_stdout
        except ImportError:
            _knowledge_base = StubKnowledgeBase()
    return _knowledge_base


async def evaluate_solution_mcp(solution_path: str) -> dict:
    """Evaluate a CANN solution via MCP.

    Reads config from environment variables set by cann-claude CLI.
    """
    from pathlib import Path
    from .evaluator import evaluate_solution, load_solution
    from .config import CANNConfig

    # Load solution
    solution = load_solution(solution_path)
    if solution is None:
        return {
            "success": False,
            "stage": "load",
            "error": f"Solution not found at: {solution_path}"
        }

    # Load config from environment
    config = CANNConfig.from_env()
    if config is None:
        return {
            "success": False,
            "stage": "config",
            "error": "CANN environment not configured. Run via 'cann-claude generate' command."
        }

    # Get python reference
    python_ref = config.python_ref_path.read_text()

    # Evaluate
    result = evaluate_solution(
        solution=solution,
        op_name=config.op_name,
        python_ref=python_ref,
        npu_type=config.npu_type,
        project_path=str(config.output_dir / "project"),
        fake_mode=config.fake_mode,
    )

    return result.to_dict()


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
                description="""Search AscendC API by name. Returns file path to header.

RETURNS:
- api_info.header_file: Path to header file containing API definition
- Use Read tool on header_file to see full signature

Example: cann_search_api("DataCopy") → header_file: "/path/to/kernel_operator_data_copy_intf.h"
Then: Read(header_file) to see DataCopy signature""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "API name (e.g., 'Add', 'DataCopy', 'ReduceSum')"
                        }
                    },
                    "required": ["name"]
                }
            ),
            Tool(
                name="cann_search_operator",
                description="""Search existing operator implementations. Returns file paths, NOT code.

RETURNS:
- primary.kernel_files: List of kernel .h/.cpp paths (may be empty for some operators)
- primary.host_files: List of host/tiling .cpp paths (may be empty)
- primary.readme_file: Path to README.md
- related: Similar operators with their paths

WORKFLOW:
1. Call this tool → get file paths
2. Read(kernel_files[0]) → see kernel implementation
3. Read(readme_file) → see documentation

Note: Some operators only have host code, no kernel code. Check if arrays are empty.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Operator name (e.g., 'relu', 'softmax', 'matmul')"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of related operators to return",
                            "default": 3
                        }
                    },
                    "required": ["name"]
                }
            ),
            Tool(
                name="cann_get_knowledge",
                description="""List all available AscendC APIs by category.

Returns categorized API list:
- Vector Compute: Add, Mul, Max, ReduceSum, ...
- Data Copy: DataCopy, DataCopyPad, ...
- Tensor: GetTensorShape, ...

Use this when you need to find which API to use for an operation.""",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            # NOTE: cann_evaluate removed - CLI handles evaluation, not Claude
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        kb = get_knowledge_base()

        if name == "cann_search_api":
            api_name = arguments.get("name", "")
            result = kb.search_api(api_name)
            # Result now includes header_file path from evotoolkit
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2, ensure_ascii=False)
            )]

        elif name == "cann_search_operator":
            op_name = arguments.get("name", "")
            top_k = arguments.get("top_k", 3)
            result = kb.search_operator(op_name, top_k)

            # Return file paths instead of code content
            # Claude can use Read tool to read files as needed
            output = {
                "confidence": result.get("confidence", "low"),
                "related": [],
            }

            primary = result.get("primary")
            if primary:
                output["primary"] = {
                    "name": primary.get("name"),
                    "repo": primary.get("repo"),
                    "category": primary.get("category"),
                    "path": primary.get("path"),
                    "kernel_files": primary.get("kernel_files", []),
                    "host_files": primary.get("host_files", []),
                    "readme_file": primary.get("readme_file"),
                }

            # Related operators with paths
            for rel in result.get("related", []):
                rel_info = {
                    "name": rel.get("name"),
                    "reason": rel.get("reason"),
                }
                if rel.get("path"):
                    rel_info["path"] = rel["path"]
                if rel.get("kernel_files"):
                    rel_info["kernel_files"] = rel["kernel_files"]
                output["related"].append(rel_info)

            return [TextContent(
                type="text",
                text=json.dumps(output, indent=2, ensure_ascii=False)
            )]

        elif name == "cann_get_knowledge":
            summary = kb.get_available_knowledge_summary()
            return [TextContent(type="text", text=summary)]

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
            server.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
