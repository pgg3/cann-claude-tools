"""
CANN Prompt Templates.

Provides prompt generation for Claude interactions during operator generation.
"""

from pathlib import Path
from typing import Optional


def build_initial_prompt(
    op_name: str,
    npu_type: str,
    output_path: Path,
    ref_path: Path,
) -> str:
    """Build the initial generation prompt for iteration 1."""
    # Determine if cube or vector based on op name
    cube_ops = {"matmul", "mat_mul", "batch_matmul", "gemm", "conv2d", "conv3d", "conv1d", "bmm"}
    is_cube = op_name.lower().replace("-", "_") in cube_ops
    guide_file = "cube.md" if is_cube else "vector.md"

    return f"""Generate Ascend C operator `{op_name}` for {npu_type}.

**Step 1: Research APIs**
- `cann_get_knowledge()` - list available APIs
- `cann_search_operator("{op_name}")` - find similar implementations

**Step 2: Read these files** (in this order):
1. `{output_path}/constraints.md` - **CRITICAL**: Code template structure & JSON format
2. `{output_path}/{guide_file}` - Hardware specs & critical rules for this operator type
3. `{output_path}/signature.json` - Operator interface
4. `{ref_path}` - Python reference
5. `{output_path}/solution_template.json` - Example format

**Step 3: Write solution**
Use the Write tool to create `{output_path}/solution.json`

Key format requirements:
- `tiling_fields`: **JSON array** `[{{"type": "uint32_t", "name": "xxx"}}]`
- `output_alloc_code`: **C++ code** `at::Tensor result = ...;`
- Tiling class: `{op_name.title().replace("_", "")}CustomTilingData`

After writing, I will compile and test."""


def build_optimization_prompt(
    op_name: str,
    output_path: Path,
    exp_path: Path,
    runtime_ms: Optional[float],
    speedup: Optional[float],
) -> str:
    """Build the optimization prompt for successful iterations."""
    _ = op_name, exp_path
    runtime_str = f"{runtime_ms:.4f}" if runtime_ms else "N/A"
    speedup_str = f"{speedup:.2f}" if speedup else "N/A"

    return f"""Passed! Runtime: {runtime_str} ms ({speedup_str}x)

**Optimize**: Try increasing tileNum or using double buffering.

Write updated solution to `{output_path}/solution.json`"""


def build_error_fix_prompt(
    output_path: Path,
    stage: str,
    error_msg: str,
    exp_path: Path,
) -> str:
    """Build the error fix prompt for failed iterations."""
    _ = exp_path

    # Detect specific error patterns
    hints = []
    if "TILING_DATA_FIELD_DEF" in error_msg:
        hints.append("- `tiling_fields` must be JSON array, not string")
    if "OpCustomTilingData" in error_msg:
        hints.append("- Use correct tiling class name: `{OpName}CustomTilingData`")
    if "GetDimNum" in error_msg or "GetDim" in error_msg:
        hints.append("- Use `shape.GetShapeSize()` to get total elements")
    if "507035" in error_msg or "vector core exception" in error_msg:
        hints.append("- All DataCopy/vector ops need count >= 8")
    if "Output value mismatch" in error_msg:
        hints.append("- Check data alignment, may need DataCopyPad")

    hints_str = "\n".join(hints) if hints else "- Re-read constraints.md for template structure"

    return f"""**FAILED** at {stage}:
```
{error_msg}
```

**Hints**:
{hints_str}

**Fix steps**:
1. Re-read `{output_path}/constraints.md` for correct format
2. Check tiling class name matches operator name
3. Use `cann_search_api()` to verify API signatures

Write fixed solution to `{output_path}/solution.json`"""
