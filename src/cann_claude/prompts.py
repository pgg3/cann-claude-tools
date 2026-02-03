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
    return f"""Generate Ascend C operator `{op_name}` for {npu_type}.

**Read these files first**:
1. `{output_path}/signature.json` - inputs, outputs, init_params
2. `{ref_path}` - Python reference
3. `{output_path}/solution_template.json` - code structure example
4. `{output_path}/constraints.md` - CRITICAL constraints

**Write solution to**: `{output_path}/solution.json`

After writing, I will compile and test it."""


def build_optimization_prompt(
    op_name: str,
    output_path: Path,
    exp_path: Path,
    runtime_ms: Optional[float],
    speedup: Optional[float],
) -> str:
    """Build the optimization prompt for successful iterations."""
    _ = op_name, exp_path  # Reserved for future use
    runtime_str = f"{runtime_ms:.4f}" if runtime_ms else "N/A"
    speedup_str = f"{speedup:.2f}" if speedup else "N/A"

    return f"""Passed! Runtime: {runtime_str} ms ({speedup_str}x)

**Optimize**: Read `{output_path}/hardware.md`, try increasing tileNum or BUFFER_NUM.

Write updated solution to `{output_path}/solution.json`"""


def build_error_fix_prompt(
    output_path: Path,
    stage: str,
    error_msg: str,
    exp_path: Path,
) -> str:
    """Build the error fix prompt for failed iterations."""
    return f"""**FAILED** at {stage}:
```
{error_msg}
```

**Fix steps**:
1. Re-read `{output_path}/constraints.md`
2. If "no matching function": use `cann_search_api()` to verify
3. Check `{exp_path}/errors/` for similar past errors

Write fixed solution to `{output_path}/solution.json`"""
