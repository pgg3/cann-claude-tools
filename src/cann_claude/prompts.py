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
    """Build the initial generation prompt for iteration 1.

    Args:
        op_name: Operator name
        npu_type: NPU type (e.g., Ascend910B2)
        output_path: Output directory path
        ref_path: Path to Python reference file

    Returns:
        Formatted prompt string
    """
    return f"""Generate an Ascend C operator '{op_name}' for {npu_type}.

## STEP 1: READ FILES

Read these files in order:
1. `{output_path}/solution_template.json` - VERIFIED code patterns
2. `{ref_path}` - Python reference implementation

## STEP 2: IMPLEMENT

Focus on the `Compute()` function in kernel_impl:
- Replace `// TODO:` with your computation logic
- Example for ReLU: `Relu(yLocal, xLocal, this->tileLength);`

The template already includes dynamic tiling for {npu_type}.

## STEP 3: WRITE

Write your solution to: `{output_path}/solution.json`

## ⚠️ CHECKLIST

- [ ] kernel_impl: Compute() implemented
- [ ] kernel_entry_body: Uses `tilingData.xxx` and `output`
- [ ] No `#include` in kernel_impl (auto-added)
- [ ] No `extern "C"` entry function (auto-generated)

## Available Reference Files

There are two types of operators. Read the appropriate files based on your operator type:

| Type | Constraints | Hardware |
|------|-------------|----------|
| **Vector** (ReLU, Add, Exp...) | `{output_path}/constraints.md` | `{output_path}/hardware.md` |
| **Cube** (MatMul, Conv2D...) | `{output_path}/cube_constraints.md` | `{output_path}/cube_hardware.md` |

After writing, I will compile and test it."""


def build_optimization_prompt(
    op_name: str,
    output_path: Path,
    exp_path: Path,
    runtime_ms: Optional[float],
    speedup: Optional[float],
) -> str:
    """Build the optimization prompt for successful iterations.

    Args:
        op_name: Operator name
        output_path: Output directory path
        exp_path: Experience directory path
        runtime_ms: Current runtime in milliseconds
        speedup: Current speedup factor

    Returns:
        Formatted prompt string
    """
    runtime_str = f"{runtime_ms:.4f}" if runtime_ms else "N/A"
    speedup_str = f"{speedup:.2f}" if speedup else "N/A"

    return f"""The solution compiled and passed correctness tests!

Current Performance:
- Runtime: {runtime_str} ms
- Speedup: {speedup_str}x

## OPTIMIZATION TASK

For optimization strategies, read the hardware specs:
- Vector: `{output_path}/hardware.md`
- Cube: `{output_path}/cube_hardware.md`

Quick options (try in order):
1. Increase tileNum: 8 → 16 → 32
2. Increase BUFFER_NUM: 2 → 4

Write updated solution.json to {output_path}

## Document Optimization

If performance improved, write to: `{exp_path}/tips/opt_{op_name}.md`
- Include before/after runtime numbers
- Only document successful optimizations"""


def build_error_fix_prompt(
    output_path: Path,
    stage: str,
    error_msg: str,
    exp_path: Path,
) -> str:
    """Build the error fix prompt for failed iterations.

    Args:
        output_path: Output directory path
        stage: Stage where error occurred
        error_msg: Error message
        exp_path: Experience directory path

    Returns:
        Formatted prompt string
    """
    return f"""## BUILD FAILED at stage: {stage}

Error:
```
{error_msg}
```

## FIX WORKFLOW

1. **Analyze the error**: Identify the root cause from the error message above
2. **Check historical errors**: Review `{exp_path}/errors/` for similar past errors and their solutions
3. **Re-read template**: `{output_path}/solution_template.json`
4. **Check constraints**:
   - Vector: `{output_path}/constraints.md`
   - Cube: `{output_path}/cube_constraints.md`
5. **Check hardware specs** (if memory/UB related):
   - Vector: `{output_path}/hardware.md`
   - Cube: `{output_path}/cube_hardware.md`

## Historical Errors

Check `{exp_path}/errors/` directory for past error records.
Each JSON file contains:
- `op`: operator name
- `stage`: where error occurred
- `error`: the error message

Look for similar errors to learn from past fixes.

Write fixed solution.json to {output_path}"""
