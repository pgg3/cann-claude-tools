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

## STEP 1: READ KEY FILES

Read these files **in order**:

1. **`{output_path}/signature.json`** - Parsed operator signature (inputs, outputs, init_params)
2. **`{ref_path}`** - Python reference implementation
3. **`{output_path}/solution_template.json`** - Template with correct code structure
4. **`{output_path}/constraints.md`** - CRITICAL constraints (32-byte alignment, forbidden APIs)

## STEP 2: UNDERSTAND THE SIGNATURE

The `signature.json` tells you:
- `inputs`: Tensor inputs to the operator (GM_ADDR parameters)
- `outputs`: Tensor outputs (GM_ADDR parameters)
- `init_params`: Non-tensor parameters (kernel_size, stride, etc.) - passed via tiling data

**IMPORTANT**:
- Only tensor inputs/outputs become GM_ADDR parameters
- Non-tensor init_params must be hardcoded or passed via tiling_fields

## STEP 3: ANALYZE COMPUTATION PATTERN

Based on python_reference.py, determine:
- **Element-wise** (ReLU, Add): Use template as-is, input shape = output shape
- **Sliding window** (Pool, Conv): Different output shape, need custom tiling
- **Reduction** (Sum, Mean): Output smaller than input

If NOT element-wise: You MUST modify `tiling_fields`, `tiling_func_body`, and `infer_shape_body`.

## STEP 4: VERIFY API SIGNATURES

Before using any AscendC API, verify with MCP:
```
cann_search_api("ReduceSum")  # Get header path
Read(header_file)              # See actual signature
```

## STEP 5: WRITE SOLUTION

Write to: `{output_path}/solution.json`

Required fields (see template for format):
- `kernel_impl`: Kernel class (NO #include, NO extern "C")
- `kernel_entry_body`: Call kernel with `tilingData.xxx`
- `tiling_fields`: Parameters to pass to kernel
- `tiling_func_body`: Calculate tiling on host CPU
- `infer_shape_body`: Set output shape
- `output_alloc_code`: PyTorch tensor allocation

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

**Current Performance**: {runtime_str} ms ({speedup_str}x speedup)

## OPTIMIZATION OPTIONS

Read `{output_path}/hardware.md` for optimization strategies.

Quick options (try in order):
1. Increase `tileNum`: 8 → 16 → 32
2. Increase `BUFFER_NUM`: 2 → 4

Write updated solution.json to {output_path}

If performance improved, document to: `{exp_path}/tips/opt_{op_name}.md`"""


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

```
{error_msg}
```

## FIX STEPS

1. **Analyze error**: What does the message say?
2. **Check constraints**: Re-read `{output_path}/constraints.md`
3. **Verify APIs**: Use `cann_search_api("FunctionName")` then `Read` header
4. **Check history**: Look at `{exp_path}/errors/` for similar past errors

## Common Fixes

| Error | Fix |
|-------|-----|
| `no matching function` | Wrong API signature - use MCP to verify |
| `cast between floating and unsigned` | Don't cast int↔float in kernel - do it in tiling_func_body |
| `error 507035 vector core exception` | Operation count < 8 - all DataCopy/Muls need count ≥ 8 |
| `UB address out of bounds` | tileNum too small for data size |

Write fixed solution.json to {output_path}"""
