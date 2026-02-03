---
name: cann-operator
description: Generate CANN Ascend C operator code from Python reference
---

# CANN Operator Generation

You are an expert Ascend C developer. Generate optimized NPU kernels.

## Supported Operator Types

| Type | Examples | Compute Unit | Tiling Strategy |
|------|----------|--------------|-----------------|
| **Vector** | ReLU, Abs, Exp, Add, Mul | Vector Unit | UB-based (totalLength, tileNum) |
| **Cube** | MatMul, Conv2D, GEMM | Cube Unit | M/K/N dimension tiling |

The system auto-detects operator type and generates appropriate templates.

## Workflow

```
[1. Read Template] → [2. Implement] → [3. Write solution.json] → (CLI evaluates) → [4. Fix/Optimize]
```

**IMPORTANT**:
- You only need to write `solution.json` to `{CANN_OUTPUT_DIR}`
- The CLI will automatically compile, test, and give you feedback
- Do NOT write test files or call evaluation tools yourself

---

## Naming Convention (CRITICAL)

For operator `foo_bar`, the system generates:
- Kernel entry: `foo_bar_custom(...)`
- TilingData class: `FooBarCustomTilingData`
- Tiling setter methods: `tiling.set_xxx(value)`

**Example**:
```cpp
// In tiling_func_body:
FooBarCustomTilingData tiling;
tiling.set_totalLength(totalLength);  // Vector
// or
tiling.set_M(M); tiling.set_K(K); tiling.set_N(N);  // Cube

// In kernel_entry_body:
KernelFooBar op;
op.Init(..., tilingData.totalLength, ...);  // Use "tilingData"
```

---

## Component Requirements

| Component | Must Include | ⚠️ DO NOT Include |
|-----------|--------------|-------------------|
| **kernel_impl** | `using namespace AscendC;`, class with Init/Process | ❌ `#include`, ❌ `extern "C"` entry |
| **kernel_entry_body** | Instantiate kernel, call Init with `tilingData.xxx` | ❌ `GET_TILING_DATA` (auto-added) |
| **tiling_fields** | Array of `{"type": "...", "name": "..."}` | |
| **tiling_func_body** | Get shapes, calculate tiling, SaveToBuffer | |
| **infer_shape_body** | Set output shape based on inputs | |
| **output_alloc_code** | `at::Tensor result = ...;` | |

**Auto-generated wrapper**:
```cpp
#include "kernel_operator.h"          // ← Auto-added
{kernel_impl}                         // ← Your class only

extern "C" __global__ __aicore__ void xxx_custom(...) {  // ← Auto-generated
    GET_TILING_DATA(tilingData, tiling);                 // ← Auto-added
{kernel_entry_body}                                      // ← Your code
}
```

---

## Quick Error Reference

| Error Pattern | Fix |
|---------------|-----|
| `'xxx' was not declared` | Check API name against template |
| `UB address out of bounds` | tileNum too small (Vector) |
| `L0A/L0B/L0C overflow` | Tile size too large (Cube) |
| `no matching function for call` | Read header for correct signature |

**For detailed constraints**: Read `{CANN_OUTPUT_DIR}/constraints.md`
**For hardware specs**: Read `{CANN_OUTPUT_DIR}/hardware.md`

---

## MCP Tools (For API Research)

| When | Use Tool |
|------|----------|
| Unknown API signature | `cann_search_api("DataCopy")` then `Read` the header |
| Need Compute pattern | `cann_search_operator("relu")` |
| List available APIs | `cann_get_knowledge()` |

**WARNING**: MCP returns FILE PATHS. Use `Read` tool to see actual code!

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `CANN_OUTPUT_DIR` | Directory for solution.json |
| `CANN_OP_NAME` | Operator name |
| `CANN_PYTHON_REF` | Path to Python reference file |
| `CANN_NPU_TYPE` | Target NPU type (e.g., {CANN_NPU_TYPE}) |
