---
name: cann-operator
description: Generate CANN Ascend C operator code from Python reference
---

# CANN Operator Generation

You are an expert Ascend C developer generating optimized NPU kernels.

## Quick Reference

### Naming Convention

For operator `foo_bar`:
- Kernel entry: `foo_bar_custom(...)`
- TilingData class: `FooBarCustomTilingData`
- Tiling setters: `tiling.set_xxx(value)`

### Code Structure (Auto-Generated)

```cpp
#include "kernel_operator.h"          // ← Auto-added
{kernel_impl}                         // ← Your class only

extern "C" __global__ __aicore__ void xxx_custom(...) {  // ← Auto-generated
    GET_TILING_DATA(tilingData, tiling);                 // ← Auto-added
{kernel_entry_body}                                      // ← Your code
}
```

**DO NOT include**: `#include`, `extern "C"`, `GET_TILING_DATA` in your code.

### Component Checklist

| Component | What to Write |
|-----------|---------------|
| `kernel_impl` | Class with Init/Process, use `tilingData.xxx` |
| `kernel_entry_body` | Instantiate and call kernel |
| `tiling_fields` | `[{"type": "uint32_t", "name": "xxx"}, ...]` |
| `tiling_func_body` | Get shapes, calculate tiling, `SaveToBuffer` |
| `infer_shape_body` | Set output shape |
| `output_alloc_code` | `at::Tensor result = ...;` |

### MCP Tools

| Task | Command |
|------|---------|
| Find API signature | `cann_search_api("DataCopy")` → `Read(header_file)` |
| Find operator example | `cann_search_operator("relu")` |
| List available APIs | `cann_get_knowledge()` |

**Note**: MCP returns FILE PATHS. Use `Read` tool to see actual code.

### Quick Error Reference

| Error | Cause | Fix |
|-------|-------|-----|
| `no matching function` | Wrong API signature | Use MCP to verify |
| `UB address out of bounds` | tileNum too small | Increase tileNum |
| `cast between floating and unsigned` | Type cast in kernel | Move to tiling_func_body |
| `error 507035 vector core exception` | Count < 8 | All ops need count ≥ 8 |

### Files to Read

- `signature.json` - Parsed inputs/outputs/init_params
- `solution_template.json` - Code structure example
- `constraints.md` - Detailed constraints
- `hardware.md` - Memory limits and alignment
