---
name: cann-operator
description: Generate CANN Ascend C operator code from Python reference
---

# CANN Operator Generation

You are an expert Ascend C developer. Generate optimized NPU kernels by writing `solution.json`.

## Output Format

Write a JSON file with these fields:

```json
{
  "kernel_impl": "C++ kernel class code",
  "kernel_entry_body": "Instantiate and call kernel",
  "tiling_fields": [{"type": "uint32_t", "name": "xxx"}, ...],
  "tiling_func_body": "Host-side tiling calculation",
  "infer_shape_body": "Output shape inference",
  "output_alloc_code": "at::Tensor result = ...;"
}
```

## Naming Convention

For operator `foo_bar`:
- Kernel class: `KernelFooBar`
- TilingData: `FooBarCustomTilingData`
- Entry function: `foo_bar_custom(...)` (auto-generated)
- Tiling setters: `tiling.set_xxx(value)`

## Code Generation Rules

Your code is wrapped automatically:

```cpp
#include "kernel_operator.h"          // ← Auto-added
{kernel_impl}                         // ← Your class

extern "C" __global__ __aicore__ void xxx_custom(...) {
    GET_TILING_DATA(tilingData, tiling);  // ← Auto-added
{kernel_entry_body}                        // ← Your code
}
```

**DO NOT write**: `#include`, `extern "C"`, `GET_TILING_DATA`

## MCP Tools

| Task | Command |
|------|---------|
| Verify API signature | `cann_search_api("DataCopy")` → `Read(header_file)` |
| Find operator example | `cann_search_operator("relu")` |

## Error Quick Reference

| Error | Fix |
|-------|-----|
| `no matching function` | Use MCP to verify API signature |
| `error 507035 vector core exception` | All operations need count ≥ 8 |
| `cast between floating and unsigned` | Move type cast to tiling_func_body |
| `UB address out of bounds` | Increase tileNum |
