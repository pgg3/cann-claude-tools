---
name: cann-operator
description: Generate CANN Ascend C operator code from Python reference
---

# CANN Operator Generation

Generate optimized NPU kernels by writing `solution.json`.

## Workflow

### Step 1: Research APIs
```
cann_get_knowledge()              # List all API categories
cann_search_api("Relu")           # Get API signature
cann_search_operator("avg_pool")  # Find similar implementations
```

### Step 2: Read Input Files
1. `signature.json` - Operator interface (inputs, outputs, params)
2. `python_reference.py` - Reference implementation
3. `constraints.md` - **Code template structure** & JSON format requirements
4. `vector.md` or `cube.md` - Hardware specs & critical rules
5. `solution_template.json` - Example JSON format

### Step 3: Write solution.json

**CRITICAL: Use the Write tool to create `solution.json`**

Read `constraints.md` to understand:
- The complete code template (what's auto-generated vs what you write)
- JSON format requirements (especially `tiling_fields` as array)
- Naming conventions (`{OpName}CustomTilingData`)

All 6 fields required:
- `kernel_impl` - Kernel class definition
- `kernel_entry_body` - Instantiate and call kernel
- `tiling_fields` - **JSON array** `[{"type": "T", "name": "N"}, ...]`
- `tiling_func_body` - Host-side tiling calculation
- `infer_shape_body` - Output shape inference
- `output_alloc_code` - **C++ code** `at::Tensor result = ...;`

## Quick Reference

### Operator Naming
For operator `foo_bar`:
- Kernel class: `KernelFooBar`
- Tiling class: `FooBarCustomTilingData`

### Available in kernel_entry_body
- `tilingData` - Access via `tilingData.totalLength`
- GM_ADDR params - `x`, `output`, etc. from signature.json

### Error Troubleshooting
| Error | Likely Cause |
|-------|--------------|
| `TILING_DATA_FIELD_DEF requires 2 arguments` | tiling_fields is string, should be JSON array |
| `OpCustomTilingData not declared` | Wrong tiling class name, use `{OpName}CustomTilingData` |
| `no member named GetDimNum` | Wrong API, use `shape.GetShapeSize()` |
