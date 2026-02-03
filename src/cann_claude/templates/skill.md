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
cann_search_api("DataCopyPad")    # Get API signature → returns header_file path
cann_search_operator("avg_pool")  # Find similar implementations → returns file paths
```

**IMPORTANT**: These tools return **file paths**, not code. Use the Read tool to view the actual content.

Example workflow:
```
1. cann_search_operator("avg_pool") → {"primary": {"kernel_files": ["/path/to/kernel.h"]}}
2. Read("/path/to/kernel.h") → see actual implementation
```

### Step 2: Read Input Files
1. `constraints.md` - **Code template structure** & JSON format requirements
2. `vector.md` or `cube.md` - Hardware specs & critical rules
3. `signature.json` - Operator interface (inputs, outputs, params)
4. `python_reference.py` - Reference implementation
5. `solution_template.json` - Example JSON format

### Step 3: Pre-flight Checks

Before writing code, verify:

**Output dimensions alignment**:
- If output width/height is NOT a multiple of 8, you **MUST** use `DataCopyPad` instead of `DataCopy`
- Example: outW=46 → use DataCopyPad to write exactly 46 elements

**Unfamiliar APIs**:
- If using an API you're unsure about, call `cann_search_api("ApiName")` first
- Then Read the returned header_file to see the exact signature

### Step 4: Write solution.json

**CRITICAL: Use the Write tool to create `solution.json`**

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

### Buffer Initialization
```cpp
// TQue: 3 arguments (que, BUFFER_NUM, size)
TQue<QuePosition::VECIN, 2> inQue;
pipe.InitBuffer(inQue, 2, bufferSize);

// TBuf: 2 arguments only (buf, size) - NO BUFFER_NUM!
TBuf<QuePosition::VECCALC> tmpBuf;
pipe.InitBuffer(tmpBuf, scratchSize);
```

### Data Transfer
```cpp
// ✅ Always use DataCopy for GM↔UB (uses DMA)
DataCopy(localTensor, xGm[offset], count);

// ⚠️ Avoid GetPhyAddr() for data transfer - very slow
// float val = *((__gm__ float*)xGm.GetPhyAddr() + idx);  // ~100x slower!
```

### Error Troubleshooting
| Error | Likely Cause |
|-------|--------------|
| `TILING_DATA_FIELD_DEF requires 2 arguments` | tiling_fields is string, should be JSON array |
| `OpCustomTilingData not declared` | Wrong tiling class name, use `{OpName}CustomTilingData` |
| `InitBuffer not supports T as TBuf` | TBuf uses 2 args: `pipe.InitBuffer(buf, size)` |
| `Output value mismatch` | Non-aligned output → use `DataCopyPad` |
| `507035 vector core exception` | Vector operation count < 8 |
