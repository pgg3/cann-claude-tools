# CANN Solution Format Guide

This document shows **exactly** how your `solution.json` fields are assembled into the final code.

---

## Complete Code Templates

Your code fills specific slots in pre-defined templates. Understanding these templates is essential.

### 1. kernel_src.cpp (Kernel Code)

```cpp
// ═══════════════════════════════════════════════════════════════
// AUTO-GENERATED HEADER - DO NOT INCLUDE THESE IN YOUR CODE
// ═══════════════════════════════════════════════════════════════
#include "kernel_operator.h"
// {kernel_includes} - your additional includes go here (optional)

// ═══════════════════════════════════════════════════════════════
// YOUR CODE: kernel_impl
// ═══════════════════════════════════════════════════════════════
{kernel_impl}

// ═══════════════════════════════════════════════════════════════
// AUTO-GENERATED ENTRY FUNCTION
// ═══════════════════════════════════════════════════════════════
extern "C" __global__ __aicore__ void {op_name}_custom(
    GM_ADDR x,      // ← from signature.json inputs
    GM_ADDR output, // ← from signature.json outputs
    GM_ADDR workspace,
    GM_ADDR tiling
) {
    GET_TILING_DATA(tilingData, tiling);  // ← AUTO: tilingData available
// ═══════════════════════════════════════════════════════════════
// YOUR CODE: kernel_entry_body
// ═══════════════════════════════════════════════════════════════
{kernel_entry_body}
}
```

**What you write:**
- `kernel_impl`: Your kernel class definition (without `#include "kernel_operator.h"`)
- `kernel_entry_body`: Code to instantiate and call your kernel (tilingData is available)

---

### 2. host_tiling.h (Tiling Data Definition)

```cpp
// ═══════════════════════════════════════════════════════════════
// AUTO-GENERATED HEADER
// ═══════════════════════════════════════════════════════════════
#include "register/tilingdata_base.h"
// {tiling_includes}

namespace optiling {

BEGIN_TILING_DATA_DEF({OpName}CustomTilingData)
// ═══════════════════════════════════════════════════════════════
// AUTO-GENERATED FROM YOUR tiling_fields
// Each {"type": "T", "name": "N"} becomes:
//     TILING_DATA_FIELD_DEF(T, N);
// ═══════════════════════════════════════════════════════════════
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS({OpName}Custom, {OpName}CustomTilingData);

}  // namespace optiling
```

**What you write:**
- `tiling_fields`: JSON array defining your tiling variables

---

### 3. host_operator.cpp (Host-side Functions)

```cpp
// ═══════════════════════════════════════════════════════════════
// AUTO-GENERATED HEADER
// ═══════════════════════════════════════════════════════════════
#include "{op_name}_custom_tiling.h"
// {tiling_func_includes}

namespace optiling {

// ═══════════════════════════════════════════════════════════════
// TILING FUNCTION - Runs on HOST CPU
// ═══════════════════════════════════════════════════════════════
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
// ═══════════════════════════════════════════════════════════════
// YOUR CODE: tiling_func_body
// ═══════════════════════════════════════════════════════════════
{tiling_func_body}
}

// ═══════════════════════════════════════════════════════════════
// SHAPE INFERENCE - Runs on HOST CPU
// ═══════════════════════════════════════════════════════════════
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
// ═══════════════════════════════════════════════════════════════
// YOUR CODE: infer_shape_body
// ═══════════════════════════════════════════════════════════════
{infer_shape_body}
}

}  // namespace optiling
```

**What you write:**
- `tiling_func_body`: Calculate tiling parameters, save to buffer
- `infer_shape_body`: Set output shape based on input shape

---

### 4. Python Binding (output_alloc_code)

```cpp
// In torch_npu binding code:
at::Tensor {op_name}_custom(const at::Tensor& x) {
    // ═══════════════════════════════════════════════════════════════
    // YOUR CODE: output_alloc_code
    // ═══════════════════════════════════════════════════════════════
    {output_alloc_code}

    // ... NPU execution code (auto-generated) ...
    return result;
}
```

**What you write:**
- `output_alloc_code`: C++ code to allocate output tensor (e.g., `at::Tensor result = at::empty_like(x);`)

---

## solution.json Format Requirements

### Field Types

| Field | Type | Example |
|-------|------|---------|
| `kernel_impl` | string | `"using namespace AscendC;\n\nclass KernelRelu {...};"` |
| `kernel_entry_body` | string | `"    KernelRelu op;\n    op.Init(...);\n    op.Process();"` |
| `tiling_fields` | **JSON array** | `[{"type": "uint32_t", "name": "totalLength"}]` |
| `tiling_func_body` | string | `"    ReluCustomTilingData tiling;\n    ..."` |
| `infer_shape_body` | string | `"    *y_shape = *x_shape;\n    return ge::GRAPH_SUCCESS;"` |
| `output_alloc_code` | string (C++) | `"at::Tensor result = at::empty_like(x);"` |

### Common Mistakes

```json
// ❌ WRONG: tiling_fields as string
"tiling_fields": "TILING_DATA_FIELD_DEF(uint32_t, totalLength);"

// ✅ CORRECT: tiling_fields as JSON array
"tiling_fields": [{"type": "uint32_t", "name": "totalLength"}]
```

```json
// ❌ WRONG: output_alloc_code as Python
"output_alloc_code": "output = torch.empty_like(x)"

// ✅ CORRECT: output_alloc_code as C++
"output_alloc_code": "at::Tensor result = at::empty_like(x);"
```

---

## Naming Conventions

For operator `foo_bar`:

| Item | Name |
|------|------|
| Kernel class | `KernelFooBar` |
| Tiling data class | `FooBarCustomTilingData` |
| Entry function | `foo_bar_custom` (auto-generated) |
| Tiling setters | `tiling.set_totalLength(value)` |

**IMPORTANT:** Use the correct TilingData class name in `tiling_func_body`:
- For `relu`: use `ReluCustomTilingData`, NOT `OpCustomTilingData`
- For `avg_pool`: use `AvgPoolCustomTilingData`

---

## Available Context in Each Section

| Section | Available Variables | Notes |
|---------|---------------------|-------|
| `kernel_impl` | None | Define your class here |
| `kernel_entry_body` | `tilingData`, GM_ADDR params | `tilingData.xxx` to access tiling fields |
| `tiling_func_body` | `context` (TilingContext*) | Runs on HOST CPU |
| `infer_shape_body` | `context` (InferShapeContext*) | Runs on HOST CPU |

---

## Minimal Working Example (ReLU)

```json
{
  "kernel_impl": "using namespace AscendC;\n\nclass KernelRelu {\npublic:\n    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength) {\n        xGm.SetGlobalBuffer((__gm__ float*)x, totalLength);\n        yGm.SetGlobalBuffer((__gm__ float*)y, totalLength);\n        this->totalLength = totalLength;\n        pipe.InitBuffer(inQue, 1, totalLength * sizeof(float));\n        pipe.InitBuffer(outQue, 1, totalLength * sizeof(float));\n    }\n    __aicore__ inline void Process() {\n        LocalTensor<float> xLocal = inQue.AllocTensor<float>();\n        DataCopy(xLocal, xGm, totalLength);\n        inQue.EnQue(xLocal);\n        xLocal = inQue.DeQue<float>();\n        LocalTensor<float> yLocal = outQue.AllocTensor<float>();\n        Relu(yLocal, xLocal, totalLength);\n        outQue.EnQue(yLocal);\n        inQue.FreeTensor(xLocal);\n        yLocal = outQue.DeQue<float>();\n        DataCopy(yGm, yLocal, totalLength);\n        outQue.FreeTensor(yLocal);\n    }\nprivate:\n    TPipe pipe;\n    TQue<QuePosition::VECIN, 1> inQue;\n    TQue<QuePosition::VECOUT, 1> outQue;\n    GlobalTensor<float> xGm, yGm;\n    uint32_t totalLength;\n};",

  "kernel_entry_body": "    KernelRelu op;\n    op.Init(x, output, tilingData.totalLength);\n    op.Process();",

  "tiling_fields": [
    {"type": "uint32_t", "name": "totalLength"}
  ],

  "tiling_func_body": "    ReluCustomTilingData tiling;\n    auto shape = context->GetInputShape(0)->GetStorageShape();\n    uint32_t totalLength = shape.GetShapeSize();\n    tiling.set_totalLength(totalLength);\n    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());\n    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());\n    context->SetBlockDim(1);\n    size_t* ws = context->GetWorkspaceSizes(1);\n    ws[0] = 0;\n    return ge::GRAPH_SUCCESS;",

  "infer_shape_body": "    *context->GetOutputShape(0) = *context->GetInputShape(0);\n    return ge::GRAPH_SUCCESS;",

  "output_alloc_code": "at::Tensor result = at::empty_like(x);"
}
```

---

## API Reference

Use MCP tools to look up API signatures:
- `cann_get_knowledge()` - List all available APIs
- `cann_search_api("DataCopy")` - Get specific API details
- `cann_search_operator("relu")` - Find similar implementations
