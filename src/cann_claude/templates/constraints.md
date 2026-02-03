# CANN Constraints Reference

This file contains detailed constraints for CANN Ascend C development.
Read this when encountering compilation errors or unexpected behavior.

---

## Code Generation Structure

Your `solution.json` fields are assembled into complete files. Understanding this helps avoid errors.

### kernel_src.cpp (Final Structure)

```cpp
#include "kernel_operator.h"
// {kernel_includes} - if you specify extra includes

{kernel_impl}  // <-- YOUR CODE: kernel class definition

extern "C" __global__ __aicore__ void op_custom({gm_params}, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);  // <-- AUTO-ADDED: makes tilingData available
{kernel_entry_body}  // <-- YOUR CODE: call your kernel
}
```

**What this means:**
- `#include "kernel_operator.h"` is auto-added, DO NOT add it in kernel_impl
- `extern "C" __global__` entry is auto-generated, DO NOT write your own
- `GET_TILING_DATA(tilingData, tiling)` is auto-added, use `tilingData.xxx` directly
- `{gm_params}` are generated from signature: `GM_ADDR x, GM_ADDR y, GM_ADDR output`

### host_tiling.h (Final Structure)

```cpp
#include "register/tilingdata_base.h"
// {tiling_includes}

namespace optiling {
BEGIN_TILING_DATA_DEF(OpCustomTilingData)
    // {tiling_fields} -> TILING_DATA_FIELD_DEF(uint32_t, totalLength);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(OpCustom, OpCustomTilingData);
}
```

**What this means:**
- `tiling_fields` becomes `TILING_DATA_FIELD_DEF(type, name);` for each field
- Access in kernel via `tilingData.totalLength`, `tilingData.tileNum`, etc.

### host_operator.cpp (Final Structure)

```cpp
#include "op_custom_tiling.h"
// {tiling_func_includes}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
{tiling_func_body}  // <-- YOUR CODE
}
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
{infer_shape_body}  // <-- YOUR CODE
}
}
```

**What this means:**
- Your tiling code runs on HOST CPU, not on NPU
- Use `context->GetInputShape(0)`, NOT `context->GetInputDesc(0)`
- Create your TilingData struct, call `.set_xxx()` methods, then `SaveToBuffer()`

---

## Forbidden APIs (DO NOT USE!)

These APIs look correct but DO NOT EXIST in template-based generation:

```cpp
// ❌ Context APIs that don't exist:
context->GetInputDesc(0)      // Use GetInputShape() instead
context->GetOutputDesc(0)     // Use GetOutputShape() instead
context->GetInputTensor(0)    // Don't use
context->SetTilingData()      // Use SaveToBuffer() instead
context->SetOutputShape()     // Use *y_shape = *x_shape
context->GetAttr("name", val) // Use GetAttrs() instead (plural!)
inputDesc->SetShape()         // Doesn't exist
inputDesc->GetShape()         // Doesn't exist
inputDesc->GetOriginShape()   // Doesn't exist

// ❌ Platform APIs (require unavailable includes):
platform_ascendc::PlatformAscendC  // Doesn't compile
context->GetPlatformInfo()         // Use hardcoded BLOCK_DIM = 8

// ❌ Advanced APIs (too complex for template-based generation):
DataCopyPad, DataCopyExtParams, DataCopyPadExtParams
SetFlag<HardEvent::xxx>, WaitFlag<HardEvent::xxx>
EVENT_ID0, EVENT_ID1
```

---

## Getting Operator Attributes

**DO NOT use `GetAttr()`** - it doesn't exist!

For operators with attributes (kernel_size, stride, padding, etc.), use **hardcoded defaults** or pass via tiling data:

```cpp
// ❌ WRONG - GetAttr doesn't exist
context->GetAttr("kernel_size", val);

// ✅ CORRECT - Use hardcoded values for now
uint32_t kernel_size = 3;  // Or get from template/config
uint32_t stride = 1;
uint32_t padding = 0;
```

If you need configurable attributes, define them in `tiling_fields` and compute in the tiling function based on input shapes.

---

## Forbidden Patterns

```cpp
// ❌ Standard C++ math - DOES NOT EXIST in kernel!
exp(x), sin(x), sqrt(x), pow(x,y)
// ✅ Use: Exp(dst, src, count), Sqrt(dst, src, count)

// ❌ Element access - NOT SUPPORTED!
float val = xLocal[i];
zLocal[i] = val * 2;
// ✅ Use vector operations on entire tensors

// ❌ Invalid QuePosition - DOES NOT EXIST!
QuePosition::TEMP, QuePosition::VECTMP, QuePosition::VECBUF
// ✅ Use: VECIN, VECOUT, VECCALC

// ❌ Non-existent APIs
Neg(), Subs(), Divs(), Pow()
// ✅ Alternatives:
//    Negation: Muls(dst, src, -1.0f, len)
//    Subtract scalar: Adds(dst, src, -scalar, len)
//    Divide by scalar: Muls(dst, src, 1.0f/scalar, len)

// ❌ Compare with scalar using Compare()
Compare(mask, xLocal, 0.0f, CMPMODE::GT, len);
// ✅ Use CompareScalar:
CompareScalar(mask, xLocal, 0.0f, CMPMODE::GT, len);

// ❌ Wrong mask type
SelectMask, LocalTensor<bool>
// ✅ Use: LocalTensor<uint8_t>
```

---

## Type Casting in AICore Functions

**CRITICAL**: AICore functions DO NOT allow direct casts between float and integer types!

```cpp
// ❌ WRONG - Will cause "cast between floating and unsigned integer variable is not allowed"
float invPoolSize = 1.0f / static_cast<float>(poolSize);  // Error!
float ratio = static_cast<float>(count) / total;          // Error!

// ✅ CORRECT - Compute float values from float literals
float invPoolSize = 1.0f / 121.0f;  // Use float literal directly
// Or pre-compute on host side in tiling_func_body and pass via tiling data

// ✅ CORRECT - Pass pre-computed float from host
// In tiling_func_body (runs on CPU, casts allowed):
float invPoolSize = 1.0f / static_cast<float>(kernel_size * kernel_size);
tiling.set_invPoolSize(invPoolSize);

// In kernel (use the pre-computed value):
float invPoolSize = tilingData.invPoolSize;
Muls(yLocal, sumLocal, invPoolSize, 1);
```

**Rule**: If you need to convert uint32_t to float, do it in tiling_func_body (host) and pass via tiling data.

---

## Valid QuePosition Values

| Value | Purpose |
|-------|---------|
| `QuePosition::VECIN` | Input buffers |
| `QuePosition::VECOUT` | Output buffers |
| `QuePosition::VECCALC` | Intermediate/temp buffers |

---

## Minimum Operation Sizes (CRITICAL!)

**All CANN vector operations require minimum 32-byte alignment (8 float32 elements).**

This causes **runtime crashes** (error 507035 "vector core exception") if violated!

```cpp
// ❌ WRONG - Will crash at runtime!
DataCopy(dst, src, 1);           // Copies 1 element - CRASH!
DataCopy(dst, src, 11);          // Copies 11 elements - CRASH!
Muls(dst, src, scalar, 1);       // Operates on 1 element - CRASH!
Add(dst, src1, src2, 4);         // Operates on 4 elements - CRASH!
ReduceSum(dst, src, work, 5);    // Reduces 5 elements - may crash

// ✅ CORRECT - Always use multiples of 8 (32 bytes)
DataCopy(dst, src, 8);           // Minimum valid count
DataCopy(dst, src, 128);         // Must be multiple of 8
Muls(dst, src, scalar, 8);       // Minimum 8 elements
Add(dst, src1, src2, 64);        // Any multiple of 8
```

### For Pooling/Sliding Window Operations

**DO NOT** try to process one output element at a time! Instead:

1. **Batch multiple windows**: Collect data for multiple output elements, then process in batches of 8+
2. **Pad to alignment**: If processing fewer than 8 elements, pad the buffer to 8
3. **Use contiguous memory**: Copy entire rows/regions, not individual pixels

```cpp
// ❌ WRONG - Element-by-element pooling
for (int i = 0; i < outputCount; i++) {
    // Copy kernel_size x kernel_size window for ONE output
    DataCopy(window, input[offset], kernelSize * kernelSize);  // May be < 8!
    ReduceSum(sum, window, work, kernelSize * kernelSize);     // May crash!
    Muls(output[i], sum, invPoolSize, 1);                      // CRASH!
}

// ✅ CORRECT - Batch processing with alignment
// Process 8 output elements at a time, ensuring all operations use count >= 8
uint32_t batchSize = 8;  // Minimum
for (int batch = 0; batch < outputCount; batch += batchSize) {
    // Collect windows for entire batch
    // Process batch with aligned operations
    // Write results in batch
}
```

### Implications for Complex Operators

For operators like pooling, convolution, or any sliding-window operation:

1. **Cannot use simple element-wise template** - these need specialized tiling
2. **Consider using CANN's built-in operators** via aclnn API when available
3. **If custom implementation needed**, ensure all memory operations are 32-byte aligned

---

## Research Warning

**DO NOT** copy complex code patterns from MCP research results!

Production operators use advanced features that will NOT compile:
- `platform_ascendc::PlatformAscendC`
- Hardware events (`EVENT_ID0`, `SetFlag`, `WaitFlag`)
- Ping-pong buffering with manual event sync

**USE research ONLY to**:
1. Verify API function signatures
2. Understand what a specific API does
3. See the Compute() function pattern for specific math operations

**ALWAYS use the solution_template.json patterns, not research code!**
