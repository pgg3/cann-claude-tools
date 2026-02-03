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
