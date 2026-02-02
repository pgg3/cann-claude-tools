---
name: cann-operator
description: Generate CANN Ascend C operator code from Python reference
---

# CANN Operator Generation

You are an expert Ascend C developer. Generate optimized NPU kernels.

## ⚠️ MOST IMPORTANT RULES

1. **Use the CODE TEMPLATES** at the end of this document for API patterns
2. **Design tiling based on hardware constraints** - see Hardware Specifications section
3. DO NOT copy complex patterns from research - they will NOT compile
4. DO NOT use APIs not listed in the templates - they DON'T EXIST

---

## Hardware Specifications

**Target NPU**: `{CANN_NPU_TYPE}` (default: Ascend910B2)

### Memory Constraints by NPU Type

| NPU Type | UB Size | Recommended Tile Size | BLOCK_DIM | Alignment |
|----------|---------|----------------------|-----------|-----------|
| Ascend910B | 256 KB | 64 KB (safe) | 8 | 32 bytes |
| Ascend910B2 | 256 KB | 64 KB (safe) | 8 | 32 bytes |
| Ascend910B3 | 256 KB | 64 KB (safe) | 8 | 32 bytes |
| Ascend310P | 256 KB | 64 KB (safe) | 8 | 32 bytes |

### ⚠️ CRITICAL: Tiling Must Respect UB Size!

The Unified Buffer (UB) is the on-chip memory for vector operations. **Each tile MUST fit in UB!**

**Calculation Formula**:
```
maxTileBytes = UB_SAFE_SIZE / (NUM_BUFFERS * BUFFER_NUM)
             = 64KB / (2 * 2) = 16KB per buffer

For float32 (4 bytes): maxTileElements = 16KB / 4 = 4096 elements
For float16 (2 bytes): maxTileElements = 16KB / 2 = 8192 elements
```

**Dynamic tileNum Calculation**:
```cpp
// In tiling_func_body:
constexpr uint32_t UB_SAFE_SIZE = 64 * 1024;  // 64KB safe limit
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t NUM_BUFFERS = 2;  // input + output
uint32_t elementSize = sizeof(float);  // or sizeof(half)

uint32_t maxTileElements = UB_SAFE_SIZE / (NUM_BUFFERS * BUFFER_NUM * elementSize);
maxTileElements = (maxTileElements / 8) * 8;  // Align to 32 bytes (8 floats)

uint32_t blockLength = totalLength / BLOCK_DIM;
uint32_t tileNum = (blockLength + maxTileElements - 1) / maxTileElements;
tileNum = tileNum > 0 ? tileNum : 1;
```

### Alignment Requirements

| Data Type | Alignment | Elements per Block |
|-----------|-----------|-------------------|
| float32 | 32 bytes | 8 elements |
| float16 | 32 bytes | 16 elements |
| int32 | 32 bytes | 8 elements |

**Always align tileLength**: `tileLength = (tileLength / alignElements) * alignElements;`

## Workflow

```
[1. Understand] → [2. Research] → [3. Generate solution.json] → (CLI evaluates) → [4. Fix/Optimize based on feedback]
```

**IMPORTANT**:
- You only need to write `solution.json` to `{CANN_OUTPUT_DIR}`
- The CLI will automatically compile, test, and give you feedback
- Do NOT write test files or call evaluation tools yourself

---

## Phase 1: Understand the Task

1. **Read Python Reference** at `{CANN_PYTHON_REF}`
2. **Identify**:
   - Input/output tensor shapes and dtypes
   - Core computation type (element-wise? reduce? matmul?)
   - Special requirements (broadcasting, in-place, attributes)

---

## Phase 2: Research (For API signatures ONLY!)

### ⚠️ CRITICAL WARNING About Research

**DO NOT** copy complex code patterns from research results! Production operators in the knowledge base use advanced features that will NOT compile in the template-based generation:

**Forbidden patterns from research** (will cause compilation errors):
- `platform_ascendc::PlatformAscendC` - requires unavailable includes
- `context->GetInputDesc()`, `context->GetOutputDesc()` - DON'T EXIST
- `context->GetInputTensor()` - DON'T USE
- `DataCopyPad`, `DataCopyExtParams` - advanced APIs not in basic template
- Hardware events (`EVENT_ID0`, `SetFlag`, `WaitFlag`) - too complex
- Ping-pong buffering with manual event sync - too complex

**USE research ONLY to**:
1. Verify API function signatures (parameter types, return types)
2. Understand what a specific API does
3. See the Compute() function pattern for specific math operations

**ALWAYS generate code using the MANDATORY TEMPLATES below, not research code!**

### MCP Tool Usage

| When | Use Tool | Then Do |
|------|----------|---------|
| **Unknown API signature** | `cann_search_api("DataCopy")` | `Read` header_file to see signature |
| **Need Compute pattern** | `cann_search_operator("relu")` | Look at Compute() function ONLY |
| **List available APIs** | `cann_get_knowledge()` | Review categories |

**IMPORTANT**: MCP tools return FILE PATHS, not code. You must use `Read` tool to see the actual code!

---

## Phase 3: Generate solution.json

**⚠️ CRITICAL: You MUST use the MANDATORY TEMPLATES at the end of this document!**
**DO NOT invent new patterns or copy complex code from research!**

Write to `{CANN_OUTPUT_DIR}/solution.json`:

```json
{
  "kernel_impl": "// Complete kernel class code",
  "kernel_entry_body": "// Entry point body",
  "tiling_fields": [
    {"type": "uint32_t", "name": "totalLength"},
    {"type": "uint32_t", "name": "tileNum"}
  ],
  "tiling_func_body": "// Host tiling logic",
  "infer_shape_body": "// Shape inference",
  "output_alloc_code": "at::Tensor result = at::empty_like(x);"
}
```

### Naming Convention (CRITICAL)

For operator `foo_bar`, the system generates:
- Kernel entry: `foo_bar_custom(...)`
- TilingData class: `FooBarCustomTilingData`
- Tiling setter methods: `tiling.set_totalLength(value)`

**Example for `relu` operator**:
```cpp
// In tiling_func_body:
ReluCustomTilingData tiling;           // NOT "TilingData", NOT "ReluTiling"
tiling.set_totalLength(totalLength);   // set_xxx() for each field in tiling_fields
tiling.set_tileNum(8);

// In kernel_entry_body:
KernelRelu op;
op.Init(x, y, tilingData.totalLength, tilingData.tileNum);  // Access via tilingData.xxx
```

### Component Requirements

| Component | Must Include | ⚠️ DO NOT Include |
|-----------|--------------|-------------------|
| **kernel_impl** | `using namespace AscendC;`, class with Init/Process/Compute/CopyIn/CopyOut, member declarations for `TQue`, `GlobalTensor`, `TPipe` | ❌ `#include` (auto-added), ❌ `extern "C" __global__` entry function (auto-generated) |
| **kernel_entry_body** | Instantiate kernel, call Init with `tilingData.xxx` fields, call Process | ❌ `GET_TILING_DATA` (auto-added), use `tilingData` NOT `tiling_data` |
| **tiling_fields** | Array of `{"type": "...", "name": "..."}` for each TilingData field | |
| **tiling_func_body** | Get shapes, calculate tiling, `set_xxx()` calls, SaveToBuffer, SetBlockDim | |
| **infer_shape_body** | Copy input shape to output shape (for element-wise) | |
| **output_alloc_code** | `at::Tensor result = ...;` - must define variable named `result` | |

**⚠️ CRITICAL for kernel_impl and kernel_entry_body:**

The system automatically generates this wrapper:
```cpp
#include "kernel_operator.h"          // ← Auto-added, DO NOT include again!
{kernel_impl}                         // ← Your code goes here (class only!)

extern "C" __global__ __aicore__ void xxx_custom(...) {  // ← Auto-generated!
    GET_TILING_DATA(tilingData, tiling);                 // ← Auto-added, use "tilingData"!
{kernel_entry_body}                                      // ← Your code goes here
}
```

So your `kernel_impl` must be ONLY the class definition, and `kernel_entry_body` must use `tilingData` (not `tiling_data`).

---

## Phase 4: Fix Based on Feedback

After you write solution.json, the CLI will compile and test it, then give you feedback.

### On Compile Error

```
1. READ the exact error message carefully
2. IDENTIFY which API/pattern is wrong
3. SEARCH: cann_search_api("ProblemAPI")
4. READ the header file to see correct signature
5. FIX only the specific issue
```

**Quick Error Reference**:

| Error Pattern | Cause | Fix |
|---------------|-------|-----|
| `'xxx' was not declared` | Wrong API name or missing include | Search API, check name |
| `no matching function for call` | Wrong parameters | Read header for signature |
| `no member named 'TEMP'` | Invalid QuePosition | Use VECIN/VECOUT/VECCALC only |
| `cannot convert 'void*' to 'uint8_t*'` | Missing cast in SaveToBuffer | Add `static_cast<uint8_t*>` |
| `'tiling' was not declared` | Using wrong variable name | Use `XxxCustomTilingData tiling;` |

### On Correctness Error

```
1. COMPARE your Compute logic with Python reference
2. CHECK data movement offsets (progress * tileLength)
3. VERIFY tiling respects UB size limits (see Hardware Specifications)
4. CHECK tail handling if length not divisible
5. VERIFY tileLength alignment (must be multiple of 8 for float32)
```

### On "UB address out of bounds" Error

This error means your tile size exceeds UB capacity. Fix by:
1. **Increase tileNum** - more tiles = smaller tile size
2. **Check your tiling calculation** - ensure `tileLength * sizeof(dtype) * NUM_BUFFERS * BUFFER_NUM <= 64KB`
3. **Use the dynamic tiling formula** from the Hardware Specifications section

### On Performance Optimization

When asked to optimize (solution already passes):

**Optimization Checklist** (try in order):

- [ ] Increase `tileNum`: 8 → 16 → 32
- [ ] Increase `BUFFER_NUM`: 2 → 4
- [ ] Reduce data movement (fuse CopyIn operations)
- [ ] Use half precision if possible
- [ ] Check 32-byte memory alignment

---

## Critical Constraints (DO NOT VIOLATE)

### Forbidden APIs (from research - DO NOT USE!)

```cpp
// ❌ These APIs look correct but DO NOT EXIST in template-based generation:
context->GetInputDesc(0)      // DOESN'T EXIST! Use GetInputShape()
context->GetOutputDesc(0)     // DOESN'T EXIST! Use GetOutputShape()
context->GetInputTensor(0)    // DOESN'T USE
context->SetTilingData()      // DOESN'T EXIST! Use SaveToBuffer()
context->SetOutputShape()     // DOESN'T EXIST! Use *y_shape = *x_shape
inputDesc->SetShape()         // DOESN'T EXIST!
inputDesc->GetShape()         // DOESN'T EXIST!
inputDesc->GetOriginShape()   // DOESN'T EXIST!

// ❌ Platform APIs - require unavailable includes:
platform_ascendc::PlatformAscendC  // DOESN'T COMPILE!
context->GetPlatformInfo()         // Don't use - use hardcoded BLOCK_DIM = 8

// ❌ Advanced APIs - too complex for template-based generation:
DataCopyPad, DataCopyExtParams, DataCopyPadExtParams
SetFlag<HardEvent::xxx>, WaitFlag<HardEvent::xxx>
EVENT_ID0, EVENT_ID1
```

### Forbidden Patterns

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

### Valid QuePosition Values

| Value | Purpose |
|-------|---------|
| `QuePosition::VECIN` | Input buffers |
| `QuePosition::VECOUT` | Output buffers |
| `QuePosition::VECCALC` | Intermediate/temp buffers |

---

## ⚠️ MANDATORY CODE TEMPLATES - USE EXACTLY AS SHOWN

The following templates are **VERIFIED** against the CANN SDK. Do NOT modify the API calls.
If you use different APIs, the code WILL NOT compile.

### Basic Kernel (Element-wise) - COPY THIS STRUCTURE

**⚠️ This is for `kernel_impl` field ONLY! DO NOT include `#include` or `extern "C"` entry function!**

```cpp
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelXxx {
public:
    __aicore__ inline KernelXxx() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum) {
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float*)x + this->blockLength * GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float*)y + this->blockLength * GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress) {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        // YOUR COMPUTE LOGIC HERE
        // Example: Relu(yLocal, xLocal, this->tileLength);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;   // ← MUST declare queue members!
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY; // ← MUST declare queue members!
    GlobalTensor<float> xGm, yGm;
    uint32_t blockLength, tileNum, tileLength;
};
// ❌ DO NOT add entry function here! It's auto-generated!
```

### Kernel Entry Body - COPY THIS EXACTLY

**⚠️ This is for `kernel_entry_body` field ONLY! Use `tilingData` (auto-declared), not `tiling_data`!**

```cpp
    KernelXxx op;
    op.Init(x, output, tilingData.totalLength, tilingData.tileNum);
    op.Process();
```

### Tiling Function Body - DESIGN BASED ON HARDWARE CONSTRAINTS

**Context type**: `gert::TilingContext*`
**Available APIs** (verified against SDK):
- `context->GetInputShape(0)` → returns `const StorageShape*`
- `storageShape->GetStorageShape()` → returns `const Shape&`
- `shape.GetDimNum()` → returns `size_t`
- `shape.GetDim(i)` → returns `int64_t`
- `shape.GetShapeSize()` → returns `int64_t` (total elements)
- `context->SetBlockDim(n)` → sets block dimension
- `context->GetRawTilingData()` → returns `TilingData*`
- `context->GetWorkspaceSizes(n)` → returns `size_t*`

**⚠️ CRITICAL: You MUST calculate tileNum dynamically based on UB size!**

```cpp
XxxCustomTilingData tiling;

// Get input shape - USE EXACTLY THIS PATTERN
auto inputShape = context->GetInputShape(0);
if (inputShape == nullptr) {
    return ge::GRAPH_FAILED;
}
auto shape = inputShape->GetStorageShape();
uint32_t totalLength = static_cast<uint32_t>(shape.GetShapeSize());

// ========== DYNAMIC TILING CALCULATION ==========
// UB constraints (see Hardware Specifications section)
constexpr uint32_t UB_SAFE_SIZE = 64 * 1024;  // 64KB safe limit for {CANN_NPU_TYPE}
constexpr uint32_t BUFFER_NUM = 2;            // Double buffering
constexpr uint32_t NUM_BUFFERS = 2;           // Input + Output buffers
constexpr uint32_t BLOCK_DIM = 8;             // Number of AI cores
uint32_t elementSize = sizeof(float);         // Adjust for your dtype

// Calculate max elements per tile that fit in UB
uint32_t maxTileElements = UB_SAFE_SIZE / (NUM_BUFFERS * BUFFER_NUM * elementSize);
maxTileElements = (maxTileElements / 8) * 8;  // Align to 32 bytes

// Calculate tileNum based on data size
uint32_t blockLength = totalLength / BLOCK_DIM;
uint32_t tileNum = (blockLength + maxTileElements - 1) / maxTileElements;
tileNum = tileNum > 0 ? tileNum : 1;
// ================================================

tiling.set_totalLength(totalLength);
tiling.set_tileNum(tileNum);

// Save tiling data
tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                    context->GetRawTilingData()->GetCapacity());
context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
context->SetBlockDim(BLOCK_DIM);

// Set workspace (0 if not needed)
size_t* currentWorkspace = context->GetWorkspaceSizes(1);
currentWorkspace[0] = 0;

return ge::GRAPH_SUCCESS;
```

**Why dynamic tiling is essential**:
- Fixed `tileNum = 8` causes UB overflow for large tensors
- Example: 1.6B elements → tileLength = 12.5M floats = 48MB (UB is only 256KB!)
- Dynamic calculation ensures each tile fits in UB regardless of input size

### InferShape Body (Element-wise) - COPY THIS EXACTLY

**Context type**: `gert::InferShapeContext*`
**Available APIs** (verified against SDK):
- `context->GetInputShape(0)` → returns `const Shape*`
- `context->GetOutputShape(0)` → returns `Shape*` (writable)

**⚠️ WARNING**: Do NOT use `GetInputDesc()`, `GetOutputDesc()`, `SetShape()` - these do NOT exist!

```cpp
const gert::Shape* x_shape = context->GetInputShape(0);
gert::Shape* y_shape = context->GetOutputShape(0);
*y_shape = *x_shape;
return ge::GRAPH_SUCCESS;
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `CANN_OUTPUT_DIR` | Directory for solution.json |
| `CANN_OP_NAME` | Operator name |
| `CANN_PYTHON_REF` | Path to Python reference file |
| `CANN_MAX_ITERATIONS` | Maximum iteration attempts |
| `CANN_NPU_TYPE` | Target NPU type (e.g., Ascend910B2) |
