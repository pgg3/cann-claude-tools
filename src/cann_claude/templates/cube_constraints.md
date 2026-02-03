# CANN Cube Operator Constraints

This file contains constraints specific to Cube operators (matrix operations).

## ⚠️ Scope: Cube Operators Only

For Vector operator constraints, see `constraints.md`.

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
    // {tiling_fields} -> TILING_DATA_FIELD_DEF(uint32_t, M);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(OpCustom, OpCustomTilingData);
}
```

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

---

## Cube Unit APIs

### Valid APIs

```cpp
// Matrix multiplication
Mmad(dstLocal, srcA, srcB, ...);  // Matrix multiply-add

// Data movement to/from L0
LoadData(dstL0, srcL1, ...);      // L1 → L0A/L0B
DataCopy(dstGM, srcL0C, ...);     // L0C → GM

// Buffer management
SetLocalTensor(tensor, addr);     // Set tensor address
GetLocalTensor(tensor);           // Get tensor
```

### Forbidden APIs (DO NOT USE)

```cpp
// ❌ Vector APIs in Cube kernel
Relu(), Abs(), Exp(), Add()       // These are Vector Unit APIs!

// ❌ Direct GM access in Cube
DataCopy(dstL0A, srcGM, ...);     // Must go through L1!

// ❌ Wrong buffer types
TQue<QuePosition::VECIN, ...>     // This is for Vector Unit
TQue<QuePosition::VECOUT, ...>    // This is for Vector Unit
```

---

## Memory Access Patterns

### Correct Pattern

```cpp
// GM → L1 → L0A/L0B → Cube → L0C → GM
// Step 1: GM → L1
DataCopy(l1A, gmA[offset], size);
DataCopy(l1B, gmB[offset], size);

// Step 2: L1 → L0
LoadData(l0A, l1A, ...);
LoadData(l0B, l1B, ...);

// Step 3: Cube computation
Mmad(l0C, l0A, l0B, ...);

// Step 4: L0C → GM
DataCopy(gmC[offset], l0C, size);
```

### Forbidden Patterns

```cpp
// ❌ Skip L1 buffer
LoadData(l0A, gmA, ...);          // Cannot load directly from GM!

// ❌ Wrong data flow
Mmad(gmC, l0A, l0B, ...);         // Cannot write directly to GM!

// ❌ Mix Vector and Cube
Relu(l0C, l0C, size);             // Cannot use Vector API on L0C!
```

---

## Data Layout Requirements

### Matrix Layout

```cpp
// A[M, K] - Row major or Column major
// B[K, N] - Must match A's K dimension
// C[M, N] - Output matrix

// For Mmad:
// - A is typically row-major (M rows, K cols)
// - B is typically column-major (K rows, N cols) or transposed
// - Check API documentation for specific requirements
```

### Alignment

```cpp
// All matrix dimensions must be aligned to Cube block size
// float16: 16×16 blocks
// int8: 32×32 blocks

// Padding may be required for non-aligned dimensions
uint32_t M_aligned = ((M + 15) / 16) * 16;
uint32_t K_aligned = ((K + 15) / 16) * 16;
uint32_t N_aligned = ((N + 15) / 16) * 16;
```

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `L0A overflow` | Tile too large | Reduce M_tile or K_tile |
| `L0B overflow` | Tile too large | Reduce K_tile or N_tile |
| `Misaligned access` | Dimension not multiple of 16 | Pad to alignment |
| `Invalid buffer` | Using Vector buffer for Cube | Use L1/L0 buffers |
| `Data layout mismatch` | Wrong matrix format | Check row/col major |

---

## Tiling Fields for MatMul

```json
{
  "tiling_fields": [
    {"type": "uint32_t", "name": "M"},
    {"type": "uint32_t", "name": "K"},
    {"type": "uint32_t", "name": "N"},
    {"type": "uint32_t", "name": "M_tile"},
    {"type": "uint32_t", "name": "K_tile"},
    {"type": "uint32_t", "name": "N_tile"}
  ]
}
```
