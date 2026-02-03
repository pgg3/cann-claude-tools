# CANN Constraints Reference

This file contains detailed constraints for CANN Ascend C development.
Read this when encountering compilation errors or unexpected behavior.

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
