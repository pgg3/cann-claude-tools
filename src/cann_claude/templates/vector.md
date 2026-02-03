# Vector Operator Guide

Complete guide for Vector operators (element-wise operations like ReLU, Add, Exp).

---

## Memory Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Global Memory (GM)                        │
│                         ↓ DataCopy                          │
├─────────────────────────────────────────────────────────────┤
│              Unified Buffer (UB) - 256KB                     │
│         ┌──────────────┬──────────────┐                     │
│         │   VECIN      │   VECOUT     │                     │
│         │  (input)     │  (output)    │                     │
│         └──────────────┴──────────────┘                     │
│                    ↓ Vector Unit                            │
│           Relu, Add, Muls, Exp, etc.                        │
├─────────────────────────────────────────────────────────────┤
│                    Global Memory (GM)                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Hardware Specifications

| NPU Type | UB Size | Safe Limit | BLOCK_DIM | Alignment |
|----------|---------|------------|-----------|-----------|
| Ascend910B | 256 KB | 64 KB | 8 | 32 bytes |
| Ascend910B2 | 256 KB | 64 KB | 8 | 32 bytes |
| Ascend910B3 | 256 KB | 64 KB | 8 | 32 bytes |
| Ascend310P | 256 KB | 64 KB | 8 | 32 bytes |

---

## Critical Rules

### 1. Minimum Operation Size
**All vector operations require count >= 8 (32 bytes for float32)**

```cpp
// ❌ CRASH - count < 8
DataCopy(dst, src, 1);
Muls(dst, src, 2.0f, 4);

// ✅ OK - count >= 8
DataCopy(dst, src, 8);
Muls(dst, src, 2.0f, 64);
```

Runtime error if violated: `507035 vector core exception`

### 2. Data Alignment
When output size is NOT a multiple of 8, use `DataCopyPad`:

```cpp
// outW = 46 (not aligned)

// ❌ DataCopy writes 48 elements, corrupts adjacent data
DataCopy(yGm[offset], yLocal, ((outW + 7) / 8) * 8);

// ✅ DataCopyPad writes exactly 46 elements
DataCopyExtParams params{1, outW * sizeof(float), 0, 0, 0};
DataCopyPad(yGm[offset], yLocal, params);
```

### 3. Type Casting
**Kernel code does NOT support float↔int casts!**

```cpp
// ❌ WRONG in kernel
float inv = 1.0f / static_cast<float>(poolSize);

// ✅ CORRECT - compute in tiling_func_body (host), pass via tiling
// tiling_func_body:
tiling.set_invPoolSize(1.0f / static_cast<float>(kernel_size * kernel_size));
// kernel:
float inv = tilingData.invPoolSize;
```

---

## Tiling Calculation

### Dynamic Tiling Formula

```cpp
// In tiling_func_body:
constexpr uint32_t UB_SAFE_SIZE = 64 * 1024;  // 64KB safe limit
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t NUM_BUFFERS = 2;  // input + output
uint32_t elementSize = sizeof(float);

// Max elements per tile
uint32_t maxTileElements = UB_SAFE_SIZE / (NUM_BUFFERS * BUFFER_NUM * elementSize);
maxTileElements = (maxTileElements / 8) * 8;  // Align to 32 bytes

// Calculate tileNum
uint32_t blockLength = totalLength / BLOCK_DIM;
uint32_t tileNum = (blockLength + maxTileElements - 1) / maxTileElements;
```

### Why Dynamic Tiling
- Fixed `tileNum = 8` causes UB overflow for large tensors
- Example: 1.6B elements → 48MB tile (UB is only 256KB!)

---

## Valid Queue Positions

| Position | Use For |
|----------|---------|
| `QuePosition::VECIN` | Input buffers |
| `QuePosition::VECOUT` | Output buffers |
| `QuePosition::VECCALC` | Intermediate buffers |

Note: `TEMP`, `VECTMP`, `VECBUF` do NOT exist.

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `507035 vector core exception` | count < 8 | Ensure all ops use count >= 8 |
| `Output value mismatch` | Data alignment | Use `DataCopyPad` for non-aligned |
| `UB address out of bounds` | Tile too large | Use dynamic tiling |
| `cast between floating and unsigned` | Type cast in kernel | Move to tiling_func_body |

---

## Performance Optimization

When solution passes, try:
- [ ] Increase `tileNum`: 8 → 16 → 32
- [ ] Increase `BUFFER_NUM`: 2 → 4
- [ ] Use half precision if possible
- [ ] Fuse multiple CopyIn operations

---

## API Quick Reference

Use `cann_search_api("ApiName")` for full signatures.

| Category | APIs |
|----------|------|
| Data Movement | `DataCopy`, `DataCopyPad` |
| Arithmetic | `Add`, `Muls`, `Adds`, `Mul` |
| Activation | `Relu`, `Abs`, `Exp`, `Sqrt` |
| Reduction | `ReduceSum`, `ReduceMax`, `ReduceMin` |
| Compare | `CompareScalar` (NOT `Compare`) |
