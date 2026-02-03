# CANN Hardware Specifications

This file contains hardware specifications for Ascend NPUs.
Read this when optimizing performance or debugging UB overflow errors.

## ⚠️ Scope: Vector Operators Only

This document covers **Vector operators** (element-wise operations).
For **Cube operators** (MatMul, Conv2D), different tiling strategies apply.

---

## Memory Constraints by NPU Type

| NPU Type | UB Size | Safe Limit | BLOCK_DIM | Alignment |
|----------|---------|------------|-----------|-----------|
| Ascend910B | 256 KB | 64 KB | 8 | 32 bytes |
| Ascend910B2 | 256 KB | 64 KB | 8 | 32 bytes |
| Ascend910B3 | 256 KB | 64 KB | 8 | 32 bytes |
| Ascend310P | 256 KB | 64 KB | 8 | 32 bytes |

---

## UB (Unified Buffer) Constraints

The Unified Buffer is the on-chip memory for vector operations.
**Each tile MUST fit in UB!**

### Calculation Formula

```
maxTileBytes = UB_SAFE_SIZE / (NUM_BUFFERS * BUFFER_NUM)
             = 64KB / (2 * 2) = 16KB per buffer

For float32 (4 bytes): maxTileElements = 16KB / 4 = 4096 elements
For float16 (2 bytes): maxTileElements = 16KB / 2 = 8192 elements
```

### Dynamic tileNum Calculation

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

### Why Dynamic Tiling is Essential

- Fixed `tileNum = 8` causes UB overflow for large tensors
- Example: 1.6B elements → tileLength = 12.5M floats = 48MB (UB is only 256KB!)
- Dynamic calculation ensures each tile fits in UB regardless of input size

---

## Alignment Requirements

| Data Type | Alignment | Elements per Block |
|-----------|-----------|-------------------|
| float32 | 32 bytes | 8 elements |
| float16 | 32 bytes | 16 elements |
| int32 | 32 bytes | 8 elements |

**Always align tileLength**:
```cpp
tileLength = (tileLength / alignElements) * alignElements;
```

---

## Performance Optimization Checklist

When optimizing (solution already passes):

- [ ] Increase `tileNum`: 8 → 16 → 32
- [ ] Increase `BUFFER_NUM`: 2 → 4
- [ ] Reduce data movement (fuse CopyIn operations)
- [ ] Use half precision if possible
- [ ] Check 32-byte memory alignment
