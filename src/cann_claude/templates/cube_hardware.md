# CANN Cube Operator Hardware Specifications

This file contains hardware specifications for Cube operators (matrix operations).

## ⚠️ Scope: Cube Operators Only

This document covers **Cube operators** (matrix operations like MatMul, Conv2D).
For **Vector operators** (ReLU, Add, etc.), see `hardware.md`.

---

## Cube Unit Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    Global Memory (GM)                        │
│                    ↓ LoadData ↓                              │
├─────────────────────────────────────────────────────────────┤
│                    L1 Buffer (~1MB)                          │
│              ┌─────────┴─────────┐                          │
│              ↓                   ↓                          │
│         L0A Buffer          L0B Buffer                      │
│         (Matrix A)          (Matrix B)                      │
│              └─────────┬─────────┘                          │
│                        ↓ Mmad                               │
│                   Cube Unit                                  │
│                        ↓                                    │
│                  L0C Buffer                                  │
│                  (Result C)                                  │
│                        ↓ DataCopy                           │
├─────────────────────────────────────────────────────────────┤
│                    Global Memory (GM)                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Buffer Sizes by NPU Type

| NPU Type | L1 Size | L0A Size | L0B Size | L0C Size |
|----------|---------|----------|----------|----------|
| Ascend910B | 1 MB | 64 KB | 64 KB | 256 KB |
| Ascend910B2 | 1 MB | 64 KB | 64 KB | 256 KB |
| Ascend910B3 | 1 MB | 64 KB | 64 KB | 256 KB |
| Ascend310P | 512 KB | 64 KB | 64 KB | 128 KB |

---

## Tiling Strategy for MatMul

For C[M,N] = A[M,K] × B[K,N]:

### Tile Size Constraints

```cpp
// L0A constraint: A tile must fit in L0A
// A_tile[M_tile, K_tile] * sizeof(dtype) <= L0A_SIZE
// For float16: M_tile * K_tile * 2 <= 64KB

// L0B constraint: B tile must fit in L0B
// B_tile[K_tile, N_tile] * sizeof(dtype) <= L0B_SIZE
// For float16: K_tile * N_tile * 2 <= 64KB

// L0C constraint: C tile must fit in L0C
// C_tile[M_tile, N_tile] * sizeof(dtype) <= L0C_SIZE
// For float16: M_tile * N_tile * 2 <= 256KB

// L1 constraint: A and B tiles must fit in L1
// (A_tile + B_tile) * sizeof(dtype) <= L1_SIZE
```

### Recommended Tile Sizes (float16)

| Matrix Size | M_tile | K_tile | N_tile |
|-------------|--------|--------|--------|
| Small (<1K) | 16 | 16 | 16 |
| Medium (1K-4K) | 64 | 64 | 64 |
| Large (>4K) | 128 | 256 | 128 |

### Alignment Requirements

| Data Type | Alignment | Cube Block Size |
|-----------|-----------|-----------------|
| float16 | 32 bytes | 16×16 |
| float32 | 32 bytes | 16×16 |
| int8 | 32 bytes | 32×32 |

**Tile dimensions must be multiples of Cube block size!**

---

## Dynamic Tiling Calculation

```cpp
// In tiling_func_body for MatMul:
constexpr uint32_t L0A_SIZE = 64 * 1024;   // 64KB
constexpr uint32_t L0B_SIZE = 64 * 1024;   // 64KB
constexpr uint32_t L0C_SIZE = 256 * 1024;  // 256KB
constexpr uint32_t CUBE_BLOCK = 16;        // For float16

uint32_t elementSize = sizeof(half);  // 2 bytes for float16

// Calculate max tile sizes
uint32_t maxMK = L0A_SIZE / elementSize;  // M_tile * K_tile
uint32_t maxKN = L0B_SIZE / elementSize;  // K_tile * N_tile
uint32_t maxMN = L0C_SIZE / elementSize;  // M_tile * N_tile

// Choose balanced tile sizes (example)
uint32_t M_tile = 128;
uint32_t K_tile = 256;
uint32_t N_tile = 128;

// Align to Cube block size
M_tile = (M_tile / CUBE_BLOCK) * CUBE_BLOCK;
K_tile = (K_tile / CUBE_BLOCK) * CUBE_BLOCK;
N_tile = (N_tile / CUBE_BLOCK) * CUBE_BLOCK;

// Calculate number of tiles
uint32_t M_tiles = (M + M_tile - 1) / M_tile;
uint32_t K_tiles = (K + K_tile - 1) / K_tile;
uint32_t N_tiles = (N + N_tile - 1) / N_tile;
```

---

## Performance Optimization

### Checklist

- [ ] Maximize tile sizes within buffer constraints
- [ ] Ensure tile dimensions are multiples of 16 (float16) or 32 (int8)
- [ ] Use double buffering for L1 → L0 transfers
- [ ] Minimize K_tiles (reduces accumulation overhead)
- [ ] Consider data layout (row-major vs column-major)

### Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| L0A overflow | M_tile × K_tile too large | Reduce M_tile or K_tile |
| L0B overflow | K_tile × N_tile too large | Reduce K_tile or N_tile |
| L0C overflow | M_tile × N_tile too large | Reduce M_tile or N_tile |
| Misalignment | Tile not multiple of 16 | Align to CUBE_BLOCK |
