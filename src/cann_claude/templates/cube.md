# Cube Operator Guide

Complete guide for Cube operators (matrix operations like MatMul, Conv2D).

---

## Memory Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Global Memory (GM)                        │
│                    ↓ DataCopy ↓                              │
├─────────────────────────────────────────────────────────────┤
│                    L1 Buffer (~1MB)                          │
│              ┌─────────┴─────────┐                          │
│              ↓ LoadData          ↓ LoadData                 │
│         L0A Buffer          L0B Buffer                      │
│         (Matrix A)          (Matrix B)                      │
│         (64 KB)             (64 KB)                         │
│              └─────────┬─────────┘                          │
│                        ↓ Mmad                               │
│                   Cube Unit                                  │
│                        ↓                                    │
│                  L0C Buffer                                  │
│                  (Result C)                                  │
│                  (256 KB)                                    │
│                        ↓ DataCopy                           │
├─────────────────────────────────────────────────────────────┤
│                    Global Memory (GM)                        │
└─────────────────────────────────────────────────────────────┘
```

**Data Flow: GM → L1 → L0A/L0B → Cube → L0C → GM**

---

## Hardware Specifications

| NPU Type | L1 Size | L0A Size | L0B Size | L0C Size |
|----------|---------|----------|----------|----------|
| Ascend910B | 1 MB | 64 KB | 64 KB | 256 KB |
| Ascend910B2 | 1 MB | 64 KB | 64 KB | 256 KB |
| Ascend910B3 | 1 MB | 64 KB | 64 KB | 256 KB |
| Ascend310P | 512 KB | 64 KB | 64 KB | 128 KB |

---

## Critical Rules

### 1. Memory Access Pattern

```cpp
// ✅ CORRECT - GM → L1 → L0 → Cube → L0C → GM
DataCopy(l1A, gmA[offset], size);    // Step 1: GM → L1
DataCopy(l1B, gmB[offset], size);
LoadData(l0A, l1A, ...);             // Step 2: L1 → L0
LoadData(l0B, l1B, ...);
Mmad(l0C, l0A, l0B, ...);            // Step 3: Cube computation
DataCopy(gmC[offset], l0C, size);    // Step 4: L0C → GM

// ❌ WRONG - Cannot skip L1
LoadData(l0A, gmA, ...);             // Cannot load directly from GM!

// ❌ WRONG - Cannot write directly to GM
Mmad(gmC, l0A, l0B, ...);            // Cannot write directly to GM!
```

### 2. No Vector APIs in Cube Kernel

```cpp
// ❌ WRONG - These are Vector Unit APIs
Relu(), Abs(), Exp(), Add()

// ❌ WRONG - These are Vector queues
TQue<QuePosition::VECIN, ...>
TQue<QuePosition::VECOUT, ...>

// ✅ CORRECT - Use Cube APIs
Mmad(), LoadData(), DataCopy()
```

### 3. Alignment Requirements

| Data Type | Cube Block Size | Alignment |
|-----------|-----------------|-----------|
| float16 | 16×16 | 32 bytes |
| float32 | 16×16 | 32 bytes |
| int8 | 32×32 | 32 bytes |

**Tile dimensions must be multiples of Cube block size!**

```cpp
// Pad to alignment if needed
uint32_t M_aligned = ((M + 15) / 16) * 16;
uint32_t K_aligned = ((K + 15) / 16) * 16;
uint32_t N_aligned = ((N + 15) / 16) * 16;
```

---

## Tiling Strategy for MatMul

For C[M,N] = A[M,K] × B[K,N]:

### Buffer Constraints

```cpp
// L0A: A_tile[M_tile, K_tile] must fit
M_tile * K_tile * sizeof(dtype) <= 64KB

// L0B: B_tile[K_tile, N_tile] must fit
K_tile * N_tile * sizeof(dtype) <= 64KB

// L0C: C_tile[M_tile, N_tile] must fit
M_tile * N_tile * sizeof(dtype) <= 256KB

// L1: A_tile + B_tile must fit
(M_tile * K_tile + K_tile * N_tile) * sizeof(dtype) <= 1MB
```

### Recommended Tile Sizes (float16)

| Matrix Size | M_tile | K_tile | N_tile |
|-------------|--------|--------|--------|
| Small (<1K) | 16 | 16 | 16 |
| Medium (1K-4K) | 64 | 64 | 64 |
| Large (>4K) | 128 | 256 | 128 |

### Dynamic Tiling Code

```cpp
// In tiling_func_body:
constexpr uint32_t L0A_SIZE = 64 * 1024;
constexpr uint32_t L0B_SIZE = 64 * 1024;
constexpr uint32_t L0C_SIZE = 256 * 1024;
constexpr uint32_t CUBE_BLOCK = 16;
uint32_t elementSize = sizeof(half);

// Start with target tile sizes
uint32_t M_tile = 128;
uint32_t K_tile = 256;
uint32_t N_tile = 128;

// Reduce if exceeds buffer
while (M_tile * K_tile * elementSize > L0A_SIZE && M_tile > CUBE_BLOCK)
    M_tile /= 2;
while (K_tile * N_tile * elementSize > L0B_SIZE && K_tile > CUBE_BLOCK)
    K_tile /= 2;
while (M_tile * N_tile * elementSize > L0C_SIZE && N_tile > CUBE_BLOCK)
    N_tile /= 2;

// Align to Cube block size
M_tile = (M_tile / CUBE_BLOCK) * CUBE_BLOCK;
K_tile = (K_tile / CUBE_BLOCK) * CUBE_BLOCK;
N_tile = (N_tile / CUBE_BLOCK) * CUBE_BLOCK;
```

---

## Tiling Fields for MatMul

```json
"tiling_fields": [
    {"type": "uint32_t", "name": "M"},
    {"type": "uint32_t", "name": "K"},
    {"type": "uint32_t", "name": "N"},
    {"type": "uint32_t", "name": "M_tile"},
    {"type": "uint32_t", "name": "K_tile"},
    {"type": "uint32_t", "name": "N_tile"}
]
```

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `L0A overflow` | M_tile × K_tile too large | Reduce M_tile or K_tile |
| `L0B overflow` | K_tile × N_tile too large | Reduce K_tile or N_tile |
| `L0C overflow` | M_tile × N_tile too large | Reduce M_tile or N_tile |
| `Misaligned access` | Tile not multiple of 16 | Align to CUBE_BLOCK |
| `Invalid buffer` | Using Vector buffer | Use L1/L0 buffers |

---

## Performance Optimization

- [ ] Maximize tile sizes within buffer constraints
- [ ] Use double buffering for L1 → L0 transfers
- [ ] Minimize K_tiles (reduces accumulation overhead)
- [ ] Consider data layout (row-major vs column-major)

---

## API Quick Reference

Use `cann_search_api("ApiName")` for full signatures.

| Category | APIs |
|----------|------|
| Matrix Multiply | `Mmad` |
| Data Movement | `LoadData`, `DataCopy` |
| Synchronization | `SetFlag`, `WaitFlag` |
