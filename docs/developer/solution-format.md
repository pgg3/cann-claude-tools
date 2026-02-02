# Solution 格式规范

## solution.json 结构

```json
{
  "kernel_impl": "...",        // 内核类实现
  "kernel_entry_body": "...",  // 入口函数体
  "tiling_fields": [...],      // Tiling 字段定义
  "tiling_func_body": "...",   // Tiling 函数体
  "infer_shape_body": "...",   // 形状推断函数体
  "output_alloc_code": "..."   // 输出分配代码
}
```

## 组件说明

### 1. kernel_impl

内核类实现，包含完整的 Ascend C 内核代码。

```cpp
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelRelu {
public:
    __aicore__ inline KernelRelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum) {
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        // 初始化全局内存和缓冲区
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
    __aicore__ inline void CopyIn(int32_t progress) { /* ... */ }
    __aicore__ inline void Compute(int32_t progress) { Relu(yLocal, xLocal, tileLength); }
    __aicore__ inline void CopyOut(int32_t progress) { /* ... */ }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<float> xGm, yGm;
    uint32_t blockLength, tileNum, tileLength;
};
```

### 2. kernel_entry_body

内核入口函数体，负责初始化和调用内核。使用 `tilingData.xxx` 访问 tiling 参数。

```cpp
KernelRelu op;
op.Init(x, output, tilingData.totalLength, tilingData.tileNum);
op.Process();
```

### 3. tiling_fields

Tiling 字段定义，支持两种格式：

**列表格式（推荐）**：
```json
[
  {"type": "uint32_t", "name": "totalLength"},
  {"type": "uint32_t", "name": "tileNum"}
]
```

**字符串格式（兼容）**：
```
"uint32_t totalLength;\nuint32_t tileNum;"
```

evaluator 会自动将字符串格式转换为列表格式。

### 4. tiling_func_body

Tiling 函数体，计算分块参数。使用 `XxxCustomTilingData` 类和 `set_xxx()` 方法。

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
// UB constraints for Ascend910B2
constexpr uint32_t UB_SAFE_SIZE = 64 * 1024;  // 64KB safe limit
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

### 5. infer_shape_body

形状推断函数体。

```cpp
const gert::Shape* x_shape = context->GetInputShape(0);
gert::Shape* y_shape = context->GetOutputShape(0);
*y_shape = *x_shape;
return ge::GRAPH_SUCCESS;
```

### 6. output_alloc_code

输出张量分配代码。

```cpp
at::Tensor result = at::empty_like(x);
```

## 数据流

```
                    ┌──────────────────┐
                    │  solution.json   │
                    └────────┬─────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          CLI 迭代循环 (cli.py)                            │
│                                                                          │
│  for iteration in 1..N:                                                  │
│    1. 调用 Claude Code (--print --session-id/--resume)                   │
│    2. 读取 solution.json                                                 │
│    3. 保存到 solution-{N}/ 目录                                           │
│    4. 调用 evaluator.evaluate_solution()                                 │
│       - 设置 umask(0o022) 确保文件权限                                    │
│       - 编译检查                                                          │
│       - 正确性验证                                                        │
│       - 性能测量                                                          │
│    5. 记录到 iteration_history.json                                      │
│    6. 更新 best_solution/ (如果更优)                                       │
│    7. 构建下一次迭代的反馈 prompt                                          │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
       ┌────────────┐               ┌────────────┐
       │  继续迭代  │               │  完成/退出  │
       │ (i < N)    │               │ (i >= N)   │
       └────────────┘               └────────────┘
```

## iteration_history.json 格式

```json
{
  "config": {
    "op_name": "relu",
    "max_iterations": 10,
    "npu_type": "Ascend910B2",
    "session_id": "uuid-string"
  },
  "summary": {
    "total": 10,
    "successful": 7,
    "best_iteration": 5,
    "best_runtime_ms": 0.0156,
    "best_speedup": 18.2,
    "best_score": 0.9845
  },
  "iterations": [
    {
      "iteration": 1,
      "success": false,
      "stage": "compile",
      "error": "...",
      "solution_dir": "solution-1"
    },
    {
      "iteration": 2,
      "success": true,
      "stage": "complete",
      "runtime_ms": 0.025,
      "speedup": 12.5,
      "score": 0.85,
      "solution_dir": "solution-2",
      "is_best": true
    }
  ]
}
```
