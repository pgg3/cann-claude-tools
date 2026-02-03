# Solution 格式规范

## solution.json 结构

```json
{
  "kernel_impl": "...",        // 内核类实现
  "kernel_entry_body": "...",  // 入口函数体
  "tiling_fields": [...],      // Tiling 字段定义（JSON 数组）
  "tiling_func_body": "...",   // Tiling 函数体
  "infer_shape_body": "...",   // 形状推断函数体
  "output_alloc_code": "..."   // 输出分配代码（C++）
}
```

## 字段类型要求

| 字段 | 类型 | 说明 |
|------|------|------|
| `kernel_impl` | string | 内核类定义代码 |
| `kernel_entry_body` | string | 实例化并调用内核的代码 |
| `tiling_fields` | **JSON array** | `[{"type": "uint32_t", "name": "xxx"}]` |
| `tiling_func_body` | string | Host 端 tiling 计算代码 |
| `infer_shape_body` | string | 形状推断代码 |
| `output_alloc_code` | string (C++) | `at::Tensor result = ...;` |

## 常见格式错误

### tiling_fields 必须是 JSON 数组

```json
// ❌ WRONG: 字符串格式
"tiling_fields": "TILING_DATA_FIELD_DEF(uint32_t, totalLength);"

// ✅ CORRECT: JSON 数组格式
"tiling_fields": [{"type": "uint32_t", "name": "totalLength"}]
```

错误信息：`TILING_DATA_FIELD_DEF requires 2 arguments`

### output_alloc_code 必须是 C++

```json
// ❌ WRONG: Python 代码
"output_alloc_code": "output = torch.empty_like(x)"

// ✅ CORRECT: C++ 代码
"output_alloc_code": "at::Tensor result = at::empty_like(x);"
```

### Tiling 类名必须正确

```cpp
// ❌ WRONG: 通用名称
OpCustomTilingData tiling;

// ✅ CORRECT: 算子特定名称
ReluCustomTilingData tiling;       // 对于 relu 算子
AvgPoolCustomTilingData tiling;    // 对于 avg_pool 算子
```

错误信息：`OpCustomTilingData not declared in this scope`

## 命名约定

对于算子 `foo_bar`：

| 项目 | 名称 |
|------|------|
| 内核类 | `KernelFooBar` |
| Tiling 数据类 | `FooBarCustomTilingData` |
| 入口函数 | `foo_bar_custom`（自动生成） |
| Tiling setter | `tiling.set_totalLength(value)` |

## 完整示例（ReLU）

```json
{
  "kernel_impl": "using namespace AscendC;\n\nclass KernelRelu {\npublic:\n    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength) {\n        xGm.SetGlobalBuffer((__gm__ float*)x, totalLength);\n        yGm.SetGlobalBuffer((__gm__ float*)y, totalLength);\n        this->totalLength = totalLength;\n        pipe.InitBuffer(inQue, 1, totalLength * sizeof(float));\n        pipe.InitBuffer(outQue, 1, totalLength * sizeof(float));\n    }\n    __aicore__ inline void Process() {\n        LocalTensor<float> xLocal = inQue.AllocTensor<float>();\n        DataCopy(xLocal, xGm, totalLength);\n        inQue.EnQue(xLocal);\n        xLocal = inQue.DeQue<float>();\n        LocalTensor<float> yLocal = outQue.AllocTensor<float>();\n        Relu(yLocal, xLocal, totalLength);\n        outQue.EnQue(yLocal);\n        inQue.FreeTensor(xLocal);\n        yLocal = outQue.DeQue<float>();\n        DataCopy(yGm, yLocal, totalLength);\n        outQue.FreeTensor(yLocal);\n    }\nprivate:\n    TPipe pipe;\n    TQue<QuePosition::VECIN, 1> inQue;\n    TQue<QuePosition::VECOUT, 1> outQue;\n    GlobalTensor<float> xGm, yGm;\n    uint32_t totalLength;\n};",

  "kernel_entry_body": "    KernelRelu op;\n    op.Init(x, output, tilingData.totalLength);\n    op.Process();",

  "tiling_fields": [
    {"type": "uint32_t", "name": "totalLength"}
  ],

  "tiling_func_body": "    ReluCustomTilingData tiling;\n    auto shape = context->GetInputShape(0)->GetStorageShape();\n    uint32_t totalLength = shape.GetShapeSize();\n    tiling.set_totalLength(totalLength);\n    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());\n    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());\n    context->SetBlockDim(1);\n    size_t* ws = context->GetWorkspaceSizes(1);\n    ws[0] = 0;\n    return ge::GRAPH_SUCCESS;",

  "infer_shape_body": "    *context->GetOutputShape(0) = *context->GetInputShape(0);\n    return ge::GRAPH_SUCCESS;",

  "output_alloc_code": "at::Tensor result = at::empty_like(x);"
}
```

## 代码模板上下文

了解你的代码如何被组装到最终模板中。详见 `constraints.md`。

| 字段 | 可用变量 | 说明 |
|------|----------|------|
| `kernel_impl` | 无 | 定义内核类 |
| `kernel_entry_body` | `tilingData`, GM_ADDR 参数 | 通过 `tilingData.xxx` 访问 tiling 字段 |
| `tiling_func_body` | `context` (TilingContext*) | 运行在 HOST CPU |
| `infer_shape_body` | `context` (InferShapeContext*) | 运行在 HOST CPU |

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
│       - 编译检查                                                          │
│       - 正确性验证                                                        │
│       - 性能测量                                                          │
│    5. 记录到 iteration_history.json                                      │
│    6. 更新 best_solution/ (如果更优)                                       │
│    7. 构建下一次迭代的反馈 prompt                                          │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```
