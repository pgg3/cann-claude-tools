# 故障排除

## 首次运行很慢

**症状**：第一次运行 `cann-claude generate` 时等待很久

**原因**：MCP Server 首次启动时会初始化知识库索引

**解决**：这是正常行为，后续运行会使用缓存。缓存位置：`~/.cache/cann_parallel_evaluator/`

---

## cann-parallel-evaluator 模块找不到

**症状**：`iteration_history.json` 显示 "No module named 'cann_parallel_evaluator'"

**解决**：确保 cann-parallel-evaluator 安装在当前 Python 环境中

```bash
python3 -c "from cann_parallel_evaluator import CANNInitTask"
```

---

## 迭代未进行

**症状**：Claude 只运行一次就停止，没有进行多次迭代

**可能原因**：
1. Claude 没有生成 solution.json
2. solution.json 格式错误

**检查**：
```bash
cat ./output/relu_*/iteration_history.json | python3 -m json.tool
```

---

## MCP Server 不工作

**症状**：Claude 说 "cann_search_api not found"

**解决**：安装 MCP SDK
```bash
pip install mcp
```

**注意**：MCP 是可选功能，即使不可用也不影响基本的算子生成。

---

## 评估总是失败

**症状**：所有迭代都显示编译错误

**检查**：
```bash
# 检查 cann-parallel-evaluator
python3 -c "from cann_parallel_evaluator import CANNInitTask"

# 使用 fake-mode 测试流程
cann-claude generate relu ./relu.py --fake-mode
```

---

## 常见编译错误

### TILING_DATA_FIELD_DEF requires 2 arguments

**原因**：`tiling_fields` 写成了字符串而不是 JSON 数组

**错误示例**：
```json
"tiling_fields": "TILING_DATA_FIELD_DEF(uint32_t, totalLength);"
```

**正确格式**：
```json
"tiling_fields": [{"type": "uint32_t", "name": "totalLength"}]
```

### OpCustomTilingData not declared

**原因**：Tiling 类名使用了通用名称而不是算子特定名称

**错误示例**：
```cpp
OpCustomTilingData tiling;
```

**正确格式**：
```cpp
ReluCustomTilingData tiling;  // 对于 relu 算子
```

### no member named GetDimNum / GetDim

**原因**：使用了不存在的 API

**正确方法**：
```cpp
// 获取元素总数
uint32_t totalLength = shape.GetShapeSize();

// 不要使用：
// shape.GetDimNum()
// shape.GetDim(i)
```

### 507035 vector core exception

**原因**：Vector 操作的 count 参数 < 8

**错误示例**：
```cpp
DataCopy(dst, src, 1);
Muls(dst, src, 2.0f, 4);
```

**正确方法**：
```cpp
DataCopy(dst, src, 8);  // count >= 8
Muls(dst, src, 2.0f, 64);
```

### Output value mismatch

**原因**：数据对齐问题，输出大小不是 8 的倍数

**解决**：使用 `DataCopyPad` 代替 `DataCopy`

```cpp
// outW = 46 (不是 8 的倍数)

// ❌ DataCopy 会写入 48 个元素
DataCopy(yGm[offset], yLocal, ((outW + 7) / 8) * 8);

// ✅ DataCopyPad 精确写入 46 个元素
DataCopyExtParams params{1, outW * sizeof(float), 0, 0, 0};
DataCopyPad(yGm[offset], yLocal, params);
```

---

## 调试技巧

### 手动测试评估

```bash
cann-claude evaluate ./solution.json \
    --op-name relu \
    --python-ref ./relu.py
```

### 查看迭代历史

```bash
cat ./output/relu_*/iteration_history.json | python3 -m json.tool
```

### 查看完整编译产物

```bash
ls -la ./output/relu_*/solution-1/project/
```

### 继续之前的实验

```bash
cann-claude generate relu ./relu.py -n 10 --continue
```
