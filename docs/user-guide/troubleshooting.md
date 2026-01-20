# 故障排除

## 首次运行很慢

**症状**：第一次运行 `cann-claude generate` 时等待很久

**原因**：MCP Server 首次启动时会自动下载知识库（约 30MB）并建立索引

**解决**：这是正常行为，后续运行会使用缓存。缓存位置：`~/.cache/evotoolkit/cann_initer/`

---

## evotoolkit 模块找不到

**症状**：`iteration_history.json` 显示 "No module named 'evotoolkit'"

**解决**：确保 evotoolkit 安装在系统 Python 中

```bash
python3 -c "from evotoolkit.task.cann_init import CANNInitTask"
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
# 检查 evotoolkit
python3 -c "from evotoolkit.task.cann_init import CANNInitTask"

# 使用 fake-mode 测试流程
cann-claude generate relu ./relu.py --fake-mode
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
