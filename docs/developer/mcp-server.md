# MCP 服务器

`mcp_server.py` 实现了一个 [Model Context Protocol](https://modelcontextprotocol.io/) 服务器，为 Claude 提供 AscendC 知识库查询能力。

## 架构

```
Claude Code ──(stdio JSON-RPC)──► mcp_server.py
                                      │
                                      ▼
                                 get_provider()  ─── 懒加载 ───►  CANNKnowledgeProvider
                                      │                                    │
                                      │                           ┌────────┴────────┐
                                      │                           │  api_scanner    │  API 索引
                                      │                           │  examples/      │  算子示例
                                      │                           │  primers/       │  编程入门
                                      │                           └─────────────────┘
                                      ▼
                              ┌───────────────────────┐
                              │  cann_search_api      │  精确/模糊搜索 API
                              │  cann_get_example     │  获取计算模式示例
                              │  cann_get_knowledge   │  列出全部 API 分类
                              └───────────────────────┘
```

### 传输协议

使用 MCP SDK 的 **stdio 传输**：newline-delimited JSON-RPC 2.0。每条消息一行 JSON，通过 stdin/stdout 交换。

### 懒加载设计

知识库初始化开销较大（扫描 header、解析示例），因此采用懒加载：

1. Server 启动时不加载知识库
2. 第一次 `call_tool` 调用 `get_provider()` 时触发初始化
3. 后续调用复用同一个 `_provider` 实例

### 轻量导入链

直接 `from evotoolkit.task.cann_init import CANNKnowledgeProvider` 会触发整个 `cann_init/__init__.py`，拉入 evaluator、torch 等重依赖。MCP 服务器通过 `importlib.util` 按需加载知识库子模块，完全绕过 `cann_init` 包初始化：

```
_import_knowledge_provider()
├── 加载 knowledge/api_scanner.py
├── 加载 knowledge/examples/curated_examples.py
├── 加载 knowledge/examples/__init__.py
├── 加载 knowledge/primers/level0_programming_model.py
├── 加载 knowledge/primers/level1_patterns.py
├── 加载 knowledge/primers/__init__.py
├── 加载 knowledge/__init__.py
└── 加载 knowledge/provider.py → 返回 CANNKnowledgeProvider
```

### stdout 隔离

MCP 使用 stdio 通道传输 JSON-RPC。知识库初始化时 evotoolkit 可能输出进度信息到 stdout，污染 JSON-RPC 流。`get_provider()` 在初始化期间将 stdout 重定向到 stderr：

```python
original_stdout = sys.stdout
sys.stdout = sys.stderr
try:
    _provider = CANNKnowledgeProvider()
finally:
    sys.stdout = original_stdout
```

### Stub 降级

当 evotoolkit 未安装时，自动降级为 `_StubProvider`，返回安装提示而非崩溃。

## 工具列表

### `cann_search_api`

按名称搜索 AscendC API。

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | string | API 名称（如 `"Add"`、`"DataCopy"`、`"ReduceSum"`） |

**返回**：

```jsonc
// 精确匹配
{
  "status": "found",
  "api_info": {
    "name": "DataCopy",
    "category": "data_copy",
    "description": "format transform(such as nd2nz) during data load from OUT to L1",
    "header": "kernel_operator_data_copy_intf.h"
  },
  "candidates": []
}

// 模糊匹配
{
  "status": "ambiguous",
  "api_info": null,
  "candidates": ["BlockReduceSum", "BlockReduceMax", "BlockReduceMin", "PairReduceSum", "WholeReduceSum"]
}

// 未找到
{
  "status": "not_found",
  "api_info": null,
  "candidates": []
}
```

### `cann_get_example`

获取特定计算模式的完整算子示例，包含全部 6 个代码组件。

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `pattern` | string | 计算模式 |

**可用模式**：

| 类别 | 模式 |
|------|------|
| Vector | `elementwise`, `reduction`, `softmax`, `broadcast`, `pooling` |
| Cube | `matmul`, `convolution`, `attention` |
| Mixed | `normalization`, `index`, `resize` |

**返回**：匹配时返回完整示例文本（含 `kernel_impl`、`kernel_entry_body`、`tiling_fields`、`tiling_func_body`、`infer_shape_body`、`output_alloc_code`）；不匹配时返回 `not_found` 状态和可用模式列表。

### `cann_get_knowledge`

列出所有可用的 AscendC API，按分类组织。

**参数**：无

**返回分类**：

| 分类 | 示例 API |
|------|----------|
| Vector Compute | Abs, Add, Mul, Max, ReduceSum, ... |
| Vector Data | Cast, DataCopy, Duplicate, ... |
| Cube/Matrix | Conv2D, Gemm, Mmad, ... |
| Data Movement | DataCopy, DataCopyPad, ... |
| Scalar | ScalarAbs, ScalarAdd, ... |
| Sync & Atomic | SetAtomicAdd, SetFlag, ... |
| System & Debug | DumpTensor, ASSERT, ... |
| Pipe & Buffer | TPipe, TQue, TBuf, ... |

## 运行方式

### 独立运行

```bash
cd cann-claude-tools
uv run python -m cann_claude.mcp_server
```

服务器启动后等待 stdin 输入 JSON-RPC 消息。按 Ctrl+C 退出。

### Claude Code MCP 配置

CLI 自动生成 `.mcp_config.json` 并通过 `--mcp-config` 传给 Claude Code：

```json
{
  "mcpServers": {
    "cann-tools": {
      "command": "python",
      "args": ["-m", "cann_claude.mcp_server"]
    }
  }
}
```

手动配置到 Claude Code 的 `settings.json`：

```json
{
  "mcpServers": {
    "cann-tools": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/cann-claude-tools", "python", "-m", "cann_claude.mcp_server"]
    }
  }
}
```

## 依赖

| 包 | 用途 | 安装 |
|----|------|------|
| `mcp>=0.9.0` | MCP SDK（Server + stdio 传输） | `pip install cann-claude-tools[mcp]` |
| `evotoolkit` | 知识库 `CANNKnowledgeProvider` | `pip install -e ./evotoolkit` |

当 `mcp` 未安装时，`create_server()` 抛出 `ImportError`。当 `evotoolkit` 未安装时，降级为 `_StubProvider`。

## 测试

用 Python 脚本通过 stdio 发送 JSON-RPC 消息验证：

```python
import asyncio, json, sys

async def test():
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "cann_claude.mcp_server",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )

    async def send(msg):
        proc.stdin.write((json.dumps(msg) + "\n").encode())
        await proc.stdin.drain()

    async def recv():
        line = await asyncio.wait_for(proc.stdout.readline(), timeout=30)
        return json.loads(line)

    # Initialize
    await send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1.0"},
    }})
    print(await recv())

    # Initialized notification
    await send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    # List tools
    await send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    resp = await recv()
    for t in resp["result"]["tools"]:
        print(t["name"])

    # Call a tool
    await send({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
        "name": "cann_search_api", "arguments": {"name": "Add"},
    }})
    print(await recv())

    proc.stdin.close()
    await proc.wait()

asyncio.run(test())
```
