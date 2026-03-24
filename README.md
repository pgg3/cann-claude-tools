# CANN Claude Tools

基于 Claude Code 的迭代式 CANN Ascend C 算子自动生成工具。

## 特性

- **迭代优化**：自动迭代 N 次，选择性能最优的解
- **知识库集成**：通过 MCP Server 查询 AscendC API 和算子示例
- **经验积累**：自动记录错误和优化经验，跨运行复用
- **自动签名解析**：从 Python 参考代码解析 inputs/outputs/init_params

## ⚠️ 支持范围

| 类型 | 算子示例 | 计算单元 | 状态 |
|------|----------|----------|------|
| **Vector** | ReLU, Abs, Exp, Add, Mul, Sqrt | Vector Unit | ✅ 完整支持 |
| **Cube** | MatMul, Conv2D, GEMM | Cube Unit | ⚠️ 模板支持 (需要实现 Process) |

系统会根据算子名称自动检测类型并生成对应模板。

## 前提条件

- **CANN 开发环境**：msopgen、Ascend C 编译工具链、NPU 驱动
- **Claude Code CLI**：`npm install -g @anthropic-ai/claude-code`
- **ANTHROPIC_API_KEY**：需设置环境变量（会自动传递给子进程）
- **Python 3.10+**

## 安装

```bash
# 克隆仓库
git clone https://github.com/pgg3/cann-claude-tools.git
cd cann-claude-tools

# 创建 venv（需要使用 NPU 环境的 Python，以访问 torch-npu）
uv venv --python /path/to/npu-python --system-site-packages

# 安装依赖（含 MCP server）
uv sync --extra mcp
```

## 快速开始

```bash
# 生成 ReLU 算子（迭代 3 次）
cann-claude generate relu ./relu.py -n 3

# 生成 Add 算子（使用 opus 模型，迭代 10 次）
cann-claude generate add ./add.py -m opus -n 10

# 继续上次的实验（自动查找最近的 relu 输出目录）
cann-claude generate relu ./relu.py -n 10 --continue

# 评估已有解决方案
cann-claude evaluate ./solution.json --op-name relu --python-ref ./relu.py
```

### Python Reference 格式

使用 MultiKernelBench 格式：

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)

# 测试参数
batch_size, channels, height, width = 16, 64, 512, 512
kernel_size = 11

def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]

def get_init_inputs():
    return [kernel_size]
```

### 输出结构

```
output/relu_20260120_201849/
├── signature.json            # 解析的算子签名（inputs, outputs, init_params）
├── solution_template.json    # 模板（Claude 参考）
├── python_reference.py       # Python 参考实现
├── constraints.md            # 格式约束（代码模板结构、JSON格式）
├── vector.md                 # Vector 算子指南（硬件规格、关键规则）
├── cube.md                   # Cube 算子指南（硬件规格、内存访问模式）
├── iteration_history.json    # 迭代历史记录
├── solution-1/               # 第 1 次迭代
│   ├── solution.json         # 生成的解决方案
│   └── project/              # 编译产物
├── solution-2/               # 第 2 次迭代
├── solution-3/               # 第 3 次迭代
└── best_solution/            # 最优解（复制）
```

## Root 用户说明

如果以 root 身份运行，工具会自动创建 `cann-claude` 用户来执行 Claude Code（因为 Claude Code 不允许 root 使用 `--dangerously-skip-permissions`）。

首次运行时会提示：
```
Root User Detected
Running as root detected.
Claude Code cannot use --dangerously-skip-permissions as root.
A dedicated user 'cann-claude' is needed.

Create dedicated user 'cann-claude'? [Y/n]:
```

选择 `Y` 后会自动：
1. 创建 `cann-claude` 用户
2. 配置 API Key
3. 设置必要的目录权限

## 故障排除

### evotoolkit not installed

```
Error: evotoolkit not installed.
Run: pip install -e ./evotoolkit[cann_init]
```

需要安装 evotoolkit 子模块。

### claude command not found

```
Error: claude command not found.
Please install Claude Code: npm install -g @anthropic-ai/claude-code
```

需要安装 Claude Code CLI。

### msopgen not found

确保 CANN 环境已正确安装并配置了环境变量：
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 文档

详细文档见 [docs/README.md](docs/README.md)

- [命令行使用指南](docs/user-guide/cli.md)
- [故障排除](docs/user-guide/troubleshooting.md)
- [模块详解](docs/developer/modules.md)
- [Solution 格式规范](docs/developer/solution-format.md)

## 许可证

MIT License
