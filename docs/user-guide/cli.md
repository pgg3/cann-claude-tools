# 命令行使用指南

## 命令概览

| 命令 | 说明 |
|------|------|
| `generate` | 启动迭代生成流程 |
| `evaluate` | 独立评估解决方案 |

## generate - 生成算子

启动迭代式算子生成流程。

```bash
cann-claude generate <op_name> <python_ref> [选项]
```

**参数**：
- `op_name`: 算子名称
- `python_ref`: Python 参考实现文件路径

**选项**：
| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-o, --output-dir PATH` | 输出目录 | `./output/{op_name}_{timestamp}/` |
| `-n, --iterations INT` | 最大迭代次数 | 10 |
| `-m, --model TEXT` | Claude 模型 | sonnet |
| `--npu-type TEXT` | NPU 类型 | Ascend910B2 |
| `--fake-mode` | 跳过编译（测试用） | False |
| `--continue` | 继续上次运行 | False |

**可用模型**：
- `sonnet` - Claude Sonnet（默认，平衡速度与质量）
- `opus` - Claude Opus（最强能力，较慢）
- `haiku` - Claude Haiku（最快，适合简单任务）

**示例**：

```bash
# 基本用法
cann-claude generate relu ./relu.py

# 指定迭代次数
cann-claude generate relu ./relu.py -n 20

# 使用 Opus 模型
cann-claude generate relu ./relu.py -m opus

# 指定输出目录
cann-claude generate relu ./relu.py -o ./my_output

# 测试模式（跳过编译）
cann-claude generate relu ./relu.py --fake-mode

# 继续上次的实验
cann-claude generate relu ./relu.py -n 10 --continue
```

**输出目录结构**：

```
output/{op_name}_{timestamp}/
├── signature.json            # 解析的算子签名（inputs, outputs, init_params）
├── solution.json             # 最新迭代的解决方案
├── solution_template.json    # 代码模板
├── python_reference.py       # Python 参考实现副本
├── constraints.md            # 格式约束文档
├── vector.md                 # Vector 算子指南
├── cube.md                   # Cube 算子指南
├── .claude_settings.json     # Claude 设置 (systemPrompt)
├── .mcp_config.json          # MCP 配置
├── experience/               # 经验记录
├── solution-1/               # 第 1 次迭代
│   ├── solution.json
│   └── project/              # 编译产物
├── solution-2/               # 第 2 次迭代
├── ...
├── best_solution/            # 性能最优的解决方案 (自动更新)
└── iteration_history.json    # 所有迭代记录
```

**Root 用户注意事项**：

以 root 用户运行时，会自动提示创建专用用户 `cann-claude`：
- Claude Code 无法以 root 身份使用 `--dangerously-skip-permissions`
- 专用用户用于安全地执行 Claude Code
- 用户创建后会自动设置必要的权限
- 输出目录自动调整到 `/home/cann-claude/cann-output/`

## evaluate - 评估解决方案

独立评估已有的解决方案。

```bash
cann-claude evaluate <solution_path> [选项]
```

**参数**：
- `solution_path`: solution.json 文件或包含它的目录

**选项**：
| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--op-name TEXT` | 算子名称（必需） | - |
| `--python-ref PATH` | Python 参考文件（必需） | - |
| `--npu-type TEXT` | NPU 类型 | Ascend910B2 |
| `--fake-mode` | 跳过编译 | False |
| `-o, --output-dir PATH` | 评估产物输出目录 | solution 同级目录 |

**示例**：

```bash
# 评估解决方案
cann-claude evaluate ./output/relu/solution.json \
    --op-name relu \
    --python-ref ./relu.py

# 测试模式
cann-claude evaluate ./solution.json \
    --op-name relu \
    --python-ref ./relu.py \
    --fake-mode
```

## 迭代控制

CLI 负责迭代控制，使用 Claude Code 的会话管理功能：

```
迭代 1: claude -p "..." --session-id UUID --mcp-config ...
迭代 2: claude -p "..." --resume UUID --mcp-config ...
迭代 N: claude -p "..." --resume UUID --mcp-config ...
```

每次迭代后，CLI 会：
1. 读取 solution.json
2. 调用评估器编译和测试
3. 保存到 solution-N/ 目录
4. 根据结果构建下一次迭代的提示

**Prompt 引导策略**：

系统根据算子类型自动引导 Claude 读取正确的文档：
- Vector 算子（relu, add, exp 等）→ `vector.md`
- Cube 算子（matmul, conv2d 等）→ `cube.md`

**配置注入**：
- MCP Server 配置通过 `--mcp-config` 动态传递
- 系统提示（Skill）通过 `--settings` 动态注入
- 无需预先安装或配置

详见 [模块详解](../developer/modules.md) 了解内部实现细节。
