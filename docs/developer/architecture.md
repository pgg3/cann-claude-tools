# 架构详解

本文档详细说明 CANN Claude Tools 的运行逻辑、模块关系和数据流。

## 程序执行流程

### 主命令：`cann-claude generate`

```
用户执行: cann-claude generate relu relu.py -n 10

cli.py:main()
└─ cli.py:generate()
   │
   ├─ 1. 环境检查
   │   ├─ 检查 evotoolkit 可用性
   │   └─ 处理 root 用户（创建 cann-claude 用户）
   │
   ├─ 2. 初始化
   │   ├─ 创建 CANNConfig 对象
   │   ├─ 创建输出目录
   │   ├─ templates.py:generate_solution_template()
   │   │   ├─ detect_operator_type() → Vector/Cube
   │   │   └─ 生成 solution_template.json
   │   ├─ 复制模板文件（3个 md 文件）
   │   ├─ 设置 MCP 配置 (.mcp_config.json)
   │   └─ iteration.py:load_history()
   │
   └─ 3. 迭代循环 (i = 1 to max_iterations)
       │
       ├─ 构建 Prompt
       │   ├─ prompts.py:build_initial_prompt()     [i == 1]
       │   ├─ prompts.py:build_optimization_prompt() [成功时]
       │   └─ prompts.py:build_error_fix_prompt()   [失败时]
       │
       ├─ 调用 Claude Code
       │   └─ cli.py:run_claude()
       │       └─ subprocess: claude -p "..." --session-id/--resume
       │
       ├─ 加载解决方案
       │   └─ evaluator.py:load_solution()
       │
       ├─ 评估解决方案
       │   └─ evaluator.py:evaluate_solution()
       │       └─ evotoolkit.CANNInitTask.evaluate_solution()
       │           ├─ 编译检查
       │           ├─ 正确性验证
       │           └─ 性能测量
       │
       ├─ 保存迭代结果
       │   ├─ iteration.py:save_iteration_solution()
       │   │   └─ evaluator.py:save_solution_files()
       │   ├─ experience.py:record_error()        [失败时]
       │   ├─ experience.py:record_optimization() [改进时]
       │   ├─ iteration.py:save_best_solution()   [最佳时]
       │   └─ iteration.py:save_history()
       │
       └─ 同步经验
           └─ experience.py:sync_tips_to_global()
```

### 时序图

```
CLI                          Claude Code                    Evaluator
 │                               │                              │
 │── 迭代 1: 初始生成 ──────────►│                              │
 │   prompt: "生成 relu 算子"    │                              │
 │                               │── 读取 constraints.md ──►    │
 │                               │── 读取 vector.md ──►         │
 │                               │── 生成 solution.json ──►     │
 │◄──────────────────────────────│                              │
 │                                                              │
 │── 评估 ─────────────────────────────────────────────────────►│
 │◄── 结果: {success: false, error: "..."}  ────────────────────│
 │                                                              │
 │── 保存 solution-1/ ──►                                       │
 │                                                              │
 │── 迭代 2: 修复错误 ─────────►│                               │
 │   prompt: "修复错误: ..."    │                               │
 │   (--resume 保持上下文)       │                               │
 │                               │── 重读文档修复问题 ──►        │
 │                               │── 修复 solution.json ──►     │
 │◄──────────────────────────────│                              │
 │                                                              │
 │── 评估 ─────────────────────────────────────────────────────►│
 │◄── 结果: {success: true, runtime: 0.5ms}  ───────────────────│
 │                                                              │
 │── 保存 solution-2/, best_solution/ ──►                       │
 │                                                              │
 │── 迭代 3..N: 优化性能 ──────►│                               │
 │   ...                         │                               │
```

## 渐进式信息披露

系统采用渐进式披露策略，根据算子类型引导 Claude 读取合适的文档。

### 输出目录结构

```
output/{op_name}_{timestamp}/
├── signature.json              # 解析的算子签名 (inputs, outputs, init_params)
├── solution_template.json      # 代码模板（始终读取）
├── python_reference.py         # Python 参考实现（始终读取）
├── constraints.md              # 格式约束：代码模板结构、JSON格式（始终读取）
├── vector.md                   # Vector 算子指南（Vector 算子时读取）
├── cube.md                     # Cube 算子指南（Cube 算子时读取）
├── .claude_settings.json       # 系统提示词 (skill.md)
├── .mcp_config.json            # MCP 配置
├── experience/                 # 经验记录
├── solution-1/                 # 迭代 1
├── solution-2/                 # 迭代 2
├── best_solution/              # 最优解
└── iteration_history.json      # 迭代历史
```

### 模板文件职责

| 文件 | 职责 | 何时读取 |
|------|------|---------|
| `constraints.md` | 纯格式约束：代码模板结构、JSON格式、命名规则 | 始终（第一个读） |
| `vector.md` | Vector 完整指南：硬件规格、关键规则（对齐、count>=8等） | Vector 算子时 |
| `cube.md` | Cube 完整指南：硬件规格、内存访问模式、tiling策略 | Cube 算子时 |

### Prompt 引导策略

采用渐进式信息披露：

| 层级 | 来源 | 内容 | 何时可见 |
|------|------|------|----------|
| **Level 0** | skill.md | 工作流程、JSON格式要点 | 每轮对话（systemPromptSuffix） |
| **Level 1** | prompts.py | 任务指令：读什么、写什么 | 每次迭代（user prompt） |
| **Level 2** | 文件 | constraints.md (格式), vector.md/cube.md (指南) | Claude 主动读取 |

### 初始 Prompt 示例

```markdown
Generate Ascend C operator `relu` for Ascend910B2.

**Step 1: Research APIs**
- `cann_get_knowledge()` - list available APIs
- `cann_get_example("elementwise")` - get a complete operator example

**Step 2: Read these files** (in this order):
1. `{output_path}/constraints.md` - **CRITICAL**: Code template structure
2. `{output_path}/vector.md` - Hardware specs & critical rules
3. `{output_path}/signature.json` - Operator interface
4. `{ref_path}` - Python reference
5. `{output_path}/solution_template.json` - Example format

**Step 3: Write solution**
Use the Write tool to create `{output_path}/solution.json`
```

## 模块依赖关系

### 依赖图

```
cli.py (主入口)
├── config.py          # 配置管理
├── templates.py       # 解决方案模板生成
├── prompts.py         # Prompt 构建
├── iteration.py       # 迭代历史管理
│   └── evaluator.py
├── evaluator.py       # 核心评估（调用 evotoolkit）
├── experience.py      # 经验记录
└── installer.py       # 包路径工具

mcp_server.py (独立运行)
├── config.py
└── evaluator.py
```

## 核心数据结构

### Solution JSON

Claude 生成的解决方案格式：

```json
{
  "kernel_impl": "class KernelRelu { ... }",
  "kernel_entry_body": "KernelRelu op; op.Init(...); op.Process();",
  "tiling_fields": [
    {"type": "uint32_t", "name": "totalLength"},
    {"type": "uint32_t", "name": "tileNum"}
  ],
  "tiling_func_body": "tiling calculation code",
  "infer_shape_body": "shape inference code",
  "output_alloc_code": "at::Tensor result = at::empty_like(x);"
}
```

### Iteration History JSON

迭代历史记录：

```json
{
  "config": {
    "op_name": "relu",
    "max_iterations": 10,
    "npu_type": "Ascend910B2",
    "session_id": "uuid-string"
  },
  "summary": {
    "total": 3,
    "successful": 2,
    "best_iteration": 2,
    "best_runtime_ms": 0.1234,
    "best_speedup": 10.5,
    "best_score": 0.95
  },
  "iterations": [...]
}
```

### EvaluationResult

评估结果数据类：

```python
@dataclass
class EvaluationResult:
    success: bool                        # 是否成功
    stage: str                           # 阶段: success/compile/correctness/performance 等
    error: Optional[str] = None          # 错误信息
    runtime_ms: Optional[float] = None   # 运行时间 (ms)
    speedup: Optional[float] = None      # 相对 Python 参考的加速比
    score: Optional[float] = None        # 评分 (score = -runtime)
```

## 环境变量

CLI 设置的环境变量（传递给 Claude Code）：

| 变量 | 说明 | 示例 |
|------|------|------|
| `CANN_OP_NAME` | 算子名称 | `relu` |
| `CANN_OUTPUT_DIR` | 输出目录 | `/path/to/output` |
| `CANN_PYTHON_REF` | Python 参考文件路径 | `/path/to/relu.py` |
| `CANN_MAX_ITERATIONS` | 最大迭代次数 | `10` |
| `CANN_NPU_TYPE` | NPU 类型 | `Ascend910B2` |
| `CANN_FAKE_MODE` | 跳过编译模式 | `0` 或 `1` |

## 模板文件

`src/cann_claude/templates/` 目录下的模板文件：

| 文件 | 用途 | 复制到输出目录 |
|------|------|----------------|
| `skill.md` | 工作流程参考（通过 `--settings` 注入） | 否 |
| `constraints.md` | 格式约束：代码模板结构、JSON格式 | ✅ |
| `vector.md` | Vector 算子指南：硬件规格、关键规则 | ✅ |
| `cube.md` | Cube 算子指南：硬件规格、内存访问模式 | ✅ |

**设计原则**：所有 3 个参考文件都复制到输出目录，Prompt 根据算子类型引导 Claude 读取正确的指南。

## 职责划分

### CLI 的职责

1. 迭代循环控制
2. 调用 Claude Code（subprocess）
3. 调用 evaluator 评估解决方案
4. 保存迭代历史
5. 管理 best_solution
6. 格式化反馈给 Claude

### Claude 的职责

1. 读取 constraints.md 理解代码模板结构和 JSON 格式
2. 根据算子类型读取 vector.md 或 cube.md
3. 理解 Python 参考实现
4. 使用 MCP 工具研究 AscendC API
5. 生成/修复 solution.json
6. 根据 CLI 反馈优化

### Claude 不需要做的事

- 调用评估工具（CLI 负责）
- 写测试文件
- 管理迭代历史

## MCP 服务器

`mcp_server.py` 提供的工具：

| 工具 | 描述 | 参数 |
|------|------|------|
| `cann_search_api` | 搜索 AscendC API（精确/模糊匹配） | `name`: API 名称 |
| `cann_get_example` | 获取计算模式的完整算子示例 | `pattern`: 计算模式 |
| `cann_get_knowledge` | 列出全部 API 分类概览 | 无 |

MCP 服务器由 CLI 通过 `--mcp-config` 动态配置，Claude 可以通过这些工具查询知识库。

> 详细设计见 [mcp-server.md](mcp-server.md)

## 外部依赖

### evotoolkit

核心外部依赖，提供：

- `CANNInitTask`: 算子编译、正确性验证、性能测试
- `KnowledgeBase`: AscendC API 和算子示例知识库

安装：
```bash
pip install -e ./evotoolkit[cann_init]
```

### Claude Code CLI

用于与 Claude 交互：

```bash
npm install -g @anthropic-ai/claude-code
```
