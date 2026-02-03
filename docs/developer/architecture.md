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
   │   ├─ 复制所有参考文件（4个 md 文件）
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
 │                               │── 读取模板和参考 ──►          │
 │                               │── 判断算子类型 ──►            │
 │                               │── 按需读取约束文档 ──►        │
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
 │                               │── 读取约束文档 ──►            │
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

系统采用渐进式披露策略，不预先判断算子类型，让 Claude 自己决定读取哪些文件。

### 输出目录结构

```
output/{op_name}_{timestamp}/
├── signature.json              # 解析的算子签名 (inputs, outputs, init_params)
├── solution_template.json      # 代码模板（始终读取）
├── python_reference.py         # Python 参考实现（始终读取）
├── constraints.md              # Vector 约束 + 签名说明 + 计算模式
├── hardware.md                 # Vector 硬件规格（按需读取）
├── cube_constraints.md         # Cube 约束（按需读取）
├── cube_hardware.md            # Cube 硬件规格（按需读取）
├── .claude_settings.json       # 系统提示词 (skill.md)
├── .mcp_config.json            # MCP 配置
├── experience/                 # 经验记录
├── solution-1/                 # 迭代 1
├── solution-2/                 # 迭代 2
├── best_solution/              # 最优解
└── iteration_history.json      # 迭代历史
```

### Prompt 引导策略

采用渐进式信息披露：

| 层级 | 来源 | 内容 | 何时可见 |
|------|------|------|----------|
| **Level 0** | skill.md | 输出格式、命名约定、错误速查 | 每轮对话（systemPromptSuffix） |
| **Level 1** | prompts.py | 任务指令：读什么、写什么 | 每次迭代（user prompt） |
| **Level 2** | 文件 | signature.json, constraints.md 等 | Claude 主动读取 |

### 初始 Prompt 示例

```markdown
Generate Ascend C operator `relu` for Ascend910B2.

**Read these files first**:
1. `{output_path}/signature.json` - inputs, outputs, init_params
2. `{ref_path}` - Python reference
3. `{output_path}/solution_template.json` - code structure example
4. `{output_path}/constraints.md` - CRITICAL constraints

**Write solution to**: `{output_path}/solution.json`

After writing, I will compile and test it.
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

### 依赖矩阵

```
              cli  config  evaluator  experience  installer  iteration  mcp_server  prompts  templates
cli            -     ✓        ✓          ✓           ✓          ✓          -          ✓        ✓
config         -     -        -          -           -          -          ✓          -        -
evaluator      -     -        -          -           -          ✓          ✓          -        -
experience     -     -        -          -           -          -          -          -        -
installer      -     -        -          -           -          -          -          -        -
iteration      -     -        ✓          -           -          -          -          -        -
mcp_server     -     ✓        ✓          -           -          -          -          -        -
prompts        -     -        -          -           -          -          -          -        -
templates      -     -        -          -           -          -          -          -        -

✓ = 依赖  - = 无依赖
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
  "output_alloc_code": "output tensor allocation",
  "_operator_type": "vector"
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
  "iterations": [
    {
      "iteration": 1,
      "success": false,
      "stage": "compile",
      "error": "error message",
      "solution_dir": "solution-1"
    },
    {
      "iteration": 2,
      "success": true,
      "stage": "complete",
      "runtime_ms": 0.1234,
      "speedup": 10.5,
      "score": 0.95,
      "solution_dir": "solution-2",
      "is_best": true
    }
  ]
}
```

### EvaluationResult

评估结果数据类：

```python
@dataclass
class EvaluationResult:
    success: bool                        # 是否成功
    stage: str                           # 阶段: setup/compile/correctness/performance/complete
    error: Optional[str] = None          # 错误信息
    runtime_ms: Optional[float] = None   # 运行时间 (ms)
    speedup: Optional[float] = None      # 相对 Python 参考的加速比
    score: Optional[float] = None        # 综合评分 (0-1)
    full_code: Optional[Dict] = None     # 完整代码（调试用）
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
| `skill.md` | 快速参考卡（输出格式、命名、错误速查） | 否（通过 `--settings` 注入） |
| `constraints.md` | Vector 约束 + 签名说明 + 计算模式 | ✅ |
| `hardware.md` | Vector 硬件规格 | ✅ |
| `cube_constraints.md` | Cube 算子约束 | ✅ |
| `cube_hardware.md` | Cube 硬件规格 | ✅ |

**设计原则**：所有 4 个参考文件都复制到输出目录，让 Claude 根据算子类型自行决定读取哪些文件。

## 职责划分

### CLI 的职责

1. 迭代循环控制
2. 调用 Claude Code（subprocess）
3. 调用 evaluator 评估解决方案
4. 保存迭代历史
5. 管理 best_solution
6. 格式化反馈给 Claude

### Claude 的职责

1. 理解 Python 参考实现
2. **判断算子类型**（Vector 或 Cube）
3. **按需读取**约束和硬件文档
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
| `cann_search_api` | 搜索 AscendC API | `name`: API 名称 |
| `cann_search_operator` | 查找算子示例代码 | `name`: 算子名称 |
| `cann_get_knowledge` | 获取知识库概览 | 无 |

MCP 服务器由 CLI 通过 `--mcp-config` 动态配置，Claude 可以通过这些工具查询知识库。

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
