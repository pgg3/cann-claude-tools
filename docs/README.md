# CANN Claude Tools 文档

基于 Claude Code 的迭代式 CANN Ascend C 算子生成工具。

## 概述

```
用户命令
    │
    ▼
cann-claude CLI (迭代控制)
    │
    ├── 迭代 1: 初始生成
    │   ├── 调用 Claude Code (--print --session-id)
    │   ├── Claude 生成 solution.json
    │   ├── CLI 评估 (evaluator.py)
    │   └── 保存 solution-1/
    │
    ├── 迭代 2..N: 修复/优化
    │   ├── 调用 Claude Code (--print --resume)
    │   ├── Claude 修复/优化 solution.json
    │   ├── CLI 评估
    │   └── 保存 solution-N/
    │
    └── 最终输出
        ├── best_solution/
        └── iteration_history.json
```

## 快速开始

```bash
# 生成算子（详细选项见 user-guide/cli.md）
cann-claude generate relu ./relu.py -n 10
```

**注意**：
- 如果以 root 用户运行，会自动提示创建专用用户 `cann-claude`
- **首次运行**会自动下载知识库（约 30MB），缓存在 `~/.cache/evotoolkit/`

## 文档导航

**用户指南** (`user-guide/`)
| 文档 | 说明 |
|------|------|
| [cli.md](user-guide/cli.md) | 命令行使用指南 |
| [troubleshooting.md](user-guide/troubleshooting.md) | 故障排除 |

**开发文档** (`developer/`)
| 文档 | 说明 |
|------|------|
| [modules.md](developer/modules.md) | 模块详解 |
| [solution-format.md](developer/solution-format.md) | Solution JSON 格式规范 |
| [extending.md](developer/extending.md) | 扩展开发指南 |

## 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                               用户界面                                       │
│   $ cann-claude generate relu ./relu.py -n 10                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         命令行层 (cli.py) - 迭代控制                         │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  for iteration in 1..N:                                             │   │
│   │    1. 构建 prompt (初始/修复/优化)                                   │   │
│   │    2. 调用 Claude Code (--print --session-id/--resume)              │   │
│   │    3. 读取 solution.json                                            │   │
│   │    4. 调用 evaluator.py 评估                                         │   │
│   │    5. 保存 solution-N/ 和 iteration_history.json                    │   │
│   │    6. 更新 best_solution/ (如果更好)                                 │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Claude Code 运行时                                  │
│   ┌─────────────────┐   ┌─────────────────┐                                 │
│   │     Skill       │   │   MCP Server    │                                 │
│   │   工作流指导     │   │   知识库查询     │                                 │
│   │ (systemPrompt)  │   │  cann-tools     │                                 │
│   └─────────────────┘   └─────────────────┘                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            核心评估模块 (evaluator.py)                       │
│   evaluate_solution() → 编译检查 → 正确性验证 → 性能测量                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 迭代控制流程

```
CLI                          Claude Code                    Evaluator
 │                               │                              │
 │── 迭代 1: 初始生成 ──────────►│                              │
 │   prompt: "生成 relu 算子"    │                              │
 │                               │── 研究 (MCP) ──►             │
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

## 文件结构

```
cann-claude-tools/
├── src/cann_claude/
│   ├── __init__.py         # 包入口
│   ├── cli.py              # 命令行接口 (包含迭代控制)
│   ├── config.py           # 配置管理
│   ├── evaluator.py        # 核心评估模块
│   ├── experience.py       # 经验管理 (错误/优化记录)
│   ├── iteration.py        # 迭代历史管理
│   ├── mcp_server.py       # MCP 服务器 (知识库查询)
│   ├── installer.py        # 包路径工具
│   └── templates/
│       └── skill.md        # Skill 模板 (通过 --settings 注入)
├── docs/                   # 文档目录
└── pyproject.toml          # Python 包配置
```

## 输出目录结构

```
output/{op_name}_{timestamp}/
├── solution.json           # 最新迭代的解决方案
├── python_reference.py     # Python 参考实现副本
├── .claude_settings.json   # Claude 设置 (systemPrompt)
├── .mcp_config.json        # MCP 配置
├── experience/             # 经验记录
│   ├── errors/             # 错误记录 (JSON)
│   └── tips/               # 优化经验 (JSON/Markdown)
├── solution-1/             # 第 1 次迭代
│   ├── solution.json
│   ├── kernel_impl.cpp
│   ├── kernel_entry.cpp
│   ├── tiling_fields.json
│   ├── tiling_func.cpp
│   ├── infer_shape.cpp
│   ├── output_alloc.cpp
│   └── project/            # 编译产物
├── solution-2/             # 第 2 次迭代
├── ...
├── best_solution/          # 性能最优的解决方案 (自动更新)
└── iteration_history.json  # 所有迭代记录
```

## 关键设计说明

### CLI 控制迭代

CLI 使用 `--print` 模式调用 Claude Code，每次迭代是一次独立的调用：
- 第一次迭代使用 `--session-id` 创建会话
- 后续迭代使用 `--resume` 恢复会话上下文
- MCP 配置通过 `--mcp-config` 动态传递
- 系统提示通过 `--settings` 动态注入

### Claude 的职责

Claude 只需要：
1. 理解 Python 参考实现
2. 使用 MCP 工具研究 AscendC API
3. 生成/修复 solution.json
4. 根据 CLI 反馈优化

Claude 不需要：
- 调用评估工具
- 写测试文件
- 管理迭代历史

### CLI 的职责

CLI 负责：
1. 迭代循环控制
2. 调用 Claude Code
3. 评估 solution.json
4. 保存迭代历史
5. 管理 best_solution
6. 格式化反馈给 Claude
