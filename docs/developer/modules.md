# 模块详解

## cli.py - 命令行接口与迭代控制

**职责**：用户入口、迭代控制核心、协调整个流程

### 核心架构

CLI 负责迭代控制，使用 `--print` 模式调用 Claude Code：

```
for iteration in 1..N:
    1. 构建 prompt（初始/修复/优化）
    2. 调用 Claude Code（--print --session-id/--resume）
    3. 读取 solution.json
    4. 调用 evaluator.py 评估
    5. 保存 solution-N/ 和 iteration_history.json
    6. 更新 best_solution/（如果更好）
```

### Root 用户处理

```python
CANN_USER = "cann-claude"  # 专用用户名

def is_root() -> bool
def cann_user_exists() -> bool
def create_cann_user() -> bool
def run_as_cann_user(cmd, env, cwd) -> int
```

Root 用户无法直接使用 `--dangerously-skip-permissions`，因此：
1. `generate` 命令自动提示创建专用用户
2. 自动切换到专用用户执行
3. 输出目录自动调整到 `/home/cann-claude/cann-output/`
4. 使用 `--print` 标志确保非交互式文本输出

### 用户切换实现

`run_as_cann_user()` 函数使用 `runuser -u` 执行命令：

```python
def run_as_cann_user(cmd: list, env: dict, cwd: Path) -> int:
    # 准备环境 - 保持当前环境，但覆盖 HOME
    run_env = env.copy()
    run_env["HOME"] = str(user_home)

    # runuser -u 自动传递所有环境变量
    # 使用 DEVNULL 作为 stdin 避免非交互式模式下阻塞
    result = subprocess.run(
        ["runuser", "-u", CANN_USER, "--"] + cmd,
        cwd=cwd,
        env=run_env,
        stdin=subprocess.DEVNULL,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    return result.returncode
```

---

## config.py - 配置管理

**职责**：环境变量与配置的转换

```python
@dataclass
class CANNConfig:
    op_name: str           # 算子名称
    output_dir: Path       # 输出目录
    python_ref_path: Path  # Python 参考文件路径
    max_iterations: int    # 最大迭代次数 (默认: 10)
    npu_type: str          # NPU 类型 (默认: Ascend910B2)
    fake_mode: bool        # 跳过编译 (默认: False)

    @classmethod
    def from_env(cls) -> Optional["CANNConfig"]  # 从环境变量加载
    def to_env(self) -> dict[str, str]           # 导出为环境变量
    def set_env(self)                            # 设置到 os.environ
```

---

## iteration.py - 迭代历史管理

**职责**：迭代历史和解决方案文件管理

### 主要函数

```python
# 历史管理
def load_history(output_dir: Path) -> Dict
def save_history(output_dir: Path, history: Dict)

# 解决方案保存
def save_iteration_solution(output_dir: Path, solution: Dict, iteration: int) -> Path
def save_best_solution(output_dir: Path, solution: Dict, iteration: int)
```

### 使用示例

```python
from .iteration import load_history, save_history, save_iteration_solution

history = load_history(output_path)
save_iteration_solution(output_path, solution, iteration)
history["iterations"].append(iteration_record)
save_history(output_path, history)
```

---

## evaluator.py - 核心评估模块

**职责**：核心评估逻辑，被 CLI 调用，使用 CANNInitTask 执行

### 评估执行

评估使用 evotoolkit 的 `CANNInitTask` 进行完整的编译、正确性验证和性能测试：

```python
# evaluator.py 中的调用
from evotoolkit.task.cann_init import CANNInitTask, CANNSolutionConfig
from evotoolkit.core import Solution

task = CANNInitTask(data=task_data, fake_mode=False)
config = CANNSolutionConfig(
    project_path=project_path,
    kernel_impl=solution.get("kernel_impl", ""),
    # ... 其他组件
)
sol = Solution(sol_string="", other_info=config.to_dict())
result = task.evaluate_solution(sol)
```

### EvaluationResult

```python
@dataclass
class EvaluationResult:
    success: bool                        # 是否成功
    stage: str                           # 阶段: setup/compile/correctness/performance/complete
    error: Optional[str] = None          # 错误信息
    runtime_ms: Optional[float] = None   # 运行时间
    speedup: Optional[float] = None      # 加速比
    score: Optional[float] = None        # 综合评分
    full_code: Optional[Dict] = None     # 完整代码（用于调试）

    def to_dict(self) -> Dict[str, Any]
```

### 主要函数

```python
def evaluate_solution(
    solution: Dict[str, Any],     # 解决方案字典（6 个组件）
    op_name: str,                 # 算子名称
    python_ref: str,              # Python 参考代码内容
    npu_type: str = "Ascend910B2", # NPU 类型
    project_path: Optional[str],  # 编译产物路径
    fake_mode: bool = False,      # 跳过编译
    save_to: Optional[str] = None # 保存解决方案到目录
) -> EvaluationResult

def normalize_tiling_fields(tiling_fields) -> Optional[List[Dict[str, str]]]
# 将 "uint32_t x;" 格式转换为 [{"type": "uint32_t", "name": "x"}]

def save_solution_files(solution: Dict, output_dir: str) -> None
def load_solution(solution_path: str) -> Optional[Dict]
```

### 权限处理

`evaluate_solution()` 在调用沙箱前会设置 umask：

```python
# msopgen 要求文件不能被 group/others 写入
old_umask = os.umask(0o022)
try:
    result = sandbox.evaluate_sandbox(full_code=full_code, ...)
finally:
    os.umask(old_umask)
```

---

## mcp_server.py - MCP 服务器

**职责**：为 Claude 提供知识库查询工具

### 工具列表

| 工具名 | 描述 | 参数 |
|--------|------|------|
| `cann_search_api` | 搜索 AscendC API | `name`: API 名称 |
| `cann_search_operator` | 查找算子示例代码 | `name`: 算子名称, `top_k`: 返回数量 |
| `cann_get_knowledge` | 获取知识库概览 | 无 |

> **注意**：`cann_evaluate` 工具已移除。CLI 直接调用 evaluator.py 进行评估，
> Claude 不需要也不应该调用评估工具。

### 知识库自动初始化

首次调用 MCP 工具时，会自动：
1. 从 GitHub Release 下载 `repo_data.tar.gz`（算子示例）
2. 扫描 CANN SDK headers（API 定义）
3. 建立索引并缓存到 `~/.cache/evotoolkit/cann_initer/`

```python
# mcp_server.py 中的懒加载逻辑
def get_knowledge_base():
    global _knowledge_base
    if _knowledge_base is None:
        config = KnowledgeBaseConfig()  # 自动检测/下载 repo_data
        _knowledge_base = RealKnowledgeBase(config, auto_build=True)
    return _knowledge_base
```

### 依赖

- `mcp` SDK（可选，不安装则 MCP 功能不可用）
- `evotoolkit.evo_method.cann_initer.knowledge`（可选，不安装则使用 Stub）

### 运行方式

MCP Server 由 CLI 通过 `--mcp-config` 动态配置，无需手动启动。

手动测试：
```bash
python -m cann_claude.mcp_server
```

---

## installer.py - 包路径工具

**职责**：提供包资源路径访问

```python
def get_package_dir() -> Path
    """获取包安装目录"""

def get_mcp_server_path() -> Path
    """获取 MCP Server 脚本路径"""

def get_skill_template_path() -> Path
    """获取 Skill 模板路径"""
```

---

## experience.py - 经验管理

**职责**：跨运行持久化记录错误和优化经验

### 存储位置

```
~/.cache/cann-claude/experience/
├── errors/                 # 错误记录（每个错误一个 JSON 文件）
│   ├── 001_relu_compile.json
│   └── 002_softmax_correctness.json
└── tips/                   # 优化记录（每个优化一个 JSON 文件）
    └── opt_relu_20250115_120000.json
```

同时保存到输出目录 `{output_dir}/experience/`，供 Claude 读取。

### 主要函数

```python
# 路径获取
def get_experience_dir() -> Path           # 获取全局经验目录
def set_output_dir(path: Optional[Path])   # 设置当前输出目录

# 错误记录
def record_error(op_name: str, stage: str, error_msg: str) -> Path
# 返回保存的文件路径

# 优化记录
def record_optimization(op_name: str, before_ms: float, after_ms: float, description: str = "")
```

### 文件格式

**错误记录** (`errors/001_relu_compile.json`):
```json
{
  "id": 1,
  "op": "relu",
  "stage": "compile",
  "error": "error message...",
  "time": "2025-01-15T12:00:00"
}
```

**优化记录** (`tips/opt_relu_20250115_120000.json`):
```json
{
  "op": "relu",
  "before_ms": 0.5,
  "after_ms": 0.3,
  "speedup": 1.67,
  "description": "...",
  "time": "2025-01-15T12:00:00"
}
```

### 使用方式

CLI 自动在每次迭代后记录经验，并在 prompt 中告诉 Claude 经验库路径。Claude 可以用 Read 工具按需查询，也可以用 Write 工具写入 tips。

```python
# CLI 中的调用
from .experience import record_error, record_optimization, set_output_dir

# 设置输出目录
set_output_dir(output_path)

# 失败时记录错误
if not result.success and result.error:
    record_error(op_name, result.stage, result.error)

# 成功且改进时记录优化
if improved:
    record_optimization(op_name, before_ms, after_ms, "description")
```

---

## templates/skill.md - Skill 模板

**职责**：引导 Claude 完成生成工作流

**内容**：
- 工作流说明
- 输出格式规范（6 个组件）
- 代码模板
- 关键约束（常见错误）
- 优化建议

**注入方式**：CLI 在运行时通过 `--settings` 动态注入，无需预先安装。
