"""
CANN Claude Tools CLI.

Usage:
    cann-claude generate <op_name> <python_ref> [options]
    cann-claude evaluate <solution_path> --op-name <name> --python-ref <file>
"""

import json
import os
import pwd
import re
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.markup import escape as rich_escape
from rich.panel import Panel
from rich.prompt import Confirm

from . import __version__, check_evotoolkit
from .config import CANNConfig
from .experience import get_experience_dir, record_error, record_optimization, set_output_dir, sync_tips_to_global
from .installer import get_mcp_server_path, get_package_dir

console = Console()

# Default dedicated user for running Claude Code (when running as root)
CANN_USER = "cann-claude"


def generate_solution_template(op_name: str, npu_type: str = "Ascend910B2") -> dict:
    """Generate a solution template with correct format for the given operator.

    This template contains verified code patterns that Claude should use as a base.
    Claude needs to implement the Compute() function and design appropriate tiling.
    """

    # Convert op_name to class name (e.g., "relu" -> "Relu", "foo_bar" -> "FooBar")
    class_name = "".join(word.capitalize() for word in op_name.split("_"))
    kernel_class = f"Kernel{class_name}"
    tiling_data_class = f"{class_name}CustomTilingData"

    # NPU-specific UB sizes (in KB)
    npu_ub_sizes = {
        "Ascend910B": 256,
        "Ascend910B2": 256,
        "Ascend910B3": 256,
        "Ascend310P": 256,
    }
    ub_size_kb = npu_ub_sizes.get(npu_type, 256)
    ub_safe_kb = ub_size_kb // 4  # Use 1/4 of UB as safe limit

    kernel_impl = f'''using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class {kernel_class} {{
public:
    __aicore__ inline {kernel_class}() {{}}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum) {{
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float*)x + this->blockLength * GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float*)y + this->blockLength * GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
    }}

    __aicore__ inline void Process() {{
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {{
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }}
    }}

private:
    __aicore__ inline void CopyIn(int32_t progress) {{
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }}

    __aicore__ inline void Compute(int32_t progress) {{
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        // TODO: Replace this with your computation logic
        // For ReLU: Relu(yLocal, xLocal, this->tileLength);
        // For Abs:  Abs(yLocal, xLocal, this->tileLength);
        // For Exp:  Exp(yLocal, xLocal, this->tileLength);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }}

    __aicore__ inline void CopyOut(int32_t progress) {{
        LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }}

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<float> xGm, yGm;
    uint32_t blockLength, tileNum, tileLength;
}};'''

    kernel_entry_body = f'''    {kernel_class} op;
    op.Init(x, output, tilingData.totalLength, tilingData.tileNum);
    op.Process();'''

    # Dynamic tiling function that respects UB size
    tiling_func_body = f'''    {tiling_data_class} tiling;

    auto inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {{
        return ge::GRAPH_FAILED;
    }}
    auto shape = inputShape->GetStorageShape();
    uint32_t totalLength = static_cast<uint32_t>(shape.GetShapeSize());

    // ========== DYNAMIC TILING FOR {npu_type} ==========
    // UB safe size: {ub_safe_kb}KB (1/4 of {ub_size_kb}KB total UB)
    constexpr uint32_t UB_SAFE_SIZE = {ub_safe_kb} * 1024;
    constexpr uint32_t BUFFER_NUM = 2;
    constexpr uint32_t NUM_BUFFERS = 2;  // input + output
    constexpr uint32_t BLOCK_DIM = 8;
    uint32_t elementSize = sizeof(float);

    // Calculate max elements per tile that fit in UB
    uint32_t maxTileElements = UB_SAFE_SIZE / (NUM_BUFFERS * BUFFER_NUM * elementSize);
    maxTileElements = (maxTileElements / 8) * 8;  // Align to 32 bytes

    // Calculate tileNum based on data size
    uint32_t blockLength = totalLength / BLOCK_DIM;
    uint32_t tileNum = (blockLength + maxTileElements - 1) / maxTileElements;
    tileNum = tileNum > 0 ? tileNum : 1;
    // ====================================================

    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(tileNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(BLOCK_DIM);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;'''

    infer_shape_body = '''    const gert::Shape* x_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    return ge::GRAPH_SUCCESS;'''

    return {
        "kernel_impl": kernel_impl,
        "kernel_entry_body": kernel_entry_body,
        "tiling_fields": [
            {"type": "uint32_t", "name": "totalLength"},
            {"type": "uint32_t", "name": "tileNum"}
        ],
        "tiling_func_body": tiling_func_body,
        "infer_shape_body": infer_shape_body,
        "output_alloc_code": "at::Tensor result = at::empty_like(x);"
    }

def get_system_prompt(npu_type: str = "Ascend910B2") -> str:
    """Get system prompt from SKILL.md template with NPU type substitution."""
    skill_path = get_package_dir() / "templates" / "skill.md"
    if skill_path.exists():
        content = skill_path.read_text()
        # Skip YAML frontmatter (between --- markers)
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                content = parts[2].strip()
        # Replace NPU type placeholder
        content = content.replace("{CANN_NPU_TYPE}", npu_type)
        return content
    return ""


def is_root() -> bool:
    """Check if running as root."""
    return os.geteuid() == 0


def cann_user_exists() -> bool:
    """Check if the dedicated CANN user exists."""
    try:
        pwd.getpwnam(CANN_USER)
        return True
    except KeyError:
        return False


def create_cann_user() -> bool:
    """Create the dedicated CANN user. Returns True on success."""
    if not is_root():
        console.print("[red]Error:[/red] Must be root to create user")
        return False

    if cann_user_exists():
        return True

    try:
        # Create user with home directory
        subprocess.run(
            ["useradd", "-m", "-s", "/bin/bash", CANN_USER],
            check=True,
            capture_output=True
        )
        console.print(f"[green]✓[/green] Created user '{CANN_USER}'")

        # Get user info
        user_info = pwd.getpwnam(CANN_USER)
        uid, gid = user_info.pw_uid, user_info.pw_gid
        home_dir = Path(user_info.pw_dir)

        # Create .claude directory for the user
        claude_dir = home_dir / ".claude"
        claude_dir.mkdir(parents=True, exist_ok=True)
        os.chown(claude_dir, uid, gid)

        # Copy ANTHROPIC_API_KEY to user's environment if exists
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            bashrc = home_dir / ".bashrc"
            with open(bashrc, "a") as f:
                f.write(f'\nexport ANTHROPIC_API_KEY="{api_key}"\n')
            os.chown(bashrc, uid, gid)
            console.print(f"[green]✓[/green] Configured API key for '{CANN_USER}'")

        # Ensure package directory is accessible to cann-claude user
        ensure_package_accessible()

        return True

    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error creating user:[/red] {rich_escape(e.stderr.decode())}")
        return False
    except Exception as e:
        console.print(f"[red]Error:[/red] {rich_escape(str(e))}")
        return False


def ensure_package_accessible() -> bool:
    """Ensure the package directory is accessible to cann-claude user."""
    package_dir = get_package_dir()

    try:
        # Walk up from package dir to find restricted directories
        path = package_dir
        paths_to_chmod = []

        while path != path.parent:
            if path.name in ('root',) or str(path).startswith('/root'):
                paths_to_chmod.append(path)
            path = path.parent

        # Set execute permission on restricted parent directories
        for p in reversed(paths_to_chmod):
            current_mode = p.stat().st_mode
            if not (current_mode & 0o001):
                os.chmod(p, current_mode | 0o001)

        # Set read+execute permission on package directory recursively
        for item in package_dir.rglob("*"):
            try:
                current_mode = item.stat().st_mode
                if item.is_dir():
                    os.chmod(item, current_mode | 0o005)
                else:
                    os.chmod(item, current_mode | 0o004)
            except (PermissionError, OSError):
                pass

        # Also set permission on package dir itself
        current_mode = package_dir.stat().st_mode
        os.chmod(package_dir, current_mode | 0o005)

        return True
    except Exception:
        return False


def run_as_cann_user(cmd: list, env: dict, cwd: Path) -> int:
    """Run a command as the dedicated CANN user."""
    if not cann_user_exists():
        console.print(f"[red]Error:[/red] User '{CANN_USER}' does not exist.")
        return 1

    # Ensure output directory is accessible
    user_info = pwd.getpwnam(CANN_USER)
    uid, gid = user_info.pw_uid, user_info.pw_gid
    user_home = Path(user_info.pw_dir)

    # Make cwd (output dir) accessible
    os.chown(cwd, uid, gid)
    for item in cwd.rglob("*"):
        try:
            os.chown(item, uid, gid)
        except (PermissionError, OSError):
            pass

    # Prepare environment - keep current env but override HOME
    run_env = env.copy()
    run_env["HOME"] = str(user_home)

    # Use runuser -u to run as cann-claude while preserving environment
    # Use DEVNULL for stdin to prevent blocking in non-interactive mode
    result = subprocess.run(
        ["runuser", "-u", CANN_USER, "--"] + cmd,
        cwd=cwd,
        env=run_env,
        stdin=subprocess.DEVNULL,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    return result.returncode


def run_claude(cmd: list, env: dict, cwd: Path, as_cann_user: bool = False) -> int:
    """Run Claude command, optionally as cann-claude user."""
    if as_cann_user:
        return run_as_cann_user(cmd, env, cwd)
    else:
        try:
            # Use DEVNULL for stdin to prevent blocking in non-interactive mode
            result = subprocess.run(
                cmd, env=env, cwd=cwd,
                stdin=subprocess.DEVNULL, stdout=sys.stdout, stderr=sys.stderr
            )
            return result.returncode
        except FileNotFoundError:
            console.print("[red]Error:[/red] claude command not found.")
            console.print("Please install Claude Code: [cyan]npm install -g @anthropic-ai/claude-code[/cyan]")
            return 1


@click.group()
@click.version_option(__version__)
def main():
    """CANN Ascend C operator generation tools for Claude Code."""
    pass


@main.command()
@click.argument("op_name")
@click.argument("python_ref", type=click.Path(exists=True))
@click.option("-o", "--output-dir", type=click.Path(), help="Output directory")
@click.option("-n", "--iterations", default=10, help="Max iterations (default: 10)")
@click.option("-m", "--model", default="sonnet", help="Claude model (default: sonnet)")
@click.option("--npu-type", default="Ascend910B2", help="NPU type (default: Ascend910B2)")
@click.option("--fake-mode", is_flag=True, help="Skip compilation (for testing)")
def generate(op_name: str, python_ref: str, output_dir: str, iterations: int,
             model: str, npu_type: str, fake_mode: bool):
    """Generate a CANN Ascend C operator with iterative refinement."""
    from .evaluator import evaluate_solution, load_solution
    from .iteration import (
        load_history, save_history, save_iteration_solution,
        save_best_solution,
    )

    # Check evotoolkit
    if not fake_mode and not check_evotoolkit():
        console.print("[red]Error:[/red] evotoolkit not installed.")
        console.print("Run: [cyan]pip install -e ./evotoolkit[cann_init][/cyan]")
        sys.exit(1)

    # Handle root user - create dedicated user if needed
    as_cann_user = False
    if is_root():
        if not cann_user_exists():
            console.print(Panel(
                f"Running as [bold]root[/bold] detected.\n\n"
                f"Claude Code cannot use --dangerously-skip-permissions as root.\n"
                f"A dedicated user '{CANN_USER}' is needed.",
                title="Root User Detected",
                border_style="yellow"
            ))
            if Confirm.ask(f"Create dedicated user '{CANN_USER}'?", default=True):
                if not create_cann_user():
                    console.print("[red]Failed to create user.[/red]")
                    sys.exit(1)
            else:
                console.print("[red]Cannot continue as root without dedicated user.[/red]")
                sys.exit(1)
        as_cann_user = True
        console.print(f"[dim]Running as user '{CANN_USER}'...[/dim]")

    # Resolve paths
    python_ref_path = Path(python_ref).resolve()

    # Determine output directory
    if output_dir:
        output_path = Path(output_dir).resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if is_root() and cann_user_exists():
            user_info = pwd.getpwnam(CANN_USER)
            base_output = Path(user_info.pw_dir) / "cann-output"
            base_output.mkdir(parents=True, exist_ok=True)
            output_path = base_output / f"{op_name}_{timestamp}"
        else:
            output_path = Path.cwd() / "output" / f"{op_name}_{timestamp}"

    output_path.mkdir(parents=True, exist_ok=True)

    # Create config
    config = CANNConfig(
        op_name=op_name,
        output_dir=output_path,
        python_ref_path=python_ref_path,
        max_iterations=iterations,
        npu_type=npu_type,
        fake_mode=fake_mode,
    )

    # Display config
    console.print(Panel(
        f"[bold]Operator:[/bold] {op_name}\n"
        f"[bold]Python Ref:[/bold] {python_ref_path}\n"
        f"[bold]Output:[/bold] {output_path}\n"
        f"[bold]Iterations:[/bold] {iterations}\n"
        f"[bold]Model:[/bold] {model}\n"
        f"[bold]NPU Type:[/bold] {npu_type}\n"
        f"[bold]Fake Mode:[/bold] {fake_mode}",
        title="CANN Operator Generation",
        border_style="blue"
    ))

    # Copy python reference to output
    (output_path / "python_reference.py").write_text(python_ref_path.read_text())

    # Generate solution template with correct format
    template = generate_solution_template(op_name, npu_type)
    template_path = output_path / "solution_template.json"
    template_path.write_text(json.dumps(template, indent=2, ensure_ascii=False))
    console.print(f"[dim]Generated solution template: {template_path}[/dim]")

    # Use copied path for prompts (accessible to cann-claude user)
    ref_path_for_prompt = output_path / "python_reference.py"

    # Set experience output directory
    set_output_dir(output_path)

    # Copy existing experience to output (for Claude to read)
    exp_src = get_experience_dir()
    exp_out = output_path / "experience"
    for subdir in ["errors", "tips"]:
        src_dir = exp_src / subdir
        if src_dir.exists():
            out_dir = exp_out / subdir
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in src_dir.glob("*"):
                (out_dir / f.name).write_text(f.read_text())

    # Get system prompt and save to settings file
    system_prompt = get_system_prompt(npu_type)
    settings_file = None
    if system_prompt:
        settings_file = output_path / ".claude_settings.json"
        settings_file.write_text(json.dumps({
            "systemPromptSuffix": system_prompt
        }, ensure_ascii=False))

    # Create MCP config file
    mcp_config_file = output_path / ".mcp_config.json"
    mcp_config = {
        "mcpServers": {
            "cann-tools": {
                "command": "python3",
                "args": [str(get_mcp_server_path())]
            }
        }
    }
    mcp_config_file.write_text(json.dumps(mcp_config))

    # Set environment variables
    config.set_env()

    # Generate session ID for conversation continuity
    session_id = str(uuid.uuid4())

    # Initialize history
    history = load_history(output_path)
    history["config"] = {
        "op_name": op_name,
        "max_iterations": iterations,
        "npu_type": npu_type,
        "session_id": session_id,
    }

    # Load python reference content
    python_ref_content = python_ref_path.read_text()

    console.print("\n[bold]Starting iterative generation...[/bold]\n")

    # ========== ITERATION LOOP ==========
    for iteration in range(1, iterations + 1):
        console.print(f"\n{'='*60}")
        console.print(f"[bold cyan]ITERATION {iteration}/{iterations}[/bold cyan]")
        console.print(f"{'='*60}\n")

        # Build prompt based on iteration state
        # Experience directory in output (accessible to cann-claude user)
        exp_path = output_path / "experience"

        if iteration == 1:
            # Initial generation prompt - reference the template file
            prompt = f"""Generate an Ascend C operator '{op_name}' for {npu_type}.

## STEP 1: READ THE TEMPLATE

**CRITICAL**: Read the solution template file FIRST:
- `{output_path}/solution_template.json`

This template contains VERIFIED code patterns. Focus on:
1. The `Compute()` function - implement your computation logic here
2. The `tiling_func_body` - contains dynamic tiling that respects UB size limits

## STEP 2: READ PYTHON REFERENCE

Read the Python reference to understand the computation:
- `{ref_path_for_prompt}`

## STEP 3: IMPLEMENT THE OPERATOR

1. **Compute() function**: Replace the `// TODO:` comment with actual computation
   - For ReLU: use `Relu(yLocal, xLocal, this->tileLength);`

2. **Tiling function**: The template includes dynamic tiling for {npu_type}
   - UB size: 256KB, safe limit: 64KB
   - You may customize tiling if needed for your operator

## STEP 4: WRITE solution.json

Write your solution to:
- `{output_path}/solution.json`

## ⚠️ SELF-CHECK BEFORE WRITING

- [ ] kernel_impl: Compute() function implemented correctly
- [ ] kernel_entry_body: Uses `tilingData.xxx` and `output` (not `y`)
- [ ] tiling_func_body: Dynamic tiling respects UB size limits
- [ ] infer_shape_body: Correct shape inference for your operator
- [ ] output_alloc_code: Correct output tensor allocation

## ⚠️ DO NOT:
- Add `#include` to kernel_impl (auto-added!)
- Add `extern "C" __global__` entry function (auto-generated!)
- Use `GET_TILING_DATA` in kernel_entry_body (auto-added!)
- Use hardcoded `tileNum = 8` for large tensors (will cause UB overflow!)

After writing solution.json, I will compile and test it, then give you feedback."""

            cmd = [
                "claude", "-p", prompt,
                "--model", model,
                "--dangerously-skip-permissions",
                "--print",
                "--session-id", session_id,
                "--mcp-config", str(mcp_config_file),
            ]
        else:
            # Continuation prompt with feedback
            last_result = history["iterations"][-1] if history["iterations"] else {}

            if last_result.get("success"):
                # Optimization prompt
                prompt = f"""The solution compiled and passed correctness tests!

Current Performance:
- Runtime: {last_result.get('runtime_ms', 'N/A')} ms
- Speedup: {last_result.get('speedup', 'N/A')}x

## OPTIMIZATION TASK

Try to improve performance. Options (try in order):
1. Increase tileNum: 8 → 16 → 32
2. Increase BUFFER_NUM: 2 → 4
3. Optimize memory access patterns

Write updated solution.json to {output_path}

## REQUIRED: Document your optimization (BE CAREFUL!)

After testing successfully, write to: {exp_path}/tips/opt_{op_name}.md

⚠️ IMPORTANT when writing tips:
1. ONLY document changes that ACTUALLY improved performance
2. Include specific numbers (before/after runtime)
3. VERIFY any code patterns against MANDATORY TEMPLATES
4. If optimization failed or regressed, DO NOT write a tip"""
            else:
                # Fix error prompt - CRITICAL: must guide Claude to research first
                error_msg = last_result.get('error', 'Unknown error')
                stage = last_result.get('stage', 'unknown')

                # Extract problematic APIs from error message
                undeclared = re.findall(r"'(\w+)' was not declared", error_msg)
                no_member = re.findall(r"has no member named '(\w+)'", error_msg)
                problem_apis = undeclared + no_member

                prompt = f"""## BUILD FAILED at stage: {stage}

Error:
```
{error_msg}
```

## FIX WORKFLOW:

1. **READ THE TEMPLATE AGAIN**: `{output_path}/solution_template.json`
   - The template contains VERIFIED code that compiles correctly!

2. **ANALYZE THE ERROR**:
   - Problematic APIs: {problem_apis if problem_apis else 'see error above'}
   - If "UB address out of bounds": tileNum is too small, increase it!

3. **FIX** based on error type:
   - For kernel_impl errors: Check Compute() function and class structure
   - For tiling errors: Ensure dynamic tiling respects UB size limits
   - For "UB out of bounds": Increase tileNum or use dynamic calculation

4. **WRITE** the fixed solution.json

## Common Fixes:

| Error Pattern | Fix |
|---------------|-----|
| `'xxx' was not declared` | Check API name against template |
| `UB address out of bounds` | tileNum too small - use dynamic tiling! |
| `cannot convert 'StorageShape*' to 'Shape*'` | Use template's tiling_func_body pattern |
| `has no member named 'xxx'` | API doesn't exist - check template |

## ⚠️ CRITICAL for UB errors

If you see "UB address out of bounds", your tile size exceeds UB capacity!
Use dynamic tiling from the template:
```cpp
uint32_t maxTileElements = UB_SAFE_SIZE / (NUM_BUFFERS * BUFFER_NUM * elementSize);
uint32_t tileNum = (blockLength + maxTileElements - 1) / maxTileElements;
```"""

            cmd = [
                "claude", "-p", prompt,
                "--model", model,
                "--dangerously-skip-permissions",
                "--print",
                "--resume", session_id,
                "--mcp-config", str(mcp_config_file),
            ]

        # Add settings file if available
        if settings_file:
            cmd.extend(["--settings", str(settings_file)])

        # Run Claude
        console.print("[dim]Running Claude...[/dim]\n")
        console.print("[dim]" + "─" * 60 + "[/dim]")
        returncode = run_claude(cmd, dict(os.environ), output_path, as_cann_user)
        console.print("[dim]" + "─" * 60 + "[/dim]\n")

        if returncode != 0:
            console.print(f"[yellow]Warning:[/yellow] Claude exited with code {returncode}")

        # Load solution
        solution = load_solution(str(output_path))

        if solution is None:
            console.print("[red]Error:[/red] No solution.json found after Claude run")
            # Record failed iteration
            history["iterations"].append({
                "iteration": iteration,
                "success": False,
                "stage": "no_solution",
                "error": "Claude did not generate solution.json",
            })
            save_history(output_path, history)
            continue

        if "_error" in solution:
            console.print(f"[red]Error:[/red] {rich_escape(solution['_error'])}")
            history["iterations"].append({
                "iteration": iteration,
                "success": False,
                "stage": "parse_error",
                "error": solution["_error"],
            })
            save_history(output_path, history)
            continue

        # Show solution summary
        console.print("[green]✓[/green] Solution generated:")
        kernel_lines = len(solution.get("kernel_impl", "").split("\n"))
        tiling_fields = solution.get("tiling_fields", [])
        field_names = [f.get("name", "?") for f in tiling_fields] if isinstance(tiling_fields, list) else []
        console.print(f"  • kernel_impl: {kernel_lines} lines")
        console.print(f"  • tiling_fields: {field_names}")
        console.print()

        # Save iteration solution
        console.print(f"[dim]Saving solution to solution-{iteration}/...[/dim]")
        save_iteration_solution(output_path, solution, iteration)

        # Evaluate solution
        if fake_mode:
            console.print("[dim]Evaluating solution (fake mode - skipping compilation)...[/dim]\n")
        else:
            console.print("[dim]Evaluating solution (compile → correctness → performance)...[/dim]\n")

        result = evaluate_solution(
            solution=solution,
            op_name=op_name,
            python_ref=python_ref_content,
            npu_type=npu_type,
            project_path=str(output_path / f"solution-{iteration}" / "project"),
            fake_mode=fake_mode,
        )

        # Record iteration
        iteration_record = {
            "iteration": iteration,
            "success": result.success,
            "stage": result.stage,
            "error": result.error,
            "runtime_ms": result.runtime_ms,
            "speedup": result.speedup,
            "score": result.score,
            "solution_dir": f"solution-{iteration}",
        }
        history["iterations"].append(iteration_record)

        # Record experience
        if not result.success and result.error:
            # Record error pattern
            record_error(op_name, result.stage, result.error)

        # Update summary
        history["summary"]["total"] = iteration
        if result.success:
            history["summary"]["successful"] = history["summary"].get("successful", 0) + 1

            # Check if this is the best (and record optimization if improved)
            current_best_score = history["summary"].get("best_score", -1)
            prev_best_runtime = history["summary"].get("best_runtime_ms")

            if (result.score or 0) > current_best_score:
                # Record optimization if we improved from a previous best
                if prev_best_runtime and result.runtime_ms and result.runtime_ms < prev_best_runtime:
                    record_optimization(
                        op_name=op_name,
                        before_ms=prev_best_runtime,
                        after_ms=result.runtime_ms,
                        description=f"Iteration {iteration} improved from {prev_best_runtime:.4f}ms to {result.runtime_ms:.4f}ms",
                    )

                history["summary"]["best_iteration"] = iteration
                history["summary"]["best_runtime_ms"] = result.runtime_ms
                history["summary"]["best_speedup"] = result.speedup
                history["summary"]["best_score"] = result.score
                iteration_record["is_best"] = True

                # Save best solution
                save_best_solution(output_path, solution, iteration)
                console.print("[green]★ New best solution![/green]")

        # Save history after each iteration
        save_history(output_path, history)

        # Sync tips to global cache (for future runs)
        sync_tips_to_global()

        # Display result
        if result.success:
            console.print(Panel(
                f"[green]✓ PASSED[/green]\n\n"
                f"Runtime:  {result.runtime_ms:.4f} ms\n"
                f"Speedup:  {result.speedup:.2f}x\n"
                f"Score:    {result.score:.4f}" if result.score else "",
                title=f"Iteration {iteration} Result",
                border_style="green"
            ))
        else:
            console.print(Panel(
                f"[red]✗ FAILED[/red]\n\n"
                f"Stage: {result.stage}\n"
                f"Error: {rich_escape(result.error or '')}",
                title=f"Iteration {iteration} Result",
                border_style="red"
            ))

    # ========== FINAL SUMMARY ==========
    console.print("\n" + "=" * 60)

    summary = history.get("summary", {})
    if summary.get("best_iteration"):
        console.print(Panel(
            f"[bold]Total Iterations:[/bold] {summary.get('total', 'N/A')}\n"
            f"[bold]Successful:[/bold] {summary.get('successful', 0)}\n"
            f"[bold]Best Iteration:[/bold] {summary['best_iteration']}\n"
            f"[bold]Best Runtime:[/bold] {summary.get('best_runtime_ms', 'N/A')} ms\n"
            f"[bold]Best Speedup:[/bold] {summary.get('best_speedup', 'N/A')}x\n"
            f"[bold]Best Score:[/bold] {summary.get('best_score', 'N/A')}\n\n"
            f"Best solution saved to: {output_path}/best_solution/",
            title="[green]Generation Complete[/green]",
            border_style="green"
        ))
    else:
        console.print(Panel(
            f"[bold]Total Iterations:[/bold] {summary.get('total', 'N/A')}\n"
            f"[bold]Successful:[/bold] 0\n\n"
            f"All iterations failed. Check the errors above and try again.",
            title="[red]Generation Failed[/red]",
            border_style="red"
        ))

    console.print(f"\nOutput directory: {output_path}")
    console.print("=" * 60)


@main.command()
@click.argument("solution_path", type=click.Path(exists=True))
@click.option("--op-name", required=True, help="Operator name")
@click.option("--python-ref", required=True, type=click.Path(exists=True), help="Python reference file")
@click.option("--npu-type", default="Ascend910B2", help="NPU type (default: Ascend910B2)")
@click.option("--fake-mode", is_flag=True, help="Skip actual compilation")
@click.option("-o", "--output-dir", type=click.Path(), help="Output directory for evaluation artifacts")
def evaluate(solution_path: str, op_name: str, python_ref: str, npu_type: str,
             fake_mode: bool, output_dir: str):
    """Evaluate a CANN solution.

    SOLUTION_PATH: Path to solution.json or directory containing it.

    Example:
        cann-claude evaluate ./solution.json --op-name relu --python-ref ./relu.py
    """
    from .evaluator import evaluate_solution, load_solution

    # Load solution
    solution = load_solution(solution_path)
    if solution is None:
        console.print(f"[red]Error:[/red] Solution not found at: {solution_path}")
        sys.exit(1)

    # Load python reference
    python_ref_path = Path(python_ref).resolve()
    python_ref_content = python_ref_path.read_text()

    # Determine output directory
    if output_dir:
        project_path = Path(output_dir).resolve() / "project"
    else:
        project_path = Path(solution_path).parent / "project"

    project_path.mkdir(parents=True, exist_ok=True)

    console.print(Panel(
        f"[bold]Solution:[/bold] {solution_path}\n"
        f"[bold]Operator:[/bold] {op_name}\n"
        f"[bold]Python Ref:[/bold] {python_ref_path}\n"
        f"[bold]NPU Type:[/bold] {npu_type}\n"
        f"[bold]Fake Mode:[/bold] {fake_mode}",
        title="CANN Solution Evaluation",
        border_style="blue"
    ))

    console.print("\n[bold]Evaluating...[/bold]\n")

    # Evaluate
    result = evaluate_solution(
        solution=solution,
        op_name=op_name,
        python_ref=python_ref_content,
        npu_type=npu_type,
        project_path=str(project_path),
        fake_mode=fake_mode,
    )

    # Show result
    if result.success:
        console.print(Panel(
            f"[bold]Stage:[/bold] {result.stage}\n"
            f"[bold]Runtime:[/bold] {result.runtime_ms:.4f} ms\n"
            f"[bold]Speedup:[/bold] {result.speedup:.2f}x\n"
            f"[bold]Score:[/bold] {result.score:.4f}" if result.score else "",
            title="[green]Evaluation Passed[/green]",
            border_style="green"
        ))
    else:
        console.print(Panel(
            f"[bold]Stage:[/bold] {result.stage}\n"
            f"[bold]Error:[/bold] {rich_escape(result.error or '')}",
            title="[red]Evaluation Failed[/red]",
            border_style="red"
        ))
        sys.exit(1)


if __name__ == "__main__":
    main()
