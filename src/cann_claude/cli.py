"""
CANN Claude Tools CLI.

Usage:
    cann-claude generate <op_name> <python_ref> [options]
    cann-claude evaluate <solution_path> --op-name <name> --python-ref <file>
"""

import json
import os
import pwd
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
from .experience import (
    get_experience_dir,
    record_error,
    record_optimization,
    set_output_dir,
    sync_tips_to_global,
)
from .installer import get_mcp_server_path, get_package_dir
from .prompts import build_error_fix_prompt, build_initial_prompt, build_optimization_prompt
from .templates import generate_solution_template

# Import signature parser from evotoolkit if available
try:
    from evotoolkit.task.cann_init.signature_parser import OperatorSignatureParser
    HAS_SIGNATURE_PARSER = True
except ImportError:
    HAS_SIGNATURE_PARSER = False

console = Console()

# Default dedicated user for running Claude Code (when running as root)
CANN_USER = "cann-claude"


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


def find_latest_output_dir(op_name: str) -> Path | None:
    """Find the most recent output directory for an operator."""
    # Check common output locations
    search_paths = []

    # If running as root, check cann-claude user's home
    if is_root() and cann_user_exists():
        user_info = pwd.getpwnam(CANN_USER)
        search_paths.append(Path(user_info.pw_dir) / "cann-output")

    # Check current directory's output folder
    search_paths.append(Path.cwd() / "output")

    # Also check if CANN_OUTPUT_DIR env is set
    env_output = os.environ.get("CANN_OUTPUT_DIR")
    if env_output:
        search_paths.append(Path(env_output).parent)

    latest_dir = None
    latest_time = None

    for base_path in search_paths:
        if not base_path.exists():
            continue

        # Find directories matching pattern: {op_name}_{timestamp}
        for d in base_path.iterdir():
            if d.is_dir() and d.name.startswith(f"{op_name}_"):
                # Check if it has iteration_history.json
                history_file = d / "iteration_history.json"
                if history_file.exists():
                    mtime = history_file.stat().st_mtime
                    if latest_time is None or mtime > latest_time:
                        latest_time = mtime
                        latest_dir = d

    return latest_dir


@main.command()
@click.argument("op_name")
@click.argument("python_ref", type=click.Path(exists=True))
@click.option("-o", "--output-dir", type=click.Path(), help="Output directory")
@click.option("-n", "--iterations", default=10, help="Max iterations (default: 10)")
@click.option("-m", "--model", default="sonnet", help="Claude model (default: sonnet)")
@click.option("--npu-type", default="Ascend910B2", help="NPU type (default: Ascend910B2)")
@click.option("--fake-mode", is_flag=True, help="Skip compilation (for testing)")
@click.option("--continue", "continue_run", is_flag=True, help="Continue from latest run")
def generate(op_name: str, python_ref: str, output_dir: str, iterations: int,
             model: str, npu_type: str, fake_mode: bool, continue_run: bool):
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
    if continue_run:
        # Find latest output directory for this operator
        output_path = find_latest_output_dir(op_name)
        if output_path is None:
            console.print(f"[red]Error:[/red] No previous run found for operator '{op_name}'")
            console.print("Run without --continue to start a new session.")
            sys.exit(1)
        console.print(f"[cyan]Continuing from:[/cyan] {output_path}")
    elif output_dir:
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
    python_ref_content = python_ref_path.read_text()
    (output_path / "python_reference.py").write_text(python_ref_content)

    # Parse signature from python reference and save as signature.json
    if HAS_SIGNATURE_PARSER:
        try:
            parser = OperatorSignatureParser()
            signature = parser.parse(python_ref_content, op_name)
            signature_path = output_path / "signature.json"
            signature_path.write_text(json.dumps(signature, indent=2, ensure_ascii=False))
            console.print(f"[dim]Parsed operator signature: {signature_path}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to parse signature: {e}[/yellow]")

    # Generate solution template with correct format (auto-detects vector/cube)
    template = generate_solution_template(op_name, npu_type)
    op_type = template.get("_operator_type", "vector")
    template_path = output_path / "solution_template.json"
    template_path.write_text(json.dumps(template, indent=2, ensure_ascii=False))
    console.print(f"[dim]Generated solution template ({op_type}): {template_path}[/dim]")

    # Copy all reference files (let Claude decide which to read)
    templates_dir = get_package_dir() / "templates"
    ref_files = [
        "constraints.md",  # Format constraints (always needed)
        "vector.md",       # Vector operator guide
        "cube.md",         # Cube operator guide
    ]
    for ref_file in ref_files:
        src = templates_dir / ref_file
        if src.exists():
            (output_path / ref_file).write_text(src.read_text())

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

    # Initialize history
    history = load_history(output_path)

    # Determine starting iteration and session
    if continue_run and history.get("iterations"):
        # Continue from where we left off
        start_iteration = len(history["iterations"]) + 1
        session_id = history.get("config", {}).get("session_id") or str(uuid.uuid4())
        prev_npu_type = history.get("config", {}).get("npu_type")
        if prev_npu_type and prev_npu_type != npu_type:
            console.print(f"[yellow]Warning:[/yellow] NPU type changed from {prev_npu_type} to {npu_type}")
        console.print(f"[dim]Resuming from iteration {start_iteration} (session: {session_id[:8]}...)[/dim]")
    else:
        start_iteration = 1
        session_id = str(uuid.uuid4())

    # Update history config
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
    for iteration in range(start_iteration, iterations + 1):
        console.print(f"\n{'='*60}")
        console.print(f"[bold cyan]ITERATION {iteration}/{iterations}[/bold cyan]")
        console.print(f"{'='*60}\n")

        # Build prompt based on iteration state
        # Experience directory in output (accessible to cann-claude user)
        exp_path = output_path / "experience"

        if iteration == 1 and not continue_run:
            # Initial generation prompt (fresh start)
            prompt = build_initial_prompt(
                op_name=op_name,
                npu_type=npu_type,
                output_path=output_path,
                ref_path=ref_path_for_prompt,
            )
            cmd = [
                "claude", "-p", prompt,
                "--model", model,
                "--dangerously-skip-permissions",
                "--print",
                "--session-id", session_id,
                "--mcp-config", str(mcp_config_file),
            ]
        else:
            # Continuation prompt with feedback (or continue from previous run)
            last_result = history["iterations"][-1] if history["iterations"] else {}

            if last_result.get("success"):
                # Optimization prompt
                prompt = build_optimization_prompt(
                    op_name=op_name,
                    output_path=output_path,
                    exp_path=exp_path,
                    runtime_ms=last_result.get("runtime_ms"),
                    speedup=last_result.get("speedup"),
                )
            else:
                # Fix error prompt
                prompt = build_error_fix_prompt(
                    output_path=output_path,
                    stage=last_result.get("stage", "unknown"),
                    error_msg=last_result.get("error", "Unknown error"),
                    exp_path=exp_path,
                )

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
