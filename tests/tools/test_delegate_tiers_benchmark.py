#!/usr/bin/env python3
"""
Real-world delegation tier benchmark runner.

Spawns actual subagents via delegate_task with different tiers,
measures tokens, latency, and output quality.

Prerequisites:
  - tmux installed
  - hermes venv at repo root
  - API keys configured (uses openai-codex provider)

Usage:
  cd /home/ubuntu/hermes-agent-dev/delegate-tiers
  source .venv/bin/activate
  python tests/tools/test_delegate_tiers_benchmark.py

Or run specific benchmark:
  python tests/tools/test_delegate_tiers_benchmark.py --tier light --tier heavy
"""

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Benchmark task definitions — designed to test different capability tiers
BENCHMARK_TASKS = {
    # --- LIGHT tier tasks (simple, deterministic) ---
    "list_files": {
        "tier": "light",
        "goal": "List all Python files in the current directory (non-recursive). Just list the filenames.",
        "expected_tools": ["terminal"],
        "max_iterations": 5,
        "description": "Simple file listing — should complete in 1-2 tool calls",
        "quality_check": lambda r: len(r) > 0 and ".py" in r,
    },
    "read_and_summarize": {
        "tier": "light",
        "goal": "Read the file pyproject.toml and summarize what this project is in one sentence.",
        "expected_tools": ["file"],
        "max_iterations": 5,
        "description": "Read one file, summarize — straightforward extraction",
        "quality_check": lambda r: "hermes" in r.lower() or "agent" in r.lower(),
    },
    "count_lines": {
        "tier": "light",
        "goal": "Count the total number of lines in tools/delegate_tool.py and report just the number.",
        "expected_tools": ["terminal", "file"],
        "max_iterations": 5,
        "description": "Count lines in one file — trivial task",
        "quality_check": lambda r: any(c.isdigit() for c in r),
    },

    # --- HEAVY tier tasks (coding, debugging) ---
    "analyze_structure": {
        "tier": "heavy",
        "goal": (
            "Analyze the structure of tools/delegate_tool.py. List all function names, "
            "their line numbers, and a one-line description of what each does. "
            "Format as a numbered list."
        ),
        "expected_tools": ["file"],
        "max_iterations": 15,
        "description": "Code analysis — requires reading and understanding structure",
        "quality_check": lambda r: "delegate_task" in r and "_build_child_agent" in r,
    },
    "trace_execution": {
        "tier": "heavy",
        "goal": (
            "Trace the execution path when delegate_task is called with a single goal. "
            "Start from the registry.register() handler and follow every function call "
            "until the result is returned. List each function in order with its purpose."
        ),
        "expected_tools": ["file"],
        "max_iterations": 20,
        "description": "Execution tracing — requires reading multiple functions and understanding flow",
        "quality_check": lambda r: "_build_child_agent" in r and "_run_single_child" in r,
    },

    # --- REVIEW tier tasks (deep analysis, judgment) ---
    "security_review": {
        "tier": "review",
        "goal": (
            "Review tools/delegate_tool.py for potential security issues. "
            "Specifically check: 1) Can a malicious subagent escape its sandbox? "
            "2) Are credentials properly isolated? 3) Is there any way to bypass "
            "the blocked tools list? Report findings with specific line numbers."
        ),
        "expected_tools": ["file"],
        "max_iterations": 25,
        "description": "Security audit — requires deep understanding and judgment",
        "quality_check": lambda r: len(r) > 200,  # substantial analysis expected
    },
    "test_coverage_review": {
        "tier": "review",
        "goal": (
            "Review tests/tools/test_delegate.py and identify gaps. "
            "What edge cases are NOT covered? Focus on: error handling, "
            "concurrent execution, credential failures, interrupt propagation. "
            "List specific missing test cases."
        ),
        "expected_tools": ["file"],
        "max_iterations": 25,
        "description": "Test gap analysis — requires reviewing existing tests and reasoning about missing coverage",
        "quality_check": lambda r: len(r) > 200,
    },

    # --- RESEARCH tier tasks (multi-source synthesis) ---
    "compare_implementations": {
        "tier": "research",
        "goal": (
            "Research how delegate_task's threading model compares to asyncio-based "
            "approaches in other agent frameworks. Read tools/delegate_tool.py to "
            "understand the current ThreadPoolExecutor approach. Then search the web "
            "for 'python asyncio subagent delegation' patterns. Summarize the trade-offs "
            "of threads vs asyncio for agent delegation in a comparison table."
        ),
        "expected_tools": ["file", "web"],
        "max_iterations": 30,
        "description": "Research + analysis — combines code reading with web research",
        "quality_check": lambda r: "thread" in r.lower() and "asyncio" in r.lower(),
    },
}

# Default tier config used by this benchmark harness so it can run independently
# of the user's persistent ~/.hermes/config.yaml.
BENCHMARK_TIER_CONFIG = {
    "model": "gpt-5.4-mini",
    "provider": "openai-codex",
    "reasoning_effort": "low",
    "max_iterations": 25,
    "default_tier": "heavy",
    "tiers": {
        "light": {
            "model": "gpt-5.4-mini",
            "provider": "openai-codex",
            "reasoning_effort": "low",
            "max_iterations": 8,
        },
        "heavy": {
            "model": "gpt-5.4",
            "provider": "openai-codex",
            "reasoning_effort": "medium",
            "max_iterations": 14,
        },
        "review": {
            "model": "gpt-5.4",
            "provider": "openai-codex",
            "reasoning_effort": "xhigh",
            "max_iterations": 18,
        },
        "planning": {
            "model": "xiaomi/mimo-v2-pro",
            "provider": "nous",
            "reasoning_effort": "high",
            "max_iterations": 18,
        },
        "research": {
            "model": "gpt-5.4",
            "provider": "openai-codex",
            "reasoning_effort": "high",
            "max_iterations": 18,
        },
    },
}


def run_benchmark_task(task_name: str, task_def: dict, verbose: bool = False) -> dict:
    """Run a single benchmark task via delegate_task and collect metrics.

    Uses a real parent AIAgent and patches delegate_tool._load_config with the
    benchmark tier config so runs are reproducible regardless of the user's
    persistent ~/.hermes/config.yaml.
    """
    from unittest.mock import patch
    from run_agent import AIAgent
    from tools.delegate_tool import delegate_task

    # Make workspace routing deterministic for child prompts.
    os.environ["TERMINAL_CWD"] = str(REPO_ROOT)

    parent = AIAgent(
        model="gpt-5.4",
        provider="openai-codex",
        quiet_mode=True,
        enabled_toolsets=["file", "terminal", "web", "delegation"],
        skip_memory=True,
        skip_context_files=True,
    )
    parent.cwd = str(REPO_ROOT)

    tier = task_def["tier"]
    goal = task_def["goal"]
    context = f"Repository root: {REPO_ROOT}. Use only files inside this exact path unless the task explicitly requires web research."

    if verbose:
        print(f"\n  [TASK] {task_name}")
        print(f"  [TIER] {tier}")
        print(f"  [GOAL] {goal[:80]}...")

    start = time.monotonic()
    try:
        with patch("tools.delegate_tool._load_config", return_value=BENCHMARK_TIER_CONFIG):
            result_json = delegate_task(
                goal=goal,
                context=context,
                tier=tier,
                max_iterations=task_def.get("max_iterations", 15),
                toolsets=task_def.get("expected_tools"),
                parent_agent=parent,
            )
        elapsed = round(time.monotonic() - start, 2)
        result = json.loads(result_json)

        if result.get("results") and len(result["results"]) > 0:
            entry = result["results"][0]
            summary = entry.get("summary", "") or ""
            status = entry.get("status", "unknown")
            api_calls = entry.get("api_calls", 0)
            duration = entry.get("duration_seconds", elapsed)
            tokens_in = entry.get("tokens", {}).get("input", 0)
            tokens_out = entry.get("tokens", {}).get("output", 0)
            model_used = entry.get("model", "unknown")
            tool_trace = entry.get("tool_trace", [])
            exit_reason = entry.get("exit_reason", "unknown")

            quality_pass = task_def.get("quality_check", lambda r: True)(summary)

            return {
                "task": task_name,
                "tier": tier,
                "status": status,
                "exit_reason": exit_reason,
                "duration_seconds": duration,
                "api_calls": api_calls,
                "tokens_input": tokens_in,
                "tokens_output": tokens_out,
                "model": model_used,
                "summary_length": len(summary),
                "tool_count": len(tool_trace),
                "tools_used": [t.get("tool", "") for t in tool_trace],
                "quality_pass": quality_pass,
                "description": task_def["description"],
                "error": entry.get("error"),
                "summary_preview": summary[:200],
            }
        else:
            return {
                "task": task_name,
                "tier": tier,
                "status": "error",
                "error": result.get("error", "No results"),
                "duration_seconds": elapsed,
            }
    except Exception as e:
        elapsed = round(time.monotonic() - start, 2)
        return {
            "task": task_name,
            "tier": tier,
            "status": "error",
            "error": str(e),
            "duration_seconds": elapsed,
        }


def print_results_table(results: list):
    """Print a formatted results table."""
    print(f"\n{'='*100}")
    print(f"{'Task':<25} {'Tier':<10} {'Status':<10} {'Duration':<10} {'Tokens':<15} {'Quality':<8} {'Model'}")
    print(f"{'-'*100}")

    for r in results:
        status_icon = "OK" if r["status"] == "completed" else "!!"
        quality_icon = "PASS" if r.get("quality_pass") else "FAIL"
        tokens = f"{r.get('tokens_input',0)}+{r.get('tokens_output',0)}"
        model = r.get("model", "?").split("/")[-1] if r.get("model") else "?"

        print(f"{r['task']:<25} {r['tier']:<10} {status_icon:<10} "
              f"{r.get('duration_seconds',0):<10.1f} {tokens:<15} "
              f"{quality_icon:<8} {model}")


def print_tier_comparison(results: list):
    """Print comparative analysis across tiers."""
    by_tier = {}
    for r in results:
        tier = r.get("tier", "unknown")
        if tier not in by_tier:
            by_tier[tier] = []
        by_tier[tier].append(r)

    print(f"\n{'='*80}")
    print(" TIER COMPARISON")
    print(f"{'='*80}")

    for tier in ["light", "heavy", "review", "research"]:
        if tier not in by_tier:
            continue
        tasks = by_tier[tier]
        durations = [t.get("duration_seconds", 0) for t in tasks if t.get("status") == "completed"]
        tokens_in = [t.get("tokens_input", 0) for t in tasks if t.get("status") == "completed"]
        tokens_out = [t.get("tokens_output", 0) for t in tasks if t.get("status") == "completed"]
        quality = [t.get("quality_pass", False) for t in tasks if t.get("status") == "completed"]

        print(f"\n  [{tier.upper()}] ({len(tasks)} tasks)")
        if durations:
            print(f"    Duration:  avg={statistics.mean(durations):.1f}s  "
                  f"min={min(durations):.1f}s  max={max(durations):.1f}s")
        if tokens_in:
            print(f"    Tokens in:  avg={statistics.mean(tokens_in):.0f}  "
                  f"total={sum(tokens_in)}")
        if tokens_out:
            print(f"    Tokens out: avg={statistics.mean(tokens_out):.0f}  "
                  f"total={sum(tokens_out)}")
        if quality:
            print(f"    Quality:   {sum(quality)}/{len(quality)} passed "
                  f"({100*sum(quality)/len(quality):.0f}%)")

        # Cost estimate (using GPT-5.4 pricing as reference)
        total_in = sum(tokens_in) if tokens_in else 0
        total_out = sum(tokens_out) if tokens_out else 0
        if tier == "light":
            cost = (total_in * 0.75 + total_out * 4.50) / 1_000_000
        elif tier == "heavy":
            cost = (total_in * 2.50 + total_out * 15.00) / 1_000_000
        elif tier in ("review", "research"):
            cost = (total_in * 2.50 + total_out * 15.00) / 1_000_000
        else:
            cost = 0
        print(f"    Est. cost: ${cost:.4f}")


def save_results(results: list, output_dir: Path):
    """Save results to JSON for analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"benchmark_{timestamp}.json"

    data = {
        "timestamp": timestamp,
        "repo": str(REPO_ROOT),
        "results": results,
        "summary": {},
    }

    by_tier = {}
    for r in results:
        tier = r.get("tier", "unknown")
        if tier not in by_tier:
            by_tier[tier] = {"count": 0, "completed": 0, "total_duration": 0, "total_tokens_in": 0, "total_tokens_out": 0}
        by_tier[tier]["count"] += 1
        if r.get("status") == "completed":
            by_tier[tier]["completed"] += 1
            by_tier[tier]["total_duration"] += r.get("duration_seconds", 0)
            by_tier[tier]["total_tokens_in"] += r.get("tokens_input", 0)
            by_tier[tier]["total_tokens_out"] += r.get("tokens_output", 0)

    data["summary"] = by_tier

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to: {output_file}")
    return str(output_file)


def main():
    parser = argparse.ArgumentParser(description="Delegation tier benchmark runner")
    parser.add_argument("--tier", action="append", help="Run only specific tier(s)")
    parser.add_argument("--task", action="append", help="Run only specific task(s)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", default=str(REPO_ROOT / "tests" / "benchmark_results"),
                        help="Output directory for results")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per task (for variance)")
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(" DELEGATION TIER BENCHMARK RUNNER")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

    # Filter tasks
    tasks_to_run = {}
    for name, task in BENCHMARK_TASKS.items():
        if args.tier and task["tier"] not in args.tier:
            continue
        if args.task and name not in args.task:
            continue
        tasks_to_run[name] = task

    print(f"\n  Tasks to run: {len(tasks_to_run)}")
    for name, task in tasks_to_run.items():
        print(f"    [{task['tier']:<8}] {name}: {task['description']}")

    # Check for real provider auth. In this environment openai-codex often comes
    # from persisted auth.json/session state rather than OPENAI_API_KEY env vars.
    auth_status = {}
    for provider_name in ("openai-codex", "openrouter"):
        try:
            from hermes_cli.runtime_provider import resolve_runtime_provider
            runtime = resolve_runtime_provider(requested=provider_name)
            auth_status[provider_name] = bool(runtime.get("api_key"))
        except Exception:
            auth_status[provider_name] = False

    if not any(auth_status.values()):
        print("\n  WARNING: No usable provider auth detected for openai-codex/openrouter.")
        print("  Tests will run with mocked API calls (structural verification only).")
        print("  Configure auth (e.g. persistent openai-codex login) for real performance data.\n")
        use_mocks = True
    else:
        print(f"\n  Real provider auth detected: {', '.join(k for k,v in auth_status.items() if v)}\n")
        use_mocks = False

    # Run benchmarks
    all_results = []
    for run_num in range(args.runs):
        if args.runs > 1:
            print(f"\n--- Run {run_num + 1}/{args.runs} ---")

        for task_name, task_def in tasks_to_run.items():
            if use_mocks:
                # Mock mode: verify structure without API calls
                from unittest.mock import patch, MagicMock
                import threading

                mock_parent = MagicMock()
                mock_parent.base_url = "https://openrouter.ai/api/v1"
                mock_parent.api_key = "***"
                mock_parent.provider = "openai-codex"
                mock_parent.api_mode = "chat_completions"
                mock_parent.model = "anthropic/claude-sonnet-4"
                mock_parent.platform = "cli"
                mock_parent.providers_allowed = None
                mock_parent.providers_ignored = None
                mock_parent.providers_order = None
                mock_parent.provider_sort = None
                mock_parent._session_db = None
                mock_parent._delegate_depth = 0
                mock_parent._active_children = []
                mock_parent._active_children_lock = threading.Lock()
                mock_parent._print_fn = None
                mock_parent.tool_progress_callback = None
                mock_parent.thinking_callback = None

                from tools.delegate_tool import resolve_tier_config, _load_config

                # Verify tier resolution works
                with patch("tools.delegate_tool._load_config") as mock_cfg:
                    mock_cfg.return_value = {
                        "model": "gpt-5.4-mini",
                        "reasoning_effort": "low",
                        "tiers": {
                            "light": {"model": "gpt-5.4-mini", "reasoning_effort": "low", "max_iterations": 25},
                            "heavy": {"model": "gpt-5.4", "reasoning_effort": "medium", "max_iterations": 50},
                            "review": {"model": "gpt-5.4", "reasoning_effort": "xhigh", "max_iterations": 60},
                            "research": {"model": "gpt-5.4", "reasoning_effort": "high", "max_iterations": 60},
                        },
                    }
                    cfg = mock_cfg()
                    resolved = resolve_tier_config(cfg, tier=task_def["tier"])

                    all_results.append({
                        "task": task_name,
                        "tier": task_def["tier"],
                        "status": "verified",
                        "duration_seconds": 0,
                        "model": resolved.get("model", "?"),
                        "reasoning_effort": resolved.get("reasoning_effort", "?"),
                        "max_iterations": resolved.get("max_iterations", 0),
                        "description": task_def["description"],
                        "quality_pass": True,
                    })
                    print(f"  [VERIFY] {task_name} [{task_def['tier']}]: "
                          f"model={resolved.get('model')}, "
                          f"reasoning={resolved.get('reasoning_effort')}, "
                          f"iters={resolved.get('max_iterations')}")
            else:
                result = run_benchmark_task(task_name, task_def, verbose=args.verbose)
                all_results.append(result)
                status = result["status"]
                dur = result.get("duration_seconds", 0)
                print(f"  [{status.upper()}] {task_name} [{task_def['tier']}]: {dur:.1f}s")

    # Print results
    if not use_mocks:
        print_results_table(all_results)
        print_tier_comparison(all_results)
    else:
        print(f"\n{'='*80}")
        print(" MOCK VERIFICATION RESULTS")
        print(f"{'='*80}")
        for tier in ["light", "heavy", "review", "research"]:
            tier_results = [r for r in all_results if r["tier"] == tier]
            if tier_results:
                print(f"\n  [{tier.upper()}]")
                for r in tier_results:
                    print(f"    {r['task']}: model={r['model']}, "
                          f"reasoning={r['reasoning_effort']}, "
                          f"iters={r['max_iterations']}")

    # Save
    output_file = save_results(all_results, Path(args.output))

    # Summary
    completed = [r for r in all_results if r.get("status") in ("completed", "verified")]
    print(f"\n{'='*80}")
    print(f" SUMMARY: {len(completed)}/{len(all_results)} tasks verified/completed")
    print(f"{'='*80}\n")

    return 0 if len(completed) == len(all_results) else 1


if __name__ == "__main__":
    sys.exit(main())
