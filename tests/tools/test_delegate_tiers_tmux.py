#!/usr/bin/env python3
"""
Tmux-based delegation tier integration tests.

These are real tmux-driven reproductions of the tier logic. They do not rely on
plain in-process execution of the outer harness; each case is written to a temp
Python script, executed inside a dedicated tmux session, logged to a temp file,
and then the outer runner validates the emitted markers.

Run:
  source .venv/bin/activate
  python tests/tools/test_delegate_tiers_tmux.py
"""

import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def run(cmd, timeout=30):
    r = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO_ROOT),
    )
    return r.stdout.strip(), r.stderr.strip(), r.returncode


def tmux_kill(session):
    run(f"tmux kill-session -t {session} 2>/dev/null || true")


def tmux_run(session, script_content, timeout=30):
    """Run a Python script inside tmux and return the full logged output.

    Important: capturing a pane after the command exits is racy because the tmux
    session often disappears immediately. So we redirect output to a temp log,
    append EXIT_CODE, keep the shell alive with ``exec bash``, and poll the log.
    """
    tmux_kill(session)

    script_fd, script_path = tempfile.mkstemp(suffix=".py", prefix=f"tmux_test_{session}_")
    os.close(script_fd)
    log_fd, log_path = tempfile.mkstemp(suffix=".log", prefix=f"tmux_test_{session}_")
    os.close(log_fd)

    with open(script_path, "w") as f:
        f.write(script_content)

    venv_python = str(REPO_ROOT / ".venv" / "bin" / "python")
    tmux_cmd = (
        f"tmux new-session -d -s {session} \"bash -lc 'cd {REPO_ROOT} && "
        f"{venv_python} {script_path} > {log_path} 2>&1; code=$?; "
        f"echo EXIT_CODE=$code >> {log_path}; exec bash'\""
    )
    run(tmux_cmd)

    output = ""
    started = time.time()
    while time.time() - started < timeout:
        time.sleep(1)
        if os.path.exists(log_path):
            with open(log_path) as f:
                output = f.read()
            if "EXIT_CODE=" in output:
                break

    if not output:
        pane_out, pane_err, _ = run(f"tmux capture-pane -t {session} -p -S -200")
        output = pane_out or pane_err or ""

    tmux_kill(session)
    for path in (script_path, log_path):
        try:
            os.unlink(path)
        except Exception:
            pass

    return output


SCRIPT_TIER_RESOLUTION = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, "{repo}")
    from tools.delegate_tool import resolve_tier_config, SUPPORTED_TIERS, _REASONING_ORDER

    cfg = {
        "model": "gpt-5.4-mini",
        "provider": "openai-codex",
        "reasoning_effort": "low",
        "max_iterations": 25,
        "default_tier": "heavy",
        "tiers": {
            "light": {"model": "gpt-5.4-mini", "reasoning_effort": "low", "max_iterations": 25},
            "heavy": {"model": "gpt-5.4", "reasoning_effort": "medium", "max_iterations": 50},
            "review": {"model": "gpt-5.4", "reasoning_effort": "xhigh", "max_iterations": 60},
            "planning": {"model": "xiaomi/mimo-v2-pro", "provider": "nous", "reasoning_effort": "high", "max_iterations": 60},
            "research": {"model": "gpt-5.4", "reasoning_effort": "high", "max_iterations": 60},
        },
    }

    print("TIER_RESOLUTION_TEST_START")
    for tier_name in sorted(SUPPORTED_TIERS):
        result = resolve_tier_config(cfg, tier=tier_name)
        print(f"  {tier_name}: model={result['model']}, reasoning={result.get('reasoning_effort')}, iters={result.get('max_iterations')}")
        assert "tiers" not in result
        assert "default_tier" not in result

    default = resolve_tier_config(cfg)
    assert default["model"] == "gpt-5.4"
    print(f"  default_tier: model={default['model']}")

    floor_test = {"tiers": {"review": {"reasoning_effort": "low"}}}
    floor_result = resolve_tier_config(floor_test, tier="review")
    assert floor_result["reasoning_effort"] == "high"
    print("  floor guardrail: review low -> high OK")

    costs = {}
    for t in SUPPORTED_TIERS:
        r = resolve_tier_config(cfg, tier=t)
        costs[t] = _REASONING_ORDER.get(r.get("reasoning_effort", "none"), 0) * r.get("max_iterations", 0)

    assert costs["light"] < costs["heavy"]
    assert costs["heavy"] < costs["review"]
    print(f"  cost order: light({costs['light']}) < heavy({costs['heavy']}) < review({costs['review']}) OK")
    print("TIER_RESOLUTION_TEST_PASS")
    """
)


SCRIPT_BATCH_PER_TASK = textwrap.dedent(
    """
    import json
    import sys
    import threading
    from unittest.mock import MagicMock, patch

    sys.path.insert(0, "{repo}")
    from tools.delegate_tool import delegate_task

    tier_cfg = {
        "model": "gpt-5.4-mini",
        "reasoning_effort": "low",
        "tiers": {
            "light": {"model": "gpt-5.4-mini", "reasoning_effort": "low", "max_iterations": 25},
            "review": {"model": "gpt-5.4", "reasoning_effort": "xhigh", "max_iterations": 60},
        },
    }

    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "***"
    parent.provider = "openai-codex"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None

    children_configs = []

    def capture_child(**kwargs):
        children_configs.append(kwargs)
        child = MagicMock()
        child.run_conversation.return_value = {
            "final_response": "done",
            "completed": True,
            "messages": [],
        }
        return child

    print("BATCH_PER_TASK_TEST_START")
    with patch("tools.delegate_tool._load_config", return_value=tier_cfg):
        with patch("tools.delegate_tool._build_child_agent", side_effect=capture_child):
            result_json = delegate_task(
                tasks=[
                    {"goal": "Quick lookup", "tier": "light"},
                    {"goal": "Deep review", "tier": "review"},
                ],
                parent_agent=parent,
            )
            result = json.loads(result_json)

    assert children_configs[0]["model"] == "gpt-5.4-mini"
    assert children_configs[0]["override_reasoning_effort"] == "low"
    assert children_configs[0]["max_iterations"] == 25
    print(f"  light: model={children_configs[0]['model']}, reasoning={children_configs[0]['override_reasoning_effort']}, iters={children_configs[0]['max_iterations']}")

    assert children_configs[1]["model"] == "gpt-5.4"
    assert children_configs[1]["override_reasoning_effort"] == "xhigh"
    assert children_configs[1]["max_iterations"] == 60
    print(f"  review: model={children_configs[1]['model']}, reasoning={children_configs[1]['override_reasoning_effort']}, iters={children_configs[1]['max_iterations']}")

    assert len(result["results"]) == 2
    print(f"  batch results: {len(result['results'])} tasks completed")
    print("BATCH_PER_TASK_TEST_PASS")
    """
)


SCRIPT_POOL_VALIDATION = textwrap.dedent(
    """
    import sys
    import threading
    from unittest.mock import MagicMock, patch

    sys.path.insert(0, "{repo}")
    from tools.delegate_tool import delegate_task, _validate_pool_model, _build_pool_description

    print("POOL_VALIDATION_TEST_START")
    pool = [
        {"model": "gpt-5.4", "provider": "openai-codex", "strengths": "coding"},
        {"model": "gpt-5.4-mini", "strengths": "quick"},
    ]

    assert _validate_pool_model("gpt-5.4", pool) == "gpt-5.4"
    assert _validate_pool_model("fake-model", pool) == "gpt-5.4"
    assert _validate_pool_model(None, pool) is None
    assert _validate_pool_model("anything", []) == "anything"
    print("  pool validation helper: OK")

    desc = _build_pool_description(pool)
    assert "gpt-5.4" in desc and "openai-codex" in desc
    print(f"  pool description: {desc[:60]}...")

    pool_cfg = {
        "model": "fake-model",
        "pool": [
            {"model": "gpt-5.4", "strengths": "coding"},
            {"model": "gpt-5.4-mini", "strengths": "quick"},
        ],
    }

    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "***"
    parent.provider = "openai-codex"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None

    with patch("tools.delegate_tool._load_config", return_value=pool_cfg):
        with patch("tools.delegate_tool._build_child_agent") as mock_build:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "done",
                "completed": True,
                "messages": [],
            }
            mock_build.return_value = mock_child
            delegate_task(goal="test", parent_agent=parent)
            call_kwargs = mock_build.call_args[1]

    assert call_kwargs["model"] == "gpt-5.4"
    print("  pool fallback in delegate_task: fake-model -> gpt-5.4 OK")
    print("POOL_VALIDATION_TEST_PASS")
    """
)


SCRIPT_BACKWARD_COMPAT = textwrap.dedent(
    """
    import json
    import sys
    import threading
    from unittest.mock import MagicMock, patch

    sys.path.insert(0, "{repo}")
    from tools.delegate_tool import delegate_task

    print("BACKWARD_COMPAT_TEST_START")
    flat_cfg = {"model": "gpt-5.4-mini", "max_iterations": 30}

    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "***"
    parent.provider = "openai-codex"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None

    with patch("tools.delegate_tool._load_config", return_value=flat_cfg):
        with patch("tools.delegate_tool._build_child_agent") as mock_build:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "ok",
                "completed": True,
                "messages": [],
            }
            mock_build.return_value = mock_child
            result_json = delegate_task(goal="test", parent_agent=parent)
            result = json.loads(result_json)
            call_kwargs = mock_build.call_args[1]

    assert "results" in result
    assert call_kwargs["model"] == "gpt-5.4-mini"
    assert call_kwargs["max_iterations"] == 30
    print(f"  flat config: model={call_kwargs['model']}, iters={call_kwargs['max_iterations']}")

    with patch("tools.delegate_tool._load_config", return_value={}):
        with patch("tools.delegate_tool._build_child_agent") as mock_build:
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "ok",
                "completed": True,
                "messages": [],
            }
            mock_build.return_value = mock_child
            result_json = delegate_task(goal="test", parent_agent=parent)
            result = json.loads(result_json)

    assert "results" in result
    print("  empty config: OK")
    print("BACKWARD_COMPAT_TEST_PASS")
    """
)


SCRIPT_REASONING_OVERRIDE = textwrap.dedent(
    """
    import sys
    import threading
    from unittest.mock import MagicMock, patch

    sys.path.insert(0, "{repo}")
    from tools.delegate_tool import _build_child_agent

    print("REASONING_OVERRIDE_TEST_START")
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "***"
    parent.provider = "openai-codex"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.reasoning_config = {"enabled": True, "effort": "low"}
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None

    with patch("tools.delegate_tool._load_config", return_value={}):
        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()
            _build_child_agent(
                task_index=0,
                goal="test",
                context=None,
                toolsets=None,
                model=None,
                max_iterations=50,
                parent_agent=parent,
                override_reasoning_effort="xhigh",
            )
            call_kwargs = MockAgent.call_args[1]
            assert call_kwargs["reasoning_config"] == {"enabled": True, "effort": "xhigh"}
    print("  xhigh override: OK")

    with patch("tools.delegate_tool._load_config", return_value={}):
        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()
            _build_child_agent(
                task_index=0,
                goal="test",
                context=None,
                toolsets=None,
                model=None,
                max_iterations=50,
                parent_agent=parent,
                override_reasoning_effort="none",
            )
            call_kwargs = MockAgent.call_args[1]
            assert call_kwargs["reasoning_config"] == {"enabled": False, "effort": "none"}
    print("  none override: OK")

    with patch("tools.delegate_tool._load_config", return_value={}):
        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()
            _build_child_agent(
                task_index=0,
                goal="test",
                context=None,
                toolsets=None,
                model=None,
                max_iterations=50,
                parent_agent=parent,
            )
            call_kwargs = MockAgent.call_args[1]
            assert call_kwargs["reasoning_config"] == {"enabled": True, "effort": "low"}
    print("  inherit parent reasoning: OK")

    with patch("tools.delegate_tool._load_config", return_value={"reasoning_effort": "low"}):
        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()
            _build_child_agent(
                task_index=0,
                goal="test",
                context=None,
                toolsets=None,
                model=None,
                max_iterations=50,
                parent_agent=parent,
                override_reasoning_effort="high",
            )
            call_kwargs = MockAgent.call_args[1]
            assert call_kwargs["reasoning_config"] == {"enabled": True, "effort": "high"}
    print("  explicit override beats config: OK")
    print("REASONING_OVERRIDE_TEST_PASS")
    """
)


SCRIPT_SCHEMA_CHECK = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, "{repo}")
    from tools.delegate_tool import DELEGATE_TASK_SCHEMA, SUPPORTED_TIERS

    print("SCHEMA_CHECK_TEST_START")
    props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
    assert "tier" in props
    assert props["tier"]["enum"] == sorted(SUPPORTED_TIERS)
    print(f"  top-level tier: enum={props['tier']['enum']}")

    task_props = props["tasks"]["items"]["properties"]
    assert "tier" in task_props
    assert task_props["tier"]["enum"] == sorted(SUPPORTED_TIERS)
    print(f"  per-task tier: enum={task_props['tier']['enum']}")

    for field in ["goal", "context", "toolsets", "tasks", "max_iterations", "acp_command", "acp_args"]:
        assert field in props
    print("  all required fields present: OK")

    assert DELEGATE_TASK_SCHEMA["name"] == "delegate_task"
    print("SCHEMA_CHECK_TEST_PASS")
    """
)


def main():
    print(f"\n{'='*70}")
    print(" TMUX-BASED DELEGATION TIER INTEGRATION TESTS")
    print(f"{'='*70}\n")

    tests = [
        ("tier_resolution", SCRIPT_TIER_RESOLUTION, "TIER_RESOLUTION_TEST_PASS"),
        ("batch_per_task", SCRIPT_BATCH_PER_TASK, "BATCH_PER_TASK_TEST_PASS"),
        ("pool_validation", SCRIPT_POOL_VALIDATION, "POOL_VALIDATION_TEST_PASS"),
        ("backward_compat", SCRIPT_BACKWARD_COMPAT, "BACKWARD_COMPAT_TEST_PASS"),
        ("reasoning_override", SCRIPT_REASONING_OVERRIDE, "REASONING_OVERRIDE_TEST_PASS"),
        ("schema_check", SCRIPT_SCHEMA_CHECK, "SCHEMA_CHECK_TEST_PASS"),
    ]

    passed = 0
    failed = 0

    for name, script, marker in tests:
        print(f"[RUN] {name}")
        try:
            filled = script.replace("{repo}", str(REPO_ROOT))
            output = tmux_run(f"test-{name}", filled, timeout=30)
            if marker in output:
                for line in output.splitlines():
                    line = line.rstrip()
                    if line and (line.startswith("  ") or "TEST_START" in line or "TEST_PASS" in line):
                        print(line)
                print(f"  [PASS] {name}\n")
                passed += 1
            else:
                lines = [l for l in output.splitlines() if l.strip()]
                print("  Last output lines:")
                for line in lines[-12:]:
                    print(f"    {line}")
                print(f"  [FAIL] {name}\n")
                failed += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            print(f"  [FAIL] {name}\n")
            failed += 1

    print(f"{'='*70}")
    print(f" RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'='*70}\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
