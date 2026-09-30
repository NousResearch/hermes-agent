"""A dispatcher-spawned worker keeps its terminal Kanban transition tools.

Regression for #129021. The toolset resolver force-appends the ``kanban``
toolset for a dispatcher-owned worker, but ``agent.disabled_toolsets`` is
subtracted LAST at tool granularity, so a profile carrying ``kanban`` in that
list (what ``_blank_slate_minimal_toolsets`` writes for a minimal worker
profile) silently undid the injection. The worker then had no in-process
terminal transition, while the delegate guard rightly refuses
``hermes kanban complete`` from that same context.
"""
from __future__ import annotations

from pathlib import Path


def _worker_profile(tmp_path: Path, kanban_entries: int = 1) -> Path:
    """A minimal worker profile exactly as Blank Slate writes it: the kanban
    toolset suppressed in ``agent.disabled_toolsets`` and absent from the
    saved CLI selection. Returns the profile home, which is what the
    dispatcher pins as the child's ``HERMES_HOME``. ``kanban_entries`` > 1
    reproduces a config carrying the duplicate entry."""
    home = tmp_path / ".hermes"
    profile = home / "profiles" / "worker"
    profile.mkdir(parents=True)
    profile.joinpath("config.yaml").write_text(
        """
platform_toolsets:
  cli:
    - file
    - skills
    - terminal
    - vision
agent:
  disabled_toolsets:
    - browser
""".lstrip()
        + "    - kanban\n" * kanban_entries
        + """toolsets:
  - hermes-cli
""",
        encoding="utf-8",
    )
    return profile


def _resolved_names() -> set:
    """The tool names this process would expose, resolved through the two
    readers the worker's own CLI boot uses for a profile."""
    from agent.skill_utils import parse_config_string_list
    from hermes_cli.config import load_config
    from hermes_cli.tools_config import _get_platform_tools
    from model_tools import get_tool_definitions

    config = load_config()
    selected = sorted(_get_platform_tools(config, "cli", include_default_mcp_servers=False))
    disabled = parse_config_string_list((config.get("agent") or {}).get("disabled_toolsets"))
    return {
        row["function"]["name"]
        for row in get_tool_definitions(
            selected, disabled_toolsets=disabled, quiet_mode=True, skip_tool_search_assembly=True,
        )
    }


def test_dispatcher_worker_keeps_terminal_kanban_transitions(tmp_path, monkeypatch):
    """The contract: every tool the Kanban module itself classifies as
    terminating a run's ownership must be reachable by a worker that the
    dispatcher spawned, whatever the assignee profile suppressed."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(_worker_profile(tmp_path)))
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    from tools.kanban_tools import _RUN_LIFECYCLE_TOOLS

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_129021")
    worker = _resolved_names()
    assert _RUN_LIFECYCLE_TOOLS <= worker
    # The exemption is for the worker's own transitions, not the board-routing
    # surface: kanban_list stays hidden from a task worker as before.
    assert "kanban_list" not in worker

    # The same profile, same config, ordinary chat session: the suppression is
    # the operator's and must still hold, so the exemption is scoped to the
    # worker's own session and not to the profile.
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    assert not (_RUN_LIFECYCLE_TOOLS & _resolved_names())


def test_delegated_child_keeps_no_kanban_transitions(tmp_path, monkeypatch):
    """An in-process delegate_task child inherits ``HERMES_KANBAN_TASK`` from
    its parent worker, and the CLI fence is deliberately absolute for it; the
    resolver must not hand that child the transitions either."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(_worker_profile(tmp_path)))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_129021")
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    from agent.delegation_context import delegated_child_context
    from tools.kanban_tools import _RUN_LIFECYCLE_TOOLS

    with delegated_child_context():
        child = _resolved_names()
    assert not (_RUN_LIFECYCLE_TOOLS & child)


def test_duplicate_kanban_entry_does_not_defeat_the_exemption(tmp_path, monkeypatch):
    """``parse_config_string_list`` does not dedupe, and both a hand-edited list and
    a ``hermes config set`` JSON array can repeat the entry: every copy must be
    exempted, not just the first."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(_worker_profile(tmp_path, kanban_entries=2)))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_129021")
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    from model_tools import _clear_tool_defs_cache
    from tools.kanban_tools import _RUN_LIFECYCLE_TOOLS

    # The memo key hashes a frozenset, so the single- and duplicate-entry selections
    # share one slot; start clean so this asserts on the duplicate's own resolution.
    _clear_tool_defs_cache()
    assert _RUN_LIFECYCLE_TOOLS <= _resolved_names()
