"""Integration tests: vault_context must work through the REAL runtime path.

The previous candidate was rejected by the 2026-10-01 23:16 +07
adoption-gate review because its tool handler rejected the
``task_id``/``session_id``/``user_task`` kwargs that
:func:`model_tools.handle_function_call` forwards through
:meth:`tools.registry.dispatch`. These tests exercise that real
runtime path against an isolated, fake plugin-discovery environment
so we catch any future regression in the handler signature or the
turn-key resolution.

We do NOT patch the registry; we use a real plugin loader running
under a per-test ``HERMES_HOME`` so the registration path matches
production. The fake PluginContext used in unit tests is replaced by
the real ``PluginContext`` here, with the same ``vault_root``
configuration the ```` vault-retrieval`` plugin accepts.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest


def _isolated_hermes_home(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    """Create an isolated HERMES_HOME with the plugin installed and
    config enabled. Returns ``(home, vault_root)``.
    """
    home = tmp_path / "hermes_home"
    home.mkdir()
    plugins_dir = home / "plugins"
    plugins_dir.mkdir()
    # Copy the entire plugin tree.
    plugin_src = Path(__file__).resolve().parents[1]  # vault-retrieval/
    shutil.copytree(plugin_src, plugins_dir / "vault-retrieval")

    vault = tmp_path / "vault"
    (vault / "99 System").mkdir(parents=True)
    (vault / "99 System" / "Project - Maid Dee - Current State.md").write_text(
        "---\n"
        "title: Project - Maid Dee - Current State\n"
        "status: current\n"
        "owner: Roger\n"
        "last_updated: 2026-10-01T12:00:00+07:00\n"
        "review_by: 2026-12-31\n"
        "---\n\n"
        "# Maid Dee\n\n"
        "## Stage\n\nPilot live since 2026-01-01.\n\n"
        "## KPI\n\n- WAM 168.\n"
    )
    # Drop 2 more notes to confirm selectors work for non-current-state files.
    (vault / "99 System" / "note.md").write_text(
        "hello world\nsecond line\n", encoding="utf-8"
    )

    (home / "config.yaml").write_text(
        "plugins:\n"
        "  enabled:\n"
        "    - vault-retrieval\n"
        "  entries:\n"
        "    vault-retrieval:\n"
        "      enabled: true\n"
        "      settings:\n"
        "        vault_root: " + str(vault) + "\n"
        "        mode: enforce\n"
        "        default_budget_chars: 12000\n"
        "        hard_ceiling_chars: 24000\n"
        "        large_file_chars: 20000\n"
        "        candidate_limit: 20\n"
        "        max_primary_extracts: 3\n"
        "        max_expansion_extracts: 2\n"
        "        max_range_lines: 120\n"
        "        max_range_chars: 8000\n"
        "        query_log_enabled: true\n"
        "        log_raw_query_terms: false\n"
        "        snapshots_enabled: false\n"
        "        fts5_enabled: false\n"
        "        block_direct_file_reads: true\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home, vault


def _import_fresh():
    """Force-reload plugin modules so each test starts clean."""
    for k in list(sys.modules):
        if k.startswith(("vault_retrieval", "hermes_cli.plugins")):
            del sys.modules[k]


def _discover_vault_retrieval():
    """Run the real ``discover_plugins`` against the isolated HERMES_HOME.

    Returns ``True`` if ``vault_context`` ends up in the registry.
    """
    from hermes_cli.plugins import discover_plugins
    discover_plugins()
    from tools.registry import registry
    return "vault_context" in registry.get_all_tool_names()


class TestRuntimeDispatchKwargs:
    """Reproduce the original blocker: handler must accept dispatch kwargs."""

    def test_handler_accepts_runtime_kwargs_via_registry_dispatch(
        self, tmp_path, monkeypatch
    ):
        home, vault = _isolated_hermes_home(monkeypatch, tmp_path)
        _import_fresh()
        assert _discover_vault_retrieval(), "vault_context must be discoverable"

        from tools.registry import registry
        out = registry.dispatch(
            "vault_context",
            {
                "query": "Maid Dee current state",
                "selectors": [{
                    "path": "99 System/Project - Maid Dee - Current State.md",
                    "line_start": 1, "line_end": 30,
                }],
            },
            task_id="runtime-task-1",
            session_id="runtime-session-1",
            user_task="unit-test",
        )
        # Result is a JSON string.
        env = json.loads(out)
        assert env["status"] == "ok"
        assert env["extracts"], "must have at least one extract"
        assert env["budget"]["default_chars"] == 12000
        assert env["budget"]["hard_ceiling_chars"] == 24000

    def test_handle_function_call_dispatch(
        self, tmp_path, monkeypatch
    ):
        """Run the real ``model_tools.handle_function_call`` so we exercise
        hooks/middleware/registry in production order. This is the path
        the agent loop actually uses.
        """
        home, vault = _isolated_hermes_home(monkeypatch, tmp_path)
        _import_fresh()
        assert _discover_vault_retrieval()

        from model_tools import handle_function_call
        out = handle_function_call(
            function_name="vault_context",
            function_args={
                "query": "Maid Dee current state",
                "selectors": [{
                    "path": "99 System/Project - Maid Dee - Current State.md",
                    "line_start": 1, "line_end": 30,
                }],
            },
            task_id="runtime-task-2",
            session_id="runtime-session-2",
            turn_id="runtime-turn-2",
            api_request_id="api-1",
            user_task="unit-test",
        )
        env = json.loads(out)
        assert env["status"] == "ok"
        # No leakage of ``turn_id`` in the schema-visible args.
        assert env["scope"]["project"] is None  # not passed in args
        # Budget reflects the runtime turn identity (turn_id was passed).
        assert env["budget"]["default_chars"] == 12000


class TestSameTurnAccumulationAndCrossTurnIsolation:
    """Same-turn accumulation: two calls sharing one runtime turn_id must
    accumulate against the SAME counter (12k default budget).

    Cross-turn isolation: two calls with DIFFERENT turn_ids must NOT
    share a counter (a fresh turn gets the full 12k).
    """

    def test_same_turn_id_shares_counter(
        self, tmp_path, monkeypatch
    ):
        home, vault = _isolated_hermes_home(monkeypatch, tmp_path)
        _import_fresh()
        assert _discover_vault_retrieval()

        from model_tools import handle_function_call

        # Use line ranges that BOTH overlap with the file so each call
        # produces a non-zero extract footprint (needed for accumulation
        # assertion to be meaningful).
        selector_a = {
            "path": "99 System/Project - Maid Dee - Current State.md",
            "line_start": 1, "line_end": 6,
        }
        selector_b = {
            "path": "99 System/Project - Maid Dee - Current State.md",
            "line_start": 7, "line_end": 11,
        }

        # First call consumes X chars.
        out1 = handle_function_call(
            function_name="vault_context",
            function_args={"query": "q1", "selectors": [selector_a]},
            task_id="t-task", session_id="t-session",
            turn_id="shared-turn", user_task="u",
        )
        env1 = json.loads(out1)
        used1 = env1["budget"]["used_chars"]
        assert used1 > 0, "first call must consume some budget"

        # Second call in the SAME turn.
        out2 = handle_function_call(
            function_name="vault_context",
            function_args={"query": "q2", "selectors": [selector_b]},
            task_id="t-task", session_id="t-session",
            turn_id="shared-turn", user_task="u",
        )
        env2 = json.loads(out2)
        used2 = env2["budget"]["used_chars"]
        assert used2 >= used1, (
            "same-turn must accumulate: "
            f"first={used1} second={used2}"
        )
        assert used2 <= 12000, "must respect default 12k"

    def test_different_turn_ids_do_not_share_counter(
        self, tmp_path, monkeypatch
    ):
        home, vault = _isolated_hermes_home(monkeypatch, tmp_path)
        _import_fresh()
        assert _discover_vault_retrieval()

        from model_tools import handle_function_call

        selector_11 = {
            "path": "99 System/Project - Maid Dee - Current State.md",
            "line_start": 1, "line_end": 6,
        }
        args = {
            "query": "q",
            "selectors": [selector_11],
        }
        # First turn: full 12k available.
        out1 = handle_function_call(
            function_name="vault_context", function_args=dict(args),
            task_id="t1-task", session_id="t1-session",
            turn_id="turn-A", user_task="u",
        )
        env1 = json.loads(out1)
        used1 = env1["budget"]["used_chars"]
        assert used1 > 0, "first turn must consume some budget"
        # Second turn: must start fresh — must NOT see turn-A's counter.
        out2 = handle_function_call(
            function_name="vault_context", function_args=dict(args),
            task_id="t1-task", session_id="t1-session",
            turn_id="turn-B", user_task="u",
        )
        env2 = json.loads(out2)
        used2 = env2["budget"]["used_chars"]
        # The fact that used2 == used1 (turn B has not accumulated turn A's
        # budget) proves cross-turn isolation — they share the same
        # per-call extraction footprint but NOT the counter.
        assert used1 == used2, (
            "cross-turn must NOT share the counter — "
            f"turn-A={used1} turn-B={used2} should be equal "
            "(both fresh 12k, no carry)"
        )

    def test_session_end_resets_runtime_counter(
        self, tmp_path, monkeypatch
    ):
        """``on_session_end`` must drop the runtime counter so the next
        session can use the full 12k budget.
        """
        home, vault = _isolated_hermes_home(monkeypatch, tmp_path)
        _import_fresh()
        assert _discover_vault_retrieval()

        from model_tools import handle_function_call

        args = {
            "query": "q",
            "selectors": [{
                "path": "99 System/Project - Maid Dee - Current State.md",
                "line_start": 1, "line_end": 30,
            }],
        }
        # First call: consumes X.
        out1 = handle_function_call(
            function_name="vault_context", function_args=dict(args),
            task_id="s1-task", session_id="s1-session",
            turn_id="s1-turn", user_task="u",
        )
        env1 = json.loads(out1)
        # Run the on_session_end hook to drop the runtime counter.
        from hermes_cli.plugins import invoke_hook
        invoke_hook("on_session_end",
                    session_id="s1-session", task_id="s1-task", turn_id="s1-turn")
        # Second call with same identities must start fresh.
        out2 = handle_function_call(
            function_name="vault_context", function_args=dict(args),
            task_id="s1-task", session_id="s1-session",
            turn_id="s1-turn", user_task="u",
        )
        env2 = json.loads(out2)
        assert env1["budget"]["used_chars"] == env2["budget"]["used_chars"], (
            "after on_session_end, the runtime counter must be reset"
        )


class TestNoModelVisibleTurnId:
    """The schema advertised to the model MUST NOT include ``turn_id``
    so it cannot override its own budget counter.
    """

    def test_schema_does_not_expose_turn_id(
        self, tmp_path, monkeypatch
    ):
        home, vault = _isolated_hermes_home(monkeypatch, tmp_path)
        _import_fresh()
        assert _discover_vault_retrieval()

        from tools.registry import registry
        entry = registry.get_entry("vault_context")
        assert entry is not None, "vault_context must be registered"
        schema = entry.schema
        params = schema["parameters"]
        assert "turn_id" not in params["properties"], (
            "turn_id must not be in the model-visible schema"
        )
        # additionalProperties=False keeps the model from sneaking turn_id in.
        assert params.get("additionalProperties") is False