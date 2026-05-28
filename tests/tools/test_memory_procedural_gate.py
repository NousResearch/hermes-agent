"""Tests for the procedural-content gate in tools/memory_tool.py.

Covers all four anti-patterns the gate rejects, plus the bypass flag
for legitimate env facts, and false-positive safety for common durable facts.
"""

import json
import pytest

from tools.memory_tool import (
    MemoryStore,
    _detect_procedural_content,
    _to_bool,
    memory_tool,
    _PROCEDURAL_REJECTION_MSG,
)


# =========================================================================
# _detect_procedural_content unit tests
# =========================================================================


class TestProceduralGateHeuristic1_FilePath:
    """Heuristic 1 — /references/ path or .md suffix."""

    def test_references_path_blocked(self):
        content = "Fix is in kanban-orchestrator/references/stuck-dispatch-and-missing-pings.md"
        result = _detect_procedural_content(content)
        assert result == _PROCEDURAL_REJECTION_MSG

    def test_any_md_file_blocked(self):
        result = _detect_procedural_content("See SKILL.md for the recipe")
        assert result == _PROCEDURAL_REJECTION_MSG

    def test_inline_references_path_blocked(self):
        result = _detect_procedural_content(
            "Recipe in kanban-orchestrator/references/human-review-approvals-and-force-push-gates.md"
        )
        assert result == _PROCEDURAL_REJECTION_MSG

    def test_dotmd_extension_blocked(self):
        result = _detect_procedural_content("braintrust-eng-process/references/ac-enumeration.md")
        assert result == _PROCEDURAL_REJECTION_MSG


class TestProceduralGateHeuristic2_SqlCode:
    """Heuristic 2 — SQL, code blocks, shell-command lines."""

    def test_sql_select_blocked(self):
        result = _detect_procedural_content("SELECT json_extract(payload,'$.reason') FROM task_events")
        assert result == _PROCEDURAL_REJECTION_MSG

    def test_sql_update_blocked(self):
        result = _detect_procedural_content(
            "UPDATE tasks SET claim_lock=NULL,claim_expires=NULL WHERE id='t_abc';"
        )
        assert result == _PROCEDURAL_REJECTION_MSG

    def test_sql_insert_blocked(self):
        result = _detect_procedural_content(
            "INSERT INTO task_events (task_id, kind) VALUES ('t_abc', 'rejected');"
        )
        assert result == _PROCEDURAL_REJECTION_MSG

    def test_triple_backtick_code_block_blocked(self):
        result = _detect_procedural_content("```bash\ngit push origin main\n```")
        assert result == _PROCEDURAL_REJECTION_MSG

    def test_shell_command_at_line_start_blocked(self):
        result = _detect_procedural_content("Fix editable install:\ncd ~/.hermes/hermes-agent")
        assert result == _PROCEDURAL_REJECTION_MSG

    def test_git_command_at_line_start_blocked(self):
        result = _detect_procedural_content("git push origin HEAD:my-branch")
        assert result == _PROCEDURAL_REJECTION_MSG

    def test_uv_command_at_line_start_blocked(self):
        result = _detect_procedural_content("uv pip install -e . --no-deps")
        assert result == _PROCEDURAL_REJECTION_MSG

    def test_hermes_command_blocked(self):
        result = _detect_procedural_content("hermes cron run <job_id>")
        assert result == _PROCEDURAL_REJECTION_MSG


class TestProceduralGateHeuristic3_NumberedSteps:
    """Heuristic 3 — numbered-step markers."""

    def test_numbered_steps_blocked(self):
        result = _detect_procedural_content(
            "OPS HYGIENE: (1) verify artifacts. (2) kill -9 workers. (3) fix editable install."
        )
        assert result == _PROCEDURAL_REJECTION_MSG

    def test_dot_numbered_steps_blocked(self):
        result = _detect_procedural_content("1. clone repo\n2. install deps\n3. run tests")
        assert result == _PROCEDURAL_REJECTION_MSG

    def test_single_numbered_step_blocked(self):
        # Even a single "1. do something" is a recipe indicator
        result = _detect_procedural_content("1. run `git fetch` to pick up upstream changes")
        assert result == _PROCEDURAL_REJECTION_MSG


class TestProceduralGateHeuristic4_SignalNearVerb:
    """Heuristic 4 — procedural signal word near imperative verb."""

    def test_via_plus_verb_blocked(self):
        # "via terminal" near an imperative verb
        result = _detect_procedural_content(
            "APPROVAL→MERGE: merge it via kanban_approve flow"
        )
        assert result == _PROCEDURAL_REJECTION_MSG

    def test_recipe_blocked(self):
        result = _detect_procedural_content("Full recipe: run the audit first then verify")
        assert result == _PROCEDURAL_REJECTION_MSG

    def test_procedure_blocked(self):
        result = _detect_procedural_content(
            "Standard procedure is to run the check and verify output"
        )
        assert result == _PROCEDURAL_REJECTION_MSG

    def test_flow_colon_blocked(self):
        result = _detect_procedural_content(
            "flow: kanban_show → find PR → gh pr view → merge if clean"
        )
        assert result == _PROCEDURAL_REJECTION_MSG


# =========================================================================
# Bypass flag — legitimate env facts that trip the heuristics
# =========================================================================


class TestProceduralGateBypass:
    """bypass_procedural_check=True lets env facts with path patterns through."""

    def test_bypass_env_fact_with_path(self):
        # A stable env fact that happens to contain a file path — but NOT an
        # .md path or /references/ path, so the gate should let it through
        # WITHOUT a bypass flag.  Confirms the env-fact pattern does not
        # over-trigger.
        result = _detect_procedural_content(
            "AWS_PROFILE=mcp-hive points at ~/.aws/credentials"
        )
        assert result is None, (
            "env fact without .md / /references/ path should not trigger detection"
        )

    @pytest.fixture()
    def store(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        s = MemoryStore(memory_char_limit=500, user_char_limit=300)
        s.load_from_disk()
        return s

    def test_store_add_blocks_md_path_by_default(self, store):
        result = store.add("memory", "See SKILL.md for the recipe")
        assert result["success"] is False
        assert "procedure" in result["error"].lower()

    def test_store_add_bypasses_with_flag(self, store):
        # A genuine env fact that contains an .md path the gate would block
        content = "Project conventions in AGENTS.md govern all contributors"
        result = store.add("memory", content, bypass_procedural_check=True)
        assert result["success"] is True

    def test_store_replace_blocks_procedural_by_default(self, store):
        store.add("memory", "initial fact", bypass_procedural_check=True)
        result = store.replace("memory", "initial fact", "1. do this 2. do that")
        assert result["success"] is False
        assert "procedure" in result["error"].lower()

    def test_store_replace_bypasses_with_flag(self, store):
        store.add("memory", "initial fact", bypass_procedural_check=True)
        result = store.replace(
            "memory",
            "initial fact",
            "Project root is ~/Desktop/work/myapp/AGENTS.md area",
            bypass_procedural_check=True,
        )
        assert result["success"] is True


# =========================================================================
# memory_tool() dispatcher integration
# =========================================================================


class TestMemoryToolDispatcherProcedural:
    """End-to-end: memory_tool() -> MemoryStore -> procedural gate."""

    @pytest.fixture()
    def store(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        s = MemoryStore(memory_char_limit=500, user_char_limit=300)
        s.load_from_disk()
        return s

    def test_add_numbered_steps_rejected(self, store):
        result = json.loads(
            memory_tool(
                action="add",
                target="memory",
                content="(1) verify artifacts (2) clear stale lock (3) reinstall editable",
                store=store,
            )
        )
        assert result["success"] is False
        assert "procedure" in result["error"].lower()

    def test_add_references_path_rejected(self, store):
        result = json.loads(
            memory_tool(
                action="add",
                target="memory",
                content="Full recipe in kanban-orchestrator/references/stuck-dispatch.md",
                store=store,
            )
        )
        assert result["success"] is False
        assert "procedure" in result["error"].lower()

    def test_add_sql_rejected(self, store):
        result = json.loads(
            memory_tool(
                action="add",
                target="memory",
                content="SELECT * FROM tasks WHERE status='ready'",
                store=store,
            )
        )
        assert result["success"] is False
        assert "procedure" in result["error"].lower()

    def test_add_code_block_rejected(self, store):
        result = json.loads(
            memory_tool(
                action="add",
                target="memory",
                content="Fix with: ```bash\ncd repo && pip install -e .\n```",
                store=store,
            )
        )
        assert result["success"] is False
        assert "procedure" in result["error"].lower()

    def test_add_bypass_flag_works(self, store):
        """Legitimate env fact with path goes through with bypass_procedural_check=True."""
        result = json.loads(
            memory_tool(
                action="add",
                target="memory",
                content="Project conventions in AGENTS.md govern all contributors",
                store=store,
                bypass_procedural_check=True,
            )
        )
        assert result["success"] is True

    def test_add_clean_fact_passes(self, store):
        """Durable env facts without any procedural signals pass without bypass."""
        result = json.loads(
            memory_tool(
                action="add",
                target="memory",
                content="User prefers dark mode and concise responses",
                store=store,
            )
        )
        assert result["success"] is True

    def test_add_env_var_fact_passes(self, store):
        """Env-var style facts pass without bypass."""
        result = json.loads(
            memory_tool(
                action="add",
                target="memory",
                content="AWS_PROFILE=mcp-hive is the default AWS profile for BrainTrust work",
                store=store,
            )
        )
        assert result["success"] is True


# =========================================================================
# False-positive safety — common durable facts should NOT be blocked
# =========================================================================


class TestProceduralGateFalsePositives:
    """Common durable user/env facts that the gate must not block."""

    def test_user_preference_passes(self):
        assert _detect_procedural_content("User prefers dark mode") is None

    def test_env_fact_passes(self):
        assert _detect_procedural_content("Project uses Python 3.12 with FastAPI") is None

    def test_provider_fact_passes(self):
        assert _detect_procedural_content("Main LLM provider is Anthropic, model claude-sonnet-4") is None

    def test_team_fact_passes(self):
        assert _detect_procedural_content("Sahil runs the braintrust team at Trilogy Innovations") is None

    def test_aws_profile_without_path_passes(self):
        assert _detect_procedural_content("AWS_PROFILE=mcp-hive is the default BrainTrust AWS profile") is None

    def test_git_identity_fact_passes(self):
        # Doesn't start with a shell command verb at line start
        assert _detect_procedural_content("Git identity: sahilm-ai, OAuth token in GH_TOKEN_SAHILM_AI") is None

    def test_tool_quirk_fact_passes(self):
        assert _detect_procedural_content("Telegram does not render pipe tables") is None

    def test_synapse_os_ids_pass(self):
        assert (
            _detect_procedural_content(
                "SYNAPSE OS: team team.trilogy-innovations, project project.braintrust, OKR f1320919"
            )
            is None
        )


# =========================================================================
# SQL regex narrowing — false-positive safety (prose with SQL-ish verbs)
# =========================================================================


class TestSQLRegexNarrowing:
    """Bare SQL-ish verbs in prose must NOT trigger the SQL heuristic.

    The old regex matched any word-boundary occurrence of UPDATE / DELETE /
    INSERT which caused false positives on legitimate user-preference memories.
    The narrowed regex requires SQL structural context (e.g. UPDATE x SET,
    DELETE FROM, INSERT INTO) to fire.
    """

    # --- Positive cases (should still be rejected) ---

    def test_sql_update_set_rejected(self):
        """Full UPDATE … SET … form is rejected."""
        result = _detect_procedural_content("UPDATE memory SET active=1 WHERE id=42;")
        assert result == _PROCEDURAL_REJECTION_MSG

    def test_sql_delete_from_rejected(self):
        result = _detect_procedural_content("DELETE FROM tasks WHERE id='t_abc'")
        assert result == _PROCEDURAL_REJECTION_MSG

    def test_sql_insert_into_rejected(self):
        result = _detect_procedural_content("INSERT INTO logs VALUES ('x', 'y')")
        assert result == _PROCEDURAL_REJECTION_MSG

    def test_sql_create_table_rejected(self):
        result = _detect_procedural_content("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
        assert result == _PROCEDURAL_REJECTION_MSG

    def test_sql_select_from_rejected(self):
        result = _detect_procedural_content("SELECT id, status FROM tasks WHERE status='ready'")
        assert result == _PROCEDURAL_REJECTION_MSG

    # --- Negative cases (prose with SQL-ish verbs — must NOT be rejected) ---

    def test_prose_update_passes(self):
        """'update' in prose (no SET clause) must not trigger the SQL heuristic."""
        result = _detect_procedural_content(
            "Sahil prefers to update the dashboard before standup"
        )
        assert result is None, (
            "bare 'update' in prose should not be rejected as SQL"
        )

    def test_prose_delete_passes(self):
        result = _detect_procedural_content("Delete the old draft after merging")
        assert result is None, (
            "bare 'delete' in prose should not be rejected as SQL"
        )

    def test_prose_insert_passes(self):
        result = _detect_procedural_content("insert custom branding into the export")
        assert result is None, (
            "bare 'insert' in prose should not be rejected as SQL"
        )

    def test_prose_select_passes(self):
        # Note: "Select ... from" is ambiguous enough that the narrowed regex
        # may still fire (SELECT + content + FROM matches the pattern even in
        # prose). The important false-positive cases are UPDATE/DELETE/INSERT.
        # Test a clearly-prose sentence without FROM to confirm no over-trigger
        # on bare "select" alone.
        result = _detect_procedural_content("Select the right model for the job")
        assert result is None, (
            "bare 'select' in prose without FROM should not be rejected"
        )


# =========================================================================
# _to_bool helper — string bypass arg hardening
# =========================================================================


class TestToBoolHardening:
    """bypass_procedural_check must treat string '0'/'no'/'false' as False.

    ``bool('0')`` is True in Python — a raw bool() coercion would silently
    bypass the procedural gate for any caller that passes a string value.
    _to_bool() provides correct semantics.
    """

    # --- _to_bool unit tests ---

    def test_to_bool_false_values(self):
        for val in ("0", "no", "false", "False", "FALSE", "n", "f"):
            assert _to_bool(val) is False, f"_to_bool({val!r}) should be False"

    def test_to_bool_true_values(self):
        for val in ("1", "yes", "true", "True", "TRUE", "y", "t"):
            assert _to_bool(val) is True, f"_to_bool({val!r}) should be True"

    def test_to_bool_bool_passthrough(self):
        assert _to_bool(True) is True
        assert _to_bool(False) is False

    def test_to_bool_int(self):
        assert _to_bool(1) is True
        assert _to_bool(0) is False

    # --- Integration: string bypass args via MemoryStore ---

    @pytest.fixture()
    def store(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        s = MemoryStore(memory_char_limit=500, user_char_limit=300)
        s.load_from_disk()
        return s

    def test_bypass_string_zero_does_not_bypass(self, store):
        """bypass_procedural_check='0' must NOT bypass the gate."""
        result = json.loads(
            memory_tool(
                action="add",
                target="memory",
                content="See SKILL.md for the recipe",
                store=store,
                bypass_procedural_check="0",
            )
        )
        assert result["success"] is False, (
            "bypass_procedural_check='0' (string) should NOT bypass the procedural gate"
        )

    def test_bypass_string_no_does_not_bypass(self, store):
        """bypass_procedural_check='no' must NOT bypass the gate."""
        result = json.loads(
            memory_tool(
                action="add",
                target="memory",
                content="See SKILL.md for the recipe",
                store=store,
                bypass_procedural_check="no",
            )
        )
        assert result["success"] is False, (
            "bypass_procedural_check='no' (string) should NOT bypass the procedural gate"
        )

    def test_bypass_string_true_does_bypass(self, store):
        """bypass_procedural_check='true' (string) SHOULD bypass the gate."""
        result = json.loads(
            memory_tool(
                action="add",
                target="memory",
                content="Project conventions in AGENTS.md govern all contributors",
                store=store,
                bypass_procedural_check="true",
            )
        )
        assert result["success"] is True, (
            "bypass_procedural_check='true' (string) should bypass the procedural gate"
        )

    def test_bypass_true_roundtrip(self, store):
        """bypass=True round-trip: entry persists in the store after add."""
        content = "Project root is ~/Desktop/work/myapp (stable env fact)"
        add_result = json.loads(
            memory_tool(
                action="add",
                target="memory",
                content=content,
                store=store,
                bypass_procedural_check=True,
            )
        )
        assert add_result["success"] is True
        # The add response carries the live entries list — confirm the entry is there
        assert any(content in entry for entry in add_result.get("entries", [])), (
            "bypassed entry should be present in the store's entries after add"
        )
        # Also confirm via the MemoryStore object directly
        assert content in store._entries_for("memory"), (
            "bypassed entry should be persisted in the in-memory store"
        )


