"""Tests for the procedural-content gate in tools/memory_tool.py.

Covers all four anti-patterns the gate rejects, plus the bypass flag
for legitimate env facts, and false-positive safety for common durable facts.
"""

import json
import pytest

from tools.memory_tool import (
    MemoryStore,
    _detect_procedural_content,
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
        # A stable env fact that happens to contain a file path
        result = _detect_procedural_content(
            "AWS_PROFILE=mcp-hive points at ~/.aws/credentials"
        )
        # This does NOT contain .md or /references/ — should not trigger
        # (Testing that bypass isn't needed for this particular fact)
        # But we test the bypass mechanism via MemoryStore below

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
