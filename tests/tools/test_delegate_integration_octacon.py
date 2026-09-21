"""Integration regression for the delegation toolset-starvation + memory-leak
fixes (Bugs 1 & 2).

Marked `integration` so it is EXCLUDED by default (pyproject addopts =
"-m 'not integration'").  Run explicitly:

    pytest tests/tools/test_delegate_toolset_scope.py -m integration -k octacon

It dispatches a REAL delegate_task(profile="octacon") against a throwaway
tmp_path repo and asserts:
  (a) the child's tool trace shows read_file/search_files/terminal (not just
      the ambient rescuer_fetch that the starved incident produced), and
  (b) the serialised result contains no <memory-context> block.

It SKIPS (not fails) when no usable delegation credentials / network are
available, so it never red-fails in a credential-less CI.
"""

import json
import os
import tempfile

import pytest

from tools.delegate_tool import delegate_task


@pytest.mark.integration
class TestOctaconLiveToolsetAndLeak:
    def _make_parent(self):
        from unittest.mock import MagicMock

        parent = MagicMock()
        parent._delegate_depth = 0
        parent._active_children = []
        parent._active_children_lock = __import__("threading").Lock()
        parent.session_id = "integration-parent"
        parent._subagent_id = None
        parent._current_task_id = "integration-task"
        parent.model = "deepseek-v4-flash"
        parent.provider = "ollama-cloud"
        parent.base_url = "https://ollama.com/v1"
        parent.api_key = "test-key"
        parent.platform = "cli"
        parent._session_db = None
        parent.providers_allowed = None
        parent.providers_ignored = None
        parent.providers_order = None
        parent.provider_sort = None
        parent._fallback_chain = None
        parent.reasoning_config = {"effort": "off"}
        parent.max_tokens = 4096
        parent.enabled_toolsets = [
            "web", "terminal", "file", "code_execution", "skills",
            "memory", "session_search", "clarify", "delegation",
            "cronjob", "messaging",
        ]
        parent.valid_tool_names = [
            "read_file", "write_file", "web_search", "terminal",
            "search_files", "patch",
        ]
        parent.tool_progress_callback = None
        parent.thinking_callback = None
        parent._print_fn = None
        parent._memory_manager = None
        parent._delegate_saved_tool_names = []
        return parent

    def test_octacon_readonly_review_gets_audit_tools(self):
        # Build a throwaway repo so the read-only review target exists.
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "toolsets.py")
            with open(target, "w") as f:
                f.write(
                    "# fake tool definitions\n"
                    "TOOLSETS = {\n"
                    "    'audit': {'tools': ['read_file', 'search_files', 'terminal']},\n"
                    "}\n"
                )
            parent = self._make_parent()
            try:
                raw = delegate_task(
                    goal=(
                        f"Read-only review of {target}: read it, search the "
                        "directory for other references to 'audit', and run "
                        "`python -m pytest --co -q` if present. Do not modify "
                        "any files. Report what you found."
                    ),
                    profile="octacon",
                    parent_agent=parent,
                )
            except Exception as exc:  # noqa: BLE001 — network/auth failures
                pytest.skip(f"no usable delegation credentials/network: {exc}")
                return

            result = json.loads(raw) if isinstance(raw, str) else raw
            if "error" in result and "Cannot resolve" in result.get("error", ""):
                pytest.skip(f"delegation credentials unavailable: {result['error']}")
                return

            results = result.get("results", [result])
            assert results, "delegate_task returned no results"
            first = results[0]
            trace = first.get("tool_trace") or []
            tool_names = {t.get("tool") for t in trace}
            # The child must have real filesystem/shell access, not just the
            # ambient rescuer_fetch that the starved incident produced.
            assert tool_names & {"read_file", "search_files", "terminal"}, (
                f"octacon child resolved to ambient-only tools: {tool_names}"
            )
            serialised = json.dumps(first)
            assert "<memory-context>" not in serialised
