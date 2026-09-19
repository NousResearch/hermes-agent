"""Compaction rebuilds dynamic tool schemas (forever-session fix, #95681 arc).

Forever-sessions (Bot Mode, gateway channels) never restart; compaction is
the only boundary where the prompt cache is already broken, so it is the
one sanctioned point for a tool-snapshot rebuild. These tests pin:
- refresh_agent_mcp_tools(content_aware=True) swaps on CONTENT change under
  a stable name set (the dynamic-schema case its name-only diff missed)
- content_aware=False keeps the old no-churn behavior (MCP-reload callers)
- the compaction helper is wired into the commit path and never raises
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class _Agent:
    quiet_mode = True
    enabled_toolsets = ["image_gen"]
    disabled_toolsets = None


def _defs(desc):
    return [{"type": "function", "function": {
        "name": "image_generate", "description": desc,
        "parameters": {"type": "object", "properties": {}},
    }}]


class TestContentAwareRefresh(unittest.TestCase):
    def _agent_with(self, desc):
        agent = _Agent()
        agent.tools = _defs(desc)
        agent.valid_tool_names = {"image_generate"}
        agent._tool_snapshot_generation = 0
        return agent

    def _refresh(self, agent, new_desc, **kw):
        from tools.mcp_tool_agent import refresh_agent_mcp_tools

        with patch("model_tools.get_tool_definitions",
                   return_value=_defs(new_desc)), \
             patch("tools.mcp_tool_agent._reinject_post_build_tools",
                   return_value=set()):
            return refresh_agent_mcp_tools(agent, **kw)

    def test_content_change_swaps_when_content_aware(self):
        agent = self._agent_with("old capabilities text")
        self._refresh(agent, "new capabilities text", content_aware=True)
        self.assertIn("new capabilities",
                      agent.tools[0]["function"]["description"])

    def test_content_change_ignored_when_name_only(self):
        """MCP-reload callers keep the historical no-churn contract."""
        agent = self._agent_with("old capabilities text")
        self._refresh(agent, "new capabilities text", content_aware=False)
        self.assertIn("old capabilities",
                      agent.tools[0]["function"]["description"])

    def test_identical_content_keeps_identity(self):
        agent = self._agent_with("same text")
        before = agent.tools
        self._refresh(agent, "same text", content_aware=True)
        self.assertIs(agent.tools, before)


class TestCompactionWiring(unittest.TestCase):
    def test_helper_delegates_content_aware(self):
        from agent.conversation_compression import _refresh_agent_tool_definitions

        agent = _Agent()
        with patch("tools.mcp_tool_agent.refresh_agent_mcp_tools",
                   return_value={"newly_added"}) as m:
            changed = _refresh_agent_tool_definitions(agent)
        self.assertTrue(changed)
        m.assert_called_once_with(agent, content_aware=True)

    def test_commit_path_calls_helper_and_survives_failure(self):
        """The commit boundary invokes the refresh after prompt
        invalidation, a raising refresh must not break compaction, and the
        usage anchor is cleared. Drives the real commit path with a real
        AIAgent and a stub compressor (harness proven by
        tests/agent/test_compression_logging_session_context.py)."""
        import tempfile
        from pathlib import Path

        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "state.db")
            try:
                self._commit_path_contract(db)
            finally:
                db.close()

    def _commit_path_contract(self, db):
        from unittest.mock import MagicMock, patch

        db.create_session("COMMIT_REFRESH_SESSION", source="cli")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent

            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                session_db=db,
                session_id="COMMIT_REFRESH_SESSION",
                skip_context_files=True,
                skip_memory=True,
            )
        compressor = MagicMock()
        compressor.compress.return_value = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        ]
        compressor.compression_count = 1
        compressor.last_prompt_tokens = 0
        compressor.last_completion_tokens = 0
        compressor._last_summary_error = None
        compressor._last_compress_aborted = False
        compressor._last_aux_model_failure_model = None
        compressor._last_aux_model_failure_error = None
        agent.context_compressor = compressor
        agent.compression_in_place = False

        events = []
        invalidated_before_refresh = {}
        original_invalidate = agent._invalidate_system_prompt

        def _recording_invalidate():
            events.append("invalidate")
            invalidated_before_refresh["done"] = True
            original_invalidate()

        original_build = agent._build_system_prompt

        def _recording_build(*args, **kwargs):
            events.append("rebuild")
            return original_build(*args, **kwargs)

        agent._build_system_prompt = _recording_build
        agent._invalidate_system_prompt = _recording_invalidate
        agent._usage_anchor = {"prompt_tokens": 1}

        messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

        seen = {}

        def _refresh(a):
            events.append("refresh")
            seen["after_invalidation"] = invalidated_before_refresh.get(
                "done", False
            )

        with patch("agent.conversation_compression._refresh_agent_tool_definitions",
                   side_effect=_refresh):
            agent._compress_context(messages, "sys", approx_tokens=120_000)
        self.assertLess(events.index("invalidate"), events.index("refresh"))
        self.assertLess(events.index("refresh"), events.index("rebuild", events.index("refresh")))
        self.assertTrue(seen.get("after_invalidation"),
                        "refresh must run after prompt invalidation")
        self.assertIsNone(agent._usage_anchor,
                          "the commit path clears the usage anchor")

        # A raising refresh must not break compaction. After the first
        # compress the session id rotated, and compression already created
        # that row, so no re-registration is needed.
        agent._invalidate_system_prompt = _recording_invalidate
        with patch("agent.conversation_compression._refresh_agent_tool_definitions",
                   side_effect=RuntimeError("refresh exploded")) as refresh:
            try:
                result, _ = agent._compress_context(messages, "sys", approx_tokens=120_000)
                refresh.assert_called_once_with(agent)
                self.assertEqual(result, compressor.compress.return_value)
            except RuntimeError as exc:
                raise AssertionError(
                    "a raising tool refresh must not escape compaction"
                ) from exc


if __name__ == "__main__":
    unittest.main()
