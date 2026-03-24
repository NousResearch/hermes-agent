"""Regression tests for per-call RetainDB tool session routing."""

import json
from unittest.mock import MagicMock

from retaindb_integration.identity import ResolvedRetainDBIdentity
from tools import retaindb_tools


class TestRetainDBToolSessionContext:
    def setup_method(self):
        self.orig_manager = retaindb_tools._session_manager
        self.orig_key = retaindb_tools._session_key

    def teardown_method(self):
        retaindb_tools._session_manager = self.orig_manager
        retaindb_tools._session_key = self.orig_key

    def test_explicit_call_context_wins_over_module_global_state(self):
        global_manager = MagicMock()
        global_manager.get_profile.return_value = {"name": "global"}
        explicit_manager = MagicMock()
        explicit_manager.get_profile.return_value = {"name": "explicit"}

        retaindb_tools.set_session_context(global_manager, "global-session")

        result = json.loads(
            retaindb_tools._handle_retaindb_profile(
                {},
                retaindb_manager=explicit_manager,
                retaindb_session_key="explicit-session",
            )
        )

        assert result == {"result": {"name": "explicit"}}
        explicit_manager.get_profile.assert_called_once_with("explicit-session")
        global_manager.get_profile.assert_not_called()

    def test_context_tool_serializes_identity_dataclass_payload(self):
        manager = MagicMock()
        manager.get_context.return_value = {
            "identity": ResolvedRetainDBIdentity(
                user_id="user-1",
                session_id="session-1",
                agent_id="hermes",
                project="default",
                source="platform",
                peer_name="alice",
                platform="telegram",
                chat_id="chat-1",
            ),
            "context": "[RetainDB Context]\nRelevant memories:\n- cedar-42",
            "profile": {"memories": []},
            "query": {"results": []},
        }

        result = json.loads(
            retaindb_tools._handle_retaindb_context(
                {"query": "What matters now?"},
                retaindb_manager=manager,
                retaindb_session_key="session-1",
            )
        )

        assert result["result"]["identity"] == {
            "user_id": "user-1",
            "session_id": "session-1",
            "agent_id": "hermes",
            "project": "default",
            "source": "platform",
            "peer_name": "alice",
            "platform": "telegram",
            "chat_id": "chat-1",
        }
        assert "RetainDB Context" in result["result"]["context"]
        manager.get_context.assert_called_once_with("session-1", "What matters now?")
