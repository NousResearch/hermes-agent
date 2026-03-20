"""Regression tests for per-call Honcho tool session routing."""

import json
from unittest.mock import MagicMock

from tools import honcho_tools


class TestHonchoToolSessionContext:
    def setup_method(self):
        self.orig_manager = honcho_tools._session_manager
        self.orig_key = honcho_tools._session_key

    def teardown_method(self):
        honcho_tools._session_manager = self.orig_manager
        honcho_tools._session_key = self.orig_key

    def test_explicit_call_context_wins_over_module_global_state(self):
        global_manager = MagicMock()
        global_manager.get_peer_card.return_value = ["global"]
        explicit_manager = MagicMock()
        explicit_manager.get_peer_card.return_value = ["explicit"]

        honcho_tools.set_session_context(global_manager, "global-session")

        result = json.loads(
            honcho_tools._handle_honcho_profile(
                {},
                honcho_manager=explicit_manager,
                honcho_session_key="explicit-session",
            )
        )

        assert result == {"result": ["explicit"]}
        explicit_manager.get_peer_card.assert_called_once_with("explicit-session")
        global_manager.get_peer_card.assert_not_called()
