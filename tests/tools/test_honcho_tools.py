"""Regression tests for per-call Honcho tool session routing."""

import json
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from tools import honcho_tools


class TestCheckHonchoAvailable:
    """Tests for _check_honcho_available (banner + runtime gating)."""

    def setup_method(self):
        self.orig_manager = honcho_tools._session_manager
        self.orig_key = honcho_tools._session_key

    def teardown_method(self):
        honcho_tools._session_manager = self.orig_manager
        honcho_tools._session_key = self.orig_key

    def test_returns_true_when_session_active(self):
        """Fast path: session context already injected (mid-conversation)."""
        honcho_tools._session_manager = MagicMock()
        honcho_tools._session_key = "test-key"
        assert honcho_tools._check_honcho_available() is True

    def test_returns_true_when_configured_but_no_session(self):
        """Slow path: honcho configured but agent not started yet (banner time)."""
        honcho_tools._session_manager = None
        honcho_tools._session_key = None

        @dataclass
        class FakeConfig:
            enabled: bool = True
            api_key: str = "test-key"
            base_url: str = None

        with patch("tools.honcho_tools.HonchoClientConfig", create=True):
            with patch(
                "honcho_integration.client.HonchoClientConfig"
            ) as mock_cls:
                mock_cls.from_global_config.return_value = FakeConfig()
                assert honcho_tools._check_honcho_available() is True

    def test_returns_false_when_not_configured(self):
        """No session, no config: tool genuinely unavailable."""
        honcho_tools._session_manager = None
        honcho_tools._session_key = None

        @dataclass
        class FakeConfig:
            enabled: bool = False
            api_key: str = None
            base_url: str = None

        with patch(
            "honcho_integration.client.HonchoClientConfig"
        ) as mock_cls:
            mock_cls.from_global_config.return_value = FakeConfig()
            assert honcho_tools._check_honcho_available() is False

    def test_returns_false_when_import_fails(self):
        """Graceful fallback when honcho_integration not installed."""
        import sys

        honcho_tools._session_manager = None
        honcho_tools._session_key = None

        # Hide honcho_integration from the import system to simulate
        # an environment where the package is not installed.
        hidden = {
            k: sys.modules.pop(k)
            for k in list(sys.modules)
            if k.startswith("honcho_integration")
        }
        try:
            with patch.dict(sys.modules, {"honcho_integration": None,
                                          "honcho_integration.client": None}):
                assert honcho_tools._check_honcho_available() is False
        finally:
            sys.modules.update(hidden)


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
