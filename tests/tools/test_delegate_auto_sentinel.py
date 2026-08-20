"""Literal ``"auto"`` sentinel normalization for ``delegation.model`` /
``delegation.provider``.

Background: ``cli-config.yaml.example`` documents ``empty = inherit parent``
for both keys, but users and setup flows commonly write the sentinel
``"auto"`` with the same intent. Before this fix the non-empty string was
treated as an explicit override:

  - ``provider="auto"`` reached ``resolve_runtime_provider(requested="auto")``,
    which matches no configured provider and silently reroutes subagents
    through the fallback chain — spending an unrelated provider's balance.
  - ``model="auto"`` was passed verbatim to the child AIAgent and then to the
    wire; the API rejects it (HTTP 401 "Model auto is not supported"), so
    every subagent spawn fails immediately.

The fix maps the literal ``"auto"`` (case-insensitive, stripped) to ``None``
so the documented inherit-from-parent path applies — mirroring the
auxiliary-path sentinel handling in ``agent/auxiliary_client.py`` and the
cron-job normalization in ``cron/scheduler.py``.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is importable.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.delegate_tool import (
    _normalize_auto_sentinel,
    _resolve_delegation_credentials,
)


class TestNormalizeAutoSentinel:
    def test_literal_auto_maps_to_none(self):
        assert _normalize_auto_sentinel("auto") is None

    def test_case_and_whitespace_insensitive(self):
        assert _normalize_auto_sentinel("AUTO") is None
        assert _normalize_auto_sentinel(" Auto ") is None

    def test_empty_and_missing_map_to_none(self):
        assert _normalize_auto_sentinel(None) is None
        assert _normalize_auto_sentinel("") is None
        assert _normalize_auto_sentinel("   ") is None

    def test_real_values_pass_through_stripped(self):
        assert _normalize_auto_sentinel("openrouter") == "openrouter"
        assert _normalize_auto_sentinel("  kimi-k2.6  ") == "kimi-k2.6"
        assert _normalize_auto_sentinel("automatic-model") == "automatic-model"


class TestDelegationCredentialsAutoSentinel:
    """``delegation.model: auto`` / ``delegation.provider: auto`` must behave
    exactly like the documented empty values: the child inherits everything
    from the parent agent and no runtime-provider resolution is attempted."""

    def _parent(self):
        parent = MagicMock()
        parent.model = "parent-model"
        parent.provider = "parent-provider"
        parent.base_url = "https://parent.example/v1"
        return parent

    def test_auto_on_both_axes_inherits_from_parent(self):
        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider"
        ) as mock_resolve:
            creds = _resolve_delegation_credentials(
                {"model": "auto", "provider": "auto"}, self._parent()
            )
        assert creds["model"] is None, (
            "literal 'auto' must normalize to inherit-parent, got "
            f"{creds['model']!r} (it would reach the wire and be rejected)"
        )
        assert creds["provider"] is None
        assert creds["base_url"] is None
        assert creds["api_key"] is None
        mock_resolve.assert_not_called()

    def test_auto_is_case_insensitive(self):
        creds = _resolve_delegation_credentials(
            {"model": " AUTO ", "provider": "Auto"}, self._parent()
        )
        assert creds["model"] is None
        assert creds["provider"] is None

    def test_auto_matches_documented_empty_behavior(self):
        """Regression: the sentinel must be indistinguishable from empty."""
        empty = _resolve_delegation_credentials({}, self._parent())
        auto = _resolve_delegation_credentials(
            {"model": "auto", "provider": "auto"}, self._parent()
        )
        assert auto == empty

    def test_explicit_override_still_wins(self):
        """A real (non-'auto') provider pin resolves full credentials."""
        runtime = {
            "api_key": "test-key",
            "base_url": "https://openrouter.example/v1",
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "request_overrides": {},
            "max_output_tokens": None,
        }
        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value=runtime,
        ) as mock_resolve:
            creds = _resolve_delegation_credentials(
                {"model": "child-model", "provider": "openrouter"},
                self._parent(),
            )
        assert creds["model"] == "child-model"
        assert creds["provider"] == "openrouter"
        assert creds["api_key"] == "test-key"
        mock_resolve.assert_called_once()
        kwargs = mock_resolve.call_args.kwargs
        assert kwargs["requested"] == "openrouter"
        assert kwargs["target_model"] == "child-model"

    def test_provider_auto_with_pinned_model_still_inherits_provider(self):
        """model pinned + provider='auto': the model override survives while
        the provider axis falls back to parent inheritance."""
        with patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider"
        ) as mock_resolve:
            creds = _resolve_delegation_credentials(
                {"model": "child-model", "provider": "auto"}, self._parent()
            )
        assert creds["model"] == "child-model"
        assert creds["provider"] is None
        mock_resolve.assert_not_called()
