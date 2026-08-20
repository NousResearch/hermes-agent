"""Tests for _should_use_native_vision_fast_path (tools/vision_tools.py).

Regression: providers that declare supports_vision_tool_messages=False
(e.g. Xiaomi/mimo-v2.5) must NOT use the native fast path even when
models.dev reports the model as vision-capable.  The old ``or`` logic
returned True for these providers, causing the agent to embed images
inside tool-result messages that the provider's API rejects/ignores —
leading to hallucinated image descriptions on the next turn.

Fix: only enable the native fast path when:
  1. _supports_media_in_tool_results() returns True  (provider allowlist), OR
  2. The user explicitly set model.supports_vision: true in config.yaml
     (deliberate escape hatch for custom/local providers).

models.dev metadata alone is no longer sufficient.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_fast_path(
    *,
    provider: str,
    model: str,
    image_mode: str = "native",
    supports_vision_lookup: bool | None = None,
    supports_vision_override: bool | None = None,
    tool_result_support: bool | None = None,
):
    """Context-manager bundle that stubs every lazy import inside
    _should_use_native_vision_fast_path.

    Because the function uses ``from X import Y`` inside the try block,
    we must patch at the *source* module, not at tools.vision_tools.
    """
    import contextlib
    from tools import vision_tools

    patches = [
        # Lazy imports inside _should_use_native_vision_fast_path
        patch("agent.auxiliary_client._read_main_provider", return_value=provider),
        patch("agent.auxiliary_client._read_main_model", return_value=model),
        patch("agent.image_routing.decide_image_input_mode", return_value=image_mode),
        patch("hermes_cli.config.load_config", return_value={}),
    ]

    if tool_result_support is not None:
        patches.append(
            patch.object(
                vision_tools,
                "_supports_media_in_tool_results",
                return_value=tool_result_support,
            )
        )

    if supports_vision_lookup is not None:
        patches.append(
            patch("agent.image_routing._lookup_supports_vision", return_value=supports_vision_lookup)
        )

    if supports_vision_override is not None:
        patches.append(
            patch("agent.image_routing._supports_vision_override", return_value=supports_vision_override)
        )

    @contextlib.contextmanager
    def _ctx():
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            yield

    return _ctx()


# ---------------------------------------------------------------------------
# Core regression: Xiaomi must NOT use native fast path
# ---------------------------------------------------------------------------


class TestXiaomiNativePathDisabled:
    """Providers with supports_vision_tool_messages=False must return False."""

    def test_xiaomi_mimo_returns_false(self):
        """mimo-v2.5: vision=True in models.dev but tool result support=False."""
        from tools.vision_tools import _should_use_native_vision_fast_path

        # _supports_media_in_tool_results reads the real xiaomi ProviderProfile
        # which has supports_vision_tool_messages=False → False.
        # _lookup_supports_vision simulates models.dev returning True.
        # The function must still return False.
        with _patch_fast_path(
            provider="xiaomi",
            model="mimo-v2.5",
            supports_vision_lookup=True,     # models.dev says vision=True
            supports_vision_override=None,   # no explicit config override
        ):
            result = _should_use_native_vision_fast_path()

        assert result is False, (
            "xiaomi/mimo-v2.5 declares supports_vision_tool_messages=False; "
            "native fast path must be disabled to prevent hallucinations."
        )

    def test_xiaomi_alias_mimo_returns_false(self):
        """Alias 'mimo' resolves to the same profile with False."""
        from tools.vision_tools import _should_use_native_vision_fast_path

        with _patch_fast_path(
            provider="mimo",
            model="mimo-v2.5",
            supports_vision_lookup=True,
            supports_vision_override=None,
        ):
            result = _should_use_native_vision_fast_path()

        assert result is False

    def test_models_dev_vision_true_without_tool_result_support_returns_false(self):
        """models.dev vision=True alone is NOT sufficient to enable native path."""
        from tools.vision_tools import _should_use_native_vision_fast_path

        # Any provider where tool_result_support=False and no config override
        with _patch_fast_path(
            provider="some-new-provider",
            model="vision-model-v1",
            tool_result_support=False,
            supports_vision_lookup=True,   # models.dev says True
            supports_vision_override=None, # no config override
        ):
            result = _should_use_native_vision_fast_path()

        assert result is False


# ---------------------------------------------------------------------------
# Escape hatch: explicit config override must still work
# ---------------------------------------------------------------------------


class TestExplicitConfigOverrideEscapeHatch:
    """model.supports_vision: true in config.yaml is the deliberate escape hatch."""

    def test_explicit_override_true_enables_native_path(self):
        """When user sets model.supports_vision: true, native path is allowed."""
        from tools.vision_tools import _should_use_native_vision_fast_path

        # _supports_media_in_tool_results=False, lookup=True, explicit override=True
        with _patch_fast_path(
            provider="my-custom-vllm",
            model="llava-1.6",
            tool_result_support=False,
            supports_vision_lookup=True,
            supports_vision_override=True,  # user explicitly declared this
        ):
            result = _should_use_native_vision_fast_path()

        assert result is True, (
            "Explicit model.supports_vision: true config override must enable "
            "the native fast path even for unknown providers."
        )

    def test_no_override_no_tool_result_support_returns_false(self):
        """No override + no tool result support → False (safe default)."""
        from tools.vision_tools import _should_use_native_vision_fast_path

        with _patch_fast_path(
            provider="unknown-provider",
            model="some-vision-model",
            tool_result_support=False,
            supports_vision_lookup=None,    # models.dev unknown
            supports_vision_override=None,  # no config override
        ):
            result = _should_use_native_vision_fast_path()

        assert result is False


# ---------------------------------------------------------------------------
# Allowlisted providers still work
# ---------------------------------------------------------------------------


class TestAllowlistedProvidersUnchanged:
    """Providers in the hardcoded allowlist must continue to use native path."""

    @pytest.mark.parametrize("provider", ["anthropic", "openai", "openrouter"])
    def test_known_providers_return_true(self, provider):
        from tools.vision_tools import _should_use_native_vision_fast_path

        # Let the real _supports_media_in_tool_results run (reads the allowlist)
        with _patch_fast_path(provider=provider, model="vision-capable-model"):
            result = _should_use_native_vision_fast_path()

        assert result is True, (
            f"Provider '{provider}' is in the allowlist and should use "
            "the native fast path."
        )


# ---------------------------------------------------------------------------
# text mode short-circuit
# ---------------------------------------------------------------------------


class TestImageModeTextShortCircuit:
    def test_text_mode_returns_false_immediately(self):
        """When decide_image_input_mode returns 'text', always return False."""
        from tools.vision_tools import _should_use_native_vision_fast_path

        with _patch_fast_path(
            provider="anthropic",
            model="claude-opus-4",
            image_mode="text",
        ):
            result = _should_use_native_vision_fast_path()

        assert result is False


# ---------------------------------------------------------------------------
# Exception safety
# ---------------------------------------------------------------------------


class TestExceptionSafety:
    def test_exception_returns_false(self):
        """Any unexpected error returns False (safe fallback)."""
        from tools.vision_tools import _should_use_native_vision_fast_path

        with patch(
            "agent.auxiliary_client._read_main_provider",
            side_effect=RuntimeError("unexpected"),
        ):
            result = _should_use_native_vision_fast_path()

        assert result is False
