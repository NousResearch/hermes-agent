"""Regression tests for _repair_aggregator_model_id.

The desktop composer persists its model pick as sticky UI state and ships it
verbatim on every ``session.create``. A pick made while the profile ran a
direct provider (anthropic's ``claude-opus-4-8``) survives a later switch to
an aggregator (nous/openrouter), whose catalog serves vendor-slugged dot-form
ids (``anthropic/claude-opus-4.8``) — every turn then fails with HTTP 404
"Model 'claude-opus-4-8' not found". _make_agent repairs the id by
canonical-form matching against the configured default + curated catalog.
"""

from unittest.mock import patch

from tui_gateway.server import _repair_aggregator_model_id

_CFG = {"model": {"default": "anthropic/claude-opus-4.8", "provider": "nous"}}
_CURATED = [
    "anthropic/claude-opus-4.8",
    "anthropic/claude-sonnet-5",
    "openai/gpt-5.5",
]


def _patched(fn):
    return (
        patch("tui_gateway.server._load_cfg", return_value=_CFG),
        patch(
            "hermes_cli.models.get_curated_nous_model_ids",
            return_value=list(_CURATED),
        ),
    )


def test_repairs_bare_hyphen_form_for_nous():
    cfg_patch, curated_patch = _patched(None)
    with cfg_patch, curated_patch:
        assert (
            _repair_aggregator_model_id("claude-opus-4-8", "nous")
            == "anthropic/claude-opus-4.8"
        )


def test_repairs_slugged_hyphen_form_for_nous():
    cfg_patch, curated_patch = _patched(None)
    with cfg_patch, curated_patch:
        assert (
            _repair_aggregator_model_id("anthropic/claude-opus-4-8", "nous")
            == "anthropic/claude-opus-4.8"
        )


def test_exact_catalog_id_passes_through():
    cfg_patch, curated_patch = _patched(None)
    with cfg_patch, curated_patch:
        assert (
            _repair_aggregator_model_id("anthropic/claude-opus-4.8", "nous")
            == "anthropic/claude-opus-4.8"
        )


def test_unknown_id_untouched():
    cfg_patch, curated_patch = _patched(None)
    with cfg_patch, curated_patch:
        assert (
            _repair_aggregator_model_id("nousresearch/hermes-4-405b", "nous")
            == "nousresearch/hermes-4-405b"
        )


def test_direct_provider_untouched():
    cfg_patch, curated_patch = _patched(None)
    with cfg_patch, curated_patch:
        assert (
            _repair_aggregator_model_id("claude-opus-4-8", "anthropic")
            == "claude-opus-4-8"
        )


def test_empty_model_or_provider_untouched():
    assert _repair_aggregator_model_id("", "nous") == ""
    assert _repair_aggregator_model_id("claude-opus-4-8", None) == "claude-opus-4-8"


def test_openrouter_repairs_via_config_default():
    cfg_patch, curated_patch = _patched(None)
    with cfg_patch, curated_patch:
        # openrouter has no curated-list hook here; the configured default
        # is still consulted as a candidate.
        assert (
            _repair_aggregator_model_id("claude-opus-4-8", "openrouter")
            == "anthropic/claude-opus-4.8"
        )


def test_config_failure_is_nonfatal():
    with (
        patch("tui_gateway.server._load_cfg", side_effect=RuntimeError("boom")),
        patch(
            "hermes_cli.models.get_curated_nous_model_ids",
            return_value=list(_CURATED),
        ),
    ):
        assert (
            _repair_aggregator_model_id("claude-opus-4-8", "nous")
            == "anthropic/claude-opus-4.8"
        )
