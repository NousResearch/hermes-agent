#!/usr/bin/env python3
"""Per-task model routing on delegate_task: provider choice for a model-only pin and typo correction.

A model named without a provider runs on plugins first, then first-party providers, then aggregators; a clear
typo of a catalog id is corrected; every such decision is reported in route_notes.
"""

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from tools import delegate_tool_routes as routes
from tools.delegate_tool import delegate_task
from tools.delegate_tool_routes import (
    TaskRouteCatalog, closest_model_id, family_model_id, prefix_model_id, resolve_task_model,
)

ROWS = [
    {"slug": "nous", "models": ["openai/gpt-6.1-sol", "openai/gpt-6-astra", "anthropic/claude-opus-5.5",
                                 "stepfun/step-5-preview:free", "anthropic/claude-haiku-4.5"]},
    {"slug": "openrouter", "models": ["openrouter/pareto-code", "openai/gpt-6.1-sol"]},
    {"slug": "copilot", "models": ["gpt-6.1-sol", "claude-opus-5.5"]},
    {"slug": "openai-codex", "models": ["gpt-6.1-sol", "gpt-6.1-sol-900k", "gpt-6-sol", "gpt-6-astra", "gpt-6-astra-900k"]},
    {"slug": "claude-native-bridge", "models": ["claude-opus-5-5", "claude-opus-5", "claude-haiku-4-5-20251001",
                                                 "claude-haiku-5-5"]},
]
PLUGINS = {"claude-native-bridge"}


@pytest.fixture(autouse=True)
def _catalog(monkeypatch):
    monkeypatch.setattr(routes, "_load_rows", lambda current: ROWS)
    monkeypatch.setattr(routes, "_is_plugin", lambda slug: slug in PLUGINS)


def _resolve(provider, model, inherited="openrouter"):
    return resolve_task_model(provider, model, TaskRouteCatalog((inherited,)))


def test_model_only_prefers_first_party_over_aggregators():
    provider, model, note = _resolve(None, "gpt-6.1-sol")
    assert (provider, model) == ("openai-codex", "gpt-6.1-sol")
    assert "openai-codex" in note


def test_model_only_prefers_plugin_provider():
    assert _resolve(None, "claude-opus-5-5")[:2] == ("claude-native-bridge", "claude-opus-5-5")


def test_inherited_provider_serving_the_model_stays_unpinned():
    assert _resolve(None, "gpt-6-sol", inherited="openai-codex") == (None, "gpt-6-sol", None)


def test_named_aggregator_maps_bare_id_to_its_catalog_id():
    provider, model, note = _resolve("nous", "gpt-6.1-sol")
    assert (provider, model) == ("nous", "openai/gpt-6.1-sol")
    assert "openai/gpt-6.1-sol" in note


def test_typo_is_corrected_and_reported():
    provider, model, note = _resolve("openai-codex", "gpt-6-astr")
    assert (provider, model) == ("openai-codex", "gpt-6-astra")
    assert "gpt-6-astr" in note and "typo correction" in note


def test_model_only_typo_finds_provider_and_corrects():
    provider, model, note = _resolve(None, "gpt-6-astr")
    assert (provider, model) == ("openai-codex", "gpt-6-astra")
    assert "typo correction" in note


@pytest.mark.parametrize("requested", ["gpt-6.2-sol", "gpt-7-astra"])
def test_different_versions_are_never_rewritten(requested):
    with pytest.raises(ValueError, match=f"'{requested}' is not in the model picker"):
        _resolve("openai-codex", requested)


def test_variant_of_a_listed_model_passes_through():
    assert _resolve("openai-codex", "gpt-6-astra:nitro") == ("openai-codex", "gpt-6-astra:nitro", None)


@pytest.mark.parametrize("requested,expected", [
    ("astra", "gpt-6-astra"), ("atra", "gpt-6-astra"), ("ASTRA", "gpt-6-astra"), ("sol", "gpt-6.1-sol"),
])
def test_family_name_picks_newest_standard_id(requested, expected):
    provider, model, note = _resolve("openai-codex", requested)
    assert (provider, model) == ("openai-codex", expected)
    assert requested in note


def test_model_only_family_name_finds_first_party_provider():
    assert _resolve(None, "atra")[:2] == ("openai-codex", "gpt-6-astra")


def test_family_name_on_aggregator_uses_its_catalog_id():
    assert _resolve("nous", "sol")[:2] == ("nous", "openai/gpt-6.1-sol")


def test_family_name_tied_between_families_is_not_guessed():
    assert family_model_id("lun", ["gpt-6-luna", "gpt-6-lune"]) is None


def test_family_name_exact_beats_near_miss():
    assert family_model_id("sol", ["upstage/solar-pro4", "openai/gpt-6.1-sol"]) == "openai/gpt-6.1-sol"


def test_leading_name_matches_the_one_listed_id():
    provider, model, note = _resolve(None, "step-5")
    assert (provider, model) == ("nous", "stepfun/step-5-preview:free")
    assert "step-5" in note


def test_leading_name_prefers_plain_id_over_variant():
    assert prefix_model_id("step-5", ["step-5-preview", "step-5-preview:free"]) == "step-5-preview"


@pytest.mark.parametrize("requested,catalog", [
    ("gpt-6", ["gpt-6-astra", "gpt-6-sol"]),          # leads two models
    ("gpt-6", ["gpt-6.1-sol"]),                       # '.' is not a boundary; 6 != 6.1
    ("step-5", ["step-5.1-preview"]),                 # different version numbers
])
def test_leading_name_never_guesses(requested, catalog):
    assert prefix_model_id(requested, catalog) is None


@pytest.mark.parametrize("requested", ["flash", "mini", "pro", "fash", "gemini", "gpt"])
def test_tier_and_vendor_words_are_not_families(requested):
    catalog = ["gemini-3.8-flash", "gemini-4-argon", "gpt-5.4-mini", "gpt-6-sol-pro", "gpt-6-sol"]
    assert family_model_id(requested, catalog) is None


def test_model_only_tier_word_fails_the_spawn():
    with pytest.raises(ValueError, match="not in the model picker"):
        _resolve(None, "flash")


def test_codex_is_a_provider_word():
    provider, model, note = _resolve("codex", "sol")
    assert (provider, model) == ("openai-codex", "gpt-6.1-sol")
    assert "'codex' -> 'openai-codex'" in note


@pytest.mark.parametrize("word", ["codex", "openai_codex", "OpenAI-Codex"])
def test_provider_profile_aliases_name_the_provider(word):
    assert _resolve(word, "gpt-6-sol")[:2] == ("openai-codex", "gpt-6-sol")


def test_provider_sent_as_model_is_refused_not_fuzzed():
    """'codex' looks like 'pareto-code'; a provider word must never be typo-corrected into a model."""
    with pytest.raises(ValueError, match="'codex' is a provider, not a model.*openai-codex.*astra"):
        _resolve(None, "codex")


def test_provider_sent_as_model_never_matches_a_family():
    rows = [*ROWS, {"slug": "copilot", "models": ["gpt-5.3-codex"]}]
    with patch.object(routes, "_load_rows", lambda current: rows):
        with pytest.raises(ValueError, match="'codex' is a provider, not a model"):
            _resolve(None, "codex")


def test_model_sent_as_provider_is_used_as_the_model():
    provider, model, note = _resolve("astra", None)
    assert (provider, model) == ("openai-codex", "gpt-6-astra")
    assert "names a model" in note


def test_unknown_provider_with_model_still_fails():
    with pytest.raises(ValueError, match="Provider 'astra' is not in the model picker"):
        _resolve("astra", "gpt-6-astra")


def test_version_spelling_and_date_suffix_match_the_plugin_id():
    provider, model, note = _resolve(None, "claude-haiku-4-5")
    assert (provider, model) == ("claude-native-bridge", "claude-haiku-4-5-20251001")
    assert "different spelling" in note
    assert _resolve("nous", "claude-haiku-4-5")[1] == "anthropic/claude-haiku-4.5"


def test_bare_family_beats_older_guess():
    assert _resolve(None, "haiku")[:2] == ("claude-native-bridge", "claude-haiku-5-5")


def test_rule_match_on_any_provider_beats_a_typo_on_an_earlier_one():
    """Steps run across every provider before the next step: a fixed-rule hit wins over a fuzzy hit elsewhere."""
    rows = [{"slug": "claude-native-bridge", "models": ["sola-1"]},
            {"slug": "openai-codex", "models": ["gpt-6-sol"]}]
    with patch.object(routes, "_load_rows", lambda current: rows):
        assert _resolve(None, "sol")[:2] == ("openai-codex", "gpt-6-sol")


def test_refusal_lists_provider_families():
    with pytest.raises(ValueError, match=r"openai-codex \(astra, sol\)"):
        _resolve("openai-codex", "zzzz")


def test_ambiguous_near_miss_is_not_guessed():
    assert closest_model_id("gpt-6-lun", ["gpt-6-luna", "gpt-6-lunb"]) is None


def test_unlisted_model_fails_with_similar_ids():
    with pytest.raises(ValueError, match="not in the model picker.*Similar: gpt-6.1-sol"):
        _resolve("openai-codex", "gpt-6.2-sol")
    with pytest.raises(ValueError, match="for any logged-in provider"):
        _resolve(None, "some-local-model")


def test_ambiguous_near_miss_fails_listing_both():
    rows = [{"slug": "openai-codex", "models": ["gpt-6-luna", "gpt-6-lunb"]}]
    with patch.object(routes, "_load_rows", lambda current: rows):
        with pytest.raises(ValueError, match="gpt-6-luna.*gpt-6-lunb|gpt-6-lunb.*gpt-6-luna"):
            _resolve("openai-codex", "gpt-6-lun")


def test_provider_missing_from_picker_fails():
    with pytest.raises(ValueError, match="not in the model picker"):
        _resolve("xai", "grok-5")


def test_unreadable_picker_lets_the_spawn_proceed():
    with patch.object(routes, "_load_rows", lambda current: []):
        assert _resolve("openai-codex", "anything") == ("openai-codex", "anything", None)


def _make_mock_parent():
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "***"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.request_overrides = {}
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    return parent


def _creds(cfg, _parent):
    provider = str((cfg or {}).get("provider") or "").strip()
    if provider == "openai-codex":
        return {"provider": "openai-codex", "model": cfg.get("model"), "base_url": "https://chatgpt.com/backend-api/codex",
                "api_key": "codex-key", "api_mode": "codex_responses", "request_overrides": {}}
    return {"model": None, "provider": None, "base_url": None, "api_key": None, "api_mode": None,
            "request_overrides": None}


def _spawn(tasks, load_rows=None):
    with (
        patch("tools.delegate_tool._load_config", return_value={}),
        patch("tools.delegate_tool._resolve_delegation_credentials", side_effect=_creds),
        patch.object(routes, "_load_rows", load_rows or (lambda current: ROWS)),
        patch("run_agent.AIAgent", return_value=MagicMock()) as agent,
        patch("tools.delegate_tool._run_batch", return_value='{"status":"built"}'),
    ):
        result = json.loads(delegate_task(tasks=tasks, parent_agent=_make_mock_parent()))
    return result, [call.kwargs for call in agent.call_args_list]


def test_sol_fast_spawns_on_codex_and_reports_the_route():
    result, (child,) = _spawn([{"goal": "Review the PR diff for regressions", "model": "gpt-6.1-sol",
                                "service_tier": "fast"}])
    assert (child["provider"], child["model"]) == ("openai-codex", "gpt-6.1-sol")
    assert child["request_overrides"]["service_tier"] == "priority"
    assert result["status"] == "built"
    assert len(result["route_notes"]) == 1 and "openai-codex" in result["route_notes"][0]


def test_unpinned_batch_never_lists_providers():
    load_rows = MagicMock(return_value=ROWS)
    result, built = _spawn([{"goal": "Summarize the PR discussion thread"}], load_rows=load_rows)
    assert result == {"status": "built"} and len(built) == 1
    load_rows.assert_not_called()
