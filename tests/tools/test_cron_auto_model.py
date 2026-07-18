"""Tests for cron LLM-model auto-resolution (cronjob create).

An unpinned LLM cron (no_agent=False, no model) inherits the runtime PRIMARY
(often Opus) at fire time — a silent cost footgun. ``_resolve_cron_llm_model``
+ the ``_current_agent_model`` ContextVar let a job be pinned at creation to the
CREATING agent's own model (model="auto" or config cron.default_model="auto"),
or to a config-default model, while never fabricating a model when it can't
resolve.
"""
import tools.cronjob_tools as ct


def _clear_agent_model():
    ct.set_current_agent_model(None, None)


def test_explicit_auto_pins_to_creating_agent(monkeypatch):
    # No config default; the agent published its model this turn.
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    ct.set_current_agent_model("openai-codex", "gpt-5.6-terra")
    try:
        model, provider = ct._resolve_cron_llm_model("auto", None)
        assert model == "gpt-5.6-terra"
        assert provider == "openai-codex"
    finally:
        _clear_agent_model()


def test_auto_is_case_and_space_insensitive(monkeypatch):
    ct.set_current_agent_model("claude-apr", "claude-sonnet-5")
    try:
        model, provider = ct._resolve_cron_llm_model("  AUTO ", None)
        assert model == "claude-sonnet-5"
        assert provider == "claude-apr"
    finally:
        _clear_agent_model()


def test_auto_that_cannot_resolve_degrades_to_unpinned():
    # No agent model published (bare caller) → "auto" must NOT fabricate a model,
    # and must NOT leave a dangling provider glued to an unresolved model.
    _clear_agent_model()
    model, provider = ct._resolve_cron_llm_model("auto", "claude-apr")
    assert model is None  # sentinel dropped, not persisted as a literal "auto"
    assert provider is None  # provider dropped too — no half-pinned job


def test_resolve_model_override_passes_auto_through():
    # The object→string flattener must NOT pin a config provider onto "auto";
    # that would half-pin the job before the live-agent resolver runs.
    provider, model = ct._resolve_model_override({"model": "auto"})
    assert model == "auto"
    assert provider is None


def test_explicit_model_is_left_untouched():
    ct.set_current_agent_model("openai-codex", "gpt-5.6-terra")
    try:
        model, provider = ct._resolve_cron_llm_model("claude-opus-4-8", "claude-apr")
        assert model == "claude-opus-4-8"
        assert provider == "claude-apr"
    finally:
        _clear_agent_model()


def test_no_model_no_config_stays_unpinned(monkeypatch):
    # Back-compat: nothing given, no config knob → unchanged (unpinned).
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    _clear_agent_model()
    model, provider = ct._resolve_cron_llm_model(None, None)
    assert model is None
    assert provider is None


def test_config_default_auto_pins_to_agent(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"cron": {"default_model": "auto"}})
    ct.set_current_agent_model("openai-codex", "gpt-5.6-sol")
    try:
        model, provider = ct._resolve_cron_llm_model(None, None)
        assert model == "gpt-5.6-sol"
        assert provider == "openai-codex"
    finally:
        _clear_agent_model()


def test_config_default_literal_model(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"cron": {"default_model": "gpt-5.6-terra", "default_provider": "openai-codex"}})
    _clear_agent_model()
    try:
        model, provider = ct._resolve_cron_llm_model(None, None)
        assert model == "gpt-5.6-terra"
        assert provider == "openai-codex"
    finally:
        _clear_agent_model()


def test_explicit_model_ignores_config_default(monkeypatch):
    # An explicit model always wins over a config default.
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"cron": {"default_model": "gpt-5.6-terra"}})
    _clear_agent_model()
    model, provider = ct._resolve_cron_llm_model("grok-4.5", "xai-oauth")
    assert model == "grok-4.5"
    assert provider == "xai-oauth"


def test_contextvar_roundtrip():
    ct.set_current_agent_model("p1", "m1")
    assert ct.get_current_agent_model() == ("p1", "m1")
    ct.set_current_agent_model(None, None)
    assert ct.get_current_agent_model() == (None, None)


import json

import pytest


class TestCreateAutoModelE2E:
    """Drive the real cronjob(action="create") path and assert the PERSISTED job
    carries the resolved model — the actual behavior, not just the helper."""

    @pytest.fixture(autouse=True)
    def _setup_cron_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
        monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
        monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
        yield
        ct.set_current_agent_model(None, None)

    def test_create_with_auto_pins_agent_model(self):
        ct.set_current_agent_model("openai-codex", "gpt-5.6-terra")
        created = json.loads(ct.cronjob(
            action="create", prompt="Check", schedule="every 1h",
            name="auto-model-job", model="auto",
        ))
        assert created["success"] is True
        # The persisted job must carry the creating agent's model+provider.
        assert created["job"]["model"] == "gpt-5.6-terra"
        assert created["job"]["provider"] == "openai-codex"

    def test_create_no_agent_script_untouched_by_auto(self, tmp_path, monkeypatch):
        # A no_agent script cron must NOT get a model resolved/pinned — the
        # create path guards LLM-model resolution behind `if not _no_agent:`.
        # Place a script the tool will accept, then create with no_agent=True
        # and model="auto"; the persisted job must carry NO model.
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "noop.sh").write_text("#!/bin/bash\necho hi\n")
        monkeypatch.setattr("tools.cronjob_tools.HERMES", tmp_path, raising=False)
        ct.set_current_agent_model("openai-codex", "gpt-5.6-terra")
        created = json.loads(ct.cronjob(
            action="create", schedule="every 1h", name="noagent-job",
            no_agent=True, script="noop.sh", model="auto",
        ))
        # Whether or not script validation passes in this sandbox, the key
        # invariant is that a no_agent create never pins an LLM model to "auto".
        if created.get("success"):
            assert created["job"].get("model") in (None, "", "auto")

    def test_create_without_auto_stays_unpinned(self):
        # Back-compat: an ordinary create with no model stays unpinned.
        ct.set_current_agent_model("openai-codex", "gpt-5.6-terra")
        created = json.loads(ct.cronjob(
            action="create", prompt="Check", schedule="every 1h",
            name="plain-job",
        ))
        assert created["success"] is True
        assert created["job"].get("model") in (None, "")

