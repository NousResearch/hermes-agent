"""profiles.configure honours the model selection guard (#95293 remainder).

The Bots-mode editor writes a profile's default model through
``profiles.configure`` — a surface that historically bypassed the
data-policy / expensive-model selection guard every other model-switch path
enforces (``config.set model`` answers ``confirm_required`` and waits for a
``confirm_expensive_model`` resend).  A guarded pick made from the Bots
surface was therefore applied silently, with no confirm flow anywhere.

These tests pin the same handshake contract on ``profiles.configure``:

* a guarded model WITHOUT ``confirm_expensive_model`` answers
  ``confirm_required`` + ``confirm_message`` and writes NOTHING;
* the confirmed resend (``confirm_expensive_model: true``) writes;
* unguarded models keep writing exactly as before.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import hermes_cli.model_selection_guards as guards
import tui_gateway.server as srv

GUARDED_MODEL = "muse-spark-1.2-contributor"
GUARD_MESSAGE = "CONTRIBUTOR TIER: this model may train on your data."


@pytest.fixture
def home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


@pytest.fixture
def contributor_guard(monkeypatch):
    """Fire the selection guard for GUARDED_MODEL only, like the real
    data-policy guard fires for ``-contributor`` ids."""

    def fake_combined_selection_warning(model_name, **_kwargs):
        if model_name == GUARDED_MODEL:
            return SimpleNamespace(message=GUARD_MESSAGE, kind="data_policy")
        return None

    monkeypatch.setattr(guards, "combined_selection_warning", fake_combined_selection_warning)


def _configure(params):
    return srv._methods["profiles.configure"]("configure", {"name": "default", **params})["result"]


def _profile_model(home: Path):
    cfg_path = home / "config.yaml"
    if not cfg_path.is_file():
        return None
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    model_cfg = cfg.get("model") or {}
    return model_cfg.get("default")


def test_guarded_model_answers_confirm_required_and_writes_nothing(home, contributor_guard):
    result = _configure({"model": GUARDED_MODEL, "provider": "opencode-go"})

    assert result.get("confirm_required") is True
    assert GUARD_MESSAGE in (result.get("confirm_message") or "")
    # The model section is PENDING confirmation, not failed — it must not
    # poison ``ok`` (the Bots editor toasts "Some sections failed" on False).
    assert result["applied"].get("model") is not False
    assert _profile_model(home) != GUARDED_MODEL


def test_confirmed_resend_writes_the_guarded_model(home, contributor_guard):
    result = _configure(
        {
            "model": GUARDED_MODEL,
            "provider": "opencode-go",
            "confirm_expensive_model": True,
        }
    )

    assert not result.get("confirm_required")
    assert result["applied"].get("model") is True
    assert _profile_model(home) == GUARDED_MODEL


def test_unguarded_model_still_writes_without_confirmation(home, contributor_guard):
    result = _configure({"model": "hermes-4.5-405b", "provider": "nous"})

    assert not result.get("confirm_required")
    assert result["applied"].get("model") is True
    assert _profile_model(home) == "hermes-4.5-405b"


def test_other_sections_still_apply_while_model_awaits_confirmation(home, contributor_guard):
    result = _configure(
        {
            "model": GUARDED_MODEL,
            "provider": "opencode-go",
            "soul": "# SOUL\nBe kind.",
        }
    )

    assert result.get("confirm_required") is True
    assert result["applied"].get("soul") is True
    assert _profile_model(home) != GUARDED_MODEL
