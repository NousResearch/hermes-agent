"""Tests: ``profiles.list`` rows satisfy the DECLARED wire contract (``ProfilesListResult``).

Why: the desktop consumes ``previous_names`` (``hermes-bots/types.ts``, ``group-membership.ts``,
``data.ts``) to re-seat a renamed teammate's persisted handle and group membership, and
``tui_gateway/methods_profiles.py`` has emitted it since the Bot Mode rename sync landed — but
``ProfileRow`` never declared the field. The model is ``extra="forbid"``, so every roster sweep
answered with an extra input, logged once per session as
``result of 'profiles.list' violates its wire contract: … profiles.0.previous_names
Extra inputs are not permitted``, and the GENERATED TypeScript never carried the field.

Contract under test:
- A profile carrying rename history reaches the wire as ``previous_names``.
- A profile without one answers ``[]`` (the key is always present, never missing).
- The whole result validates against the declared model — the same validator the runtime contract
  check uses (``tui_gateway/contracts/registry.py::check_result``).
"""

from __future__ import annotations

import yaml

import tui_gateway.server as srv
from hermes_cli.profiles import write_profile_meta
from tui_gateway.contracts.profiles_vault_complete_foreign_subagents import ProfilesListResult


def _home(tmp_path, monkeypatch):
    """Temp HERMES_HOME with the default profile plus one named profile."""
    h = tmp_path / ".hermes"
    named = h / "profiles" / "ops"
    named.mkdir(parents=True)
    # profile.yaml is the identity marker that makes the directory a served profile.
    (named / "profile.yaml").write_text(yaml.safe_dump({"name": "ops"}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


def _rows(params: dict | None = None) -> dict:
    return srv._methods["profiles.list"](1, params or {})["result"]


def _row(profiles: list[dict], name: str) -> dict:
    return next(p for p in profiles if p["name"] == name)


def test_rename_history_is_declared_and_validates(tmp_path, monkeypatch):
    home = _home(tmp_path, monkeypatch)
    write_profile_meta(home / "profiles" / "ops", previous_names=["oldops"])

    result = _rows()

    assert _row(result["profiles"], "ops")["previous_names"] == ["oldops"]
    declared = ProfilesListResult.model_validate(result)
    assert next(p.previous_names for p in declared.profiles if p.name == "ops") == ["oldops"]


def test_profile_without_rename_history_answers_an_empty_list(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)

    result = _rows()

    assert _row(result["profiles"], "ops")["previous_names"] == []
    assert ProfilesListResult.model_validate(result)
