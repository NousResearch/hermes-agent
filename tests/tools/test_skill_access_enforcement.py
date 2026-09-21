"""Phase 2: enabled_skills allowlist enforcement in skill_view.

Covers the off/shadow/enforce decision matrix, always_skills inclusion, and the
active-grant unblock path. Unit-tests the pure decision function so it does not
depend on a live ledger or profile config on disk.
"""
import pytest

import agent.skill_utils as su
import tools.skills_tool as st


@pytest.fixture
def patch(monkeypatch):
    def _set(mode, enabled, granted=False):
        monkeypatch.setattr(su, "get_skill_enforcement_mode", lambda: mode)
        monkeypatch.setattr(su, "get_enabled_skill_names", lambda: enabled)
        monkeypatch.setattr(st, "_has_active_grant", lambda profile, skill: granted)
        monkeypatch.setattr(st, "_current_profile", lambda: "octacon")
    return _set


def test_off_always_allows(patch):
    patch("off", {"arxiv"})
    assert st._skill_access_decision("anything-unlisted") == "allow"


def test_enforce_blocks_unlisted(patch):
    patch("enforce", {"arxiv"})
    assert st._skill_access_decision("comfyui") == "block"


def test_enforce_allows_listed(patch):
    patch("enforce", {"arxiv", "comfyui"})
    assert st._skill_access_decision("comfyui") == "allow"


def test_shadow_logs_but_allows(patch):
    patch("shadow", {"arxiv"})
    assert st._skill_access_decision("comfyui") == "shadow_block"


def test_active_grant_unblocks_enforce(patch):
    patch("enforce", {"arxiv"}, granted=True)
    assert st._skill_access_decision("comfyui") == "allow"


def test_no_allowlist_configured_is_unrestricted(patch):
    # enabled_skills absent (None) → not seeded yet → never block
    patch("enforce", None)
    assert st._skill_access_decision("comfyui") == "allow"


def test_enforcement_mode_defaults_off(monkeypatch, tmp_path):
    # No skills.enforcement_mode anywhere → defaults to "off"
    monkeypatch.setattr(su, "get_config_path", lambda: tmp_path / "missing.yaml")
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly", lambda: {"skills": {}}
    )
    assert su.get_skill_enforcement_mode() == "off"


def test_enabled_includes_always_skills(monkeypatch, tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("skills:\n  always_skills: [kanban-worker]\n  enabled_skills: [arxiv]\n")
    monkeypatch.setattr(su, "get_config_path", lambda: cfg)
    enabled = su.get_enabled_skill_names()
    assert enabled == {"arxiv", "kanban-worker"}


def test_active_grant_reconciles_borrow_and_revoke(monkeypatch):
    """borrow with a matching revoke = no live grant; an unrevoked borrow = grant."""
    import tools.skill_grants as _sg

    def fake_query(*, event_types, target_profile=None, object_id=None, **_):
        if "skill.borrowed" in event_types:
            return [{"event_id": "b1"}, {"event_id": "b2"}]
        if "skill.revoked" in event_types:
            return [{"payload": {"borrow_event_id": "b1"}}]  # only b1 revoked
        return []

    monkeypatch.setattr(_sg, "query_events", fake_query)
    # b2 is still live → grant active
    assert st._has_active_grant("octacon", "comfyui") is True


def test_active_grant_false_when_all_revoked(monkeypatch):
    import tools.skill_grants as _sg

    def fake_query(*, event_types, **_):
        if "skill.borrowed" in event_types:
            return [{"event_id": "b1"}]
        if "skill.revoked" in event_types:
            return [{"payload": {"borrow_event_id": "b1"}}]
        return []

    monkeypatch.setattr(_sg, "query_events", fake_query)
    assert st._has_active_grant("octacon", "comfyui") is False


def test_active_grant_fails_closed_on_error(monkeypatch):
    import tools.skill_grants as _sg

    def boom(**_):
        raise RuntimeError("ledger down")

    monkeypatch.setattr(_sg, "query_events", boom)
    assert st._has_active_grant("octacon", "comfyui") is False
