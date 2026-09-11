"""Native review retains the user's update choice through durable consent."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_wisdom.consent import ConsentActor, WisdomConsent
from hermes_wisdom.mediation_view import advice_view, interaction_view
from tests.wisdom.test_service import InstallClient, _install_service


@pytest.fixture
def native_install(tmp_path, monkeypatch):
    class Client(InstallClient):
        identity = {"owner": "member"}
        display_org_id = "org-1"
        latest = 1
        installed = 0
        security_status = "pass"

        def __init__(self):
            super().__init__()
            self.records = []

        def version(self, skill_id, version):
            result = super().version(skill_id, version)
            result.version["version"] = version
            result.version["security_check"] = {"status": self.security_status}
            result.model_dump = lambda **_: {"version": result.version}
            return result

        def skill(self, skill_id):
            result = super().skill(skill_id)
            result.versions = [{"version": self.latest}]
            return result

        def installations(self, identity):
            return [{"skill_id": "skill-1", "installed_version": self.installed,
                     "latest_version": self.latest, "update_mode": "MANUAL",
                     "skill_state": "active", "takedown_generation": 0}]

        def record_install(self, **kwargs):
            self.records.append(kwargs)
            self.installed = kwargs["version"]
            return SimpleNamespace(effective_update_mode=kwargs["update_mode"] or "REQUIRED")

    client = Client()
    service = _install_service(monkeypatch, tmp_path, client=client)
    monkeypatch.setattr(service, "require_setup", lambda: None)
    monkeypatch.setattr("hermes_wisdom.consumption.get_skills_dir", lambda: tmp_path / "skills")
    now = [1000.0]
    actor = ConsentActor("session", "telegram", "member", "42")
    consent = WisdomConsent(service, clock=lambda: now[0])
    consent.queue.register_session(
        "org-1", session_key=actor.session_key, session_id="session",
        platform=actor.platform, actor_id=actor.actor_id, private=True,
        available=True, user_activity=True, address=actor.address,
    )
    return service, actor, now


def request(consent, actor, mode):
    return consent.request(
        "org-1", {"kind": "skill", "skill_id": "skill-1", "version": 1,
                  "update_mode": mode}, actor,
        title="Managed skill", explanation="Review the selected update policy.",
    )


@pytest.mark.parametrize("mode", [None, "MANUAL", "AUTO_WITH_NOTICE", "REQUIRED"])
@pytest.mark.parametrize("expire", [False, True])
def test_native_install_preserves_selected_policy_through_recheck_and_apply(native_install, mode, expire):
    service, actor, now = native_install
    consent = WisdomConsent(service, clock=lambda: now[0])
    shown = request(consent, actor, mode)
    assert "update_mode" in shown["facts"]
    assert shown["facts"]["update_mode"] == mode
    policy_line = next(line for line in interaction_view(shown).to_text().splitlines()
                       if line.startswith("Future updates:"))
    for expanded in (False, True):
        view = advice_view([{"interaction": shown, "advice": {
            "title": "Selected skill", "explanation": "Review this skill.", "relevance": "recommend",
        }}], checks_expanded=expanded)
        assert policy_line in view.to_text()
    assert service.store.installation("skill-1") is None
    assert service.client.records == []
    # A new controller reads the same SQLite review rather than transport memory.
    consent = WisdomConsent(service, clock=lambda: now[0])
    if expire:
        now[0] = shown["expires_at"] + 1
        assert consent.resolve("org-1", shown["id"], actor, "confirm")["state"] == "expired"
        refreshed = consent.resolve("org-1", shown["id"], actor, "recheck")
        assert refreshed["id"] != shown["id"]
        assert refreshed["facts"]["update_mode"] == mode
        shown = refreshed
    assert service.client.records == []
    assert service.store.installation("skill-1") is None
    result = consent.resolve("org-1", shown["id"], actor, "confirm")
    assert result["state"] == "completed"
    assert consent.resolve("org-1", shown["id"], actor, "confirm")["state"] == "completed"
    assert len(service.client.records) == 1
    assert service.client.records[0]["update_mode"] == mode
    installed = service.store.installation("skill-1")
    assert installed["update_mode"] == (mode or "REQUIRED")
    assert (Path(installed["target_path"]) / "SKILL.md").read_bytes() == service.client.files[0][2]


@pytest.mark.parametrize("delivery_state", ["ready", "delivered", "delivering", "delivery_uncertain"])
def test_different_policy_never_reuses_prior_pending_approval(native_install, delivery_state):
    from hermes_wisdom.client import WisdomConflict

    service, actor, now = native_install
    consent = WisdomConsent(service, clock=lambda: now[0])
    original = request(consent, actor, "MANUAL")
    with service.store.transaction() as db:
        db.execute("UPDATE wisdom_assessment SET state=? WHERE id=?",
                   (delivery_state, original["assessment_id"]))
    if delivery_state in {"delivering", "delivery_uncertain"}:
        with pytest.raises(WisdomConflict):
            request(consent, actor, "AUTO_WITH_NOTICE")
    else:
        replacement = request(consent, actor, "AUTO_WITH_NOTICE")
        assert replacement["id"] != original["id"]
        assert replacement["facts"]["update_mode"] == "AUTO_WITH_NOTICE"
        assert request(consent, actor, "AUTO_WITH_NOTICE")["id"] == replacement["id"]
        assert consent.resolve("org-1", original["id"], actor, "confirm")["state"] == "stale"
    assert service.client.records == []
    assert service.store.installation("skill-1") is None
