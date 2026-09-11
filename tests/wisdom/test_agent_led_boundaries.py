"""Exercise model egress and setup boundaries through the real flow APIs."""

import json

import pytest

from hermes_wisdom.agent_led.agent import review_candidates
from hermes_wisdom.agent_led.install_flow import InstallFlow
from hermes_wisdom.agent_led.privacy import model_safe_data, model_safe_text
from hermes_wisdom.agent_led.policy import load_policy
from hermes_wisdom.agent_led.schemas import SchemaRejected
from hermes_wisdom.agent_led.share_flow import ShareFlow, _list_files
from hermes_wisdom.agent_led.templates import render_share


@pytest.mark.parametrize("value", [
    "token = ghp_" + "a" * 36,
    "Authorization: Bearer " + "b" * 40,
    "postgres://person:" + "p" * 24 + "@db.example/db",
    "https://example.com/api?access_token=" + "c" * 40,
])
def test_model_egress_removes_complete_credentials(value):
    assert model_safe_text(value).strip() == "[REDACTED]"


@pytest.mark.parametrize("closing", ["", "\n-----END PRIVATE KEY-----"])
def test_model_egress_removes_multiline_and_unterminated_private_keys(closing):
    value = "-----BEGIN PRIVATE KEY-----\nprivate-material" + closing
    assert "private-material" not in model_safe_text(value)


def test_model_egress_scrubs_nested_private_locations():
    result = model_safe_data({"files": [{"content": "See /Users/alice/private/config and api.corp"}]})
    assert "alice" not in json.dumps(result)
    assert "api.corp" not in json.dumps(result)


def test_real_prompt_path_uses_redacted_evidence():
    captured = []

    def call(messages, schema):
        captured.extend(messages)
        return json.dumps({"schema_version": 1, "window_days": 7, "recommendations": []})

    review_candidates({"description": "token = ghp_" + "c" * 36}, model_call=call)
    assert "c" * 36 not in json.dumps(captured)
    assert "[REDACTED]" in json.dumps(captured)


@pytest.mark.parametrize("problem", ["symlink", "unsupported", "oversized"])
def test_packaging_rejects_unsafe_source_without_model_call(tmp_path, problem):
    root = tmp_path / "skill"
    root.mkdir()
    (root / "SKILL.md").write_text("# Skill")
    if problem == "symlink":
        outside = tmp_path / "private.txt"
        outside.write_text("private")
        (root / "reference.txt").symlink_to(outside)
    elif problem == "unsupported":
        (root / "unknown.bin").write_bytes(b"binary")
    else:
        (root / "large.md").write_text("x" * 200_001)
    with pytest.raises(SchemaRejected):
        _list_files(root)


def test_packaging_input_redacts_before_model_call(tmp_path):
    root = tmp_path / "skill"
    root.mkdir()
    secret = "ghp_" + "d" * 36
    (root / "SKILL.md").write_text(f"# Skill\n{secret}\nSee /home/alice/private")
    flow = ShareFlow(root=tmp_path / "flows")
    flow.start(root)

    def call(messages, schema):
        encoded = json.dumps(messages)
        assert secret not in encoded
        assert "/home/alice" not in encoded
        raise RuntimeError("stop after observing sanitized model input")

    with pytest.raises(RuntimeError, match="observing sanitized"):
        flow.package(model_call=call)


def _install(tmp_path, package=None):
    flow = InstallFlow(root=tmp_path / "flows")
    flow.start(skill_id="skill", version=1, package=package or {"verification_step": "unused"})
    return flow


def test_verification_without_separate_authorized_runner_cannot_execute(tmp_path):
    target = tmp_path / "must-not-exist"
    flow = _install(tmp_path, {"verification_step": f"touch {target}"})
    flow.check_prerequisites()
    flow.apply(lambda _: {"installed": True})
    result = flow.verify()
    assert not target.exists()
    assert result["verification"]["approval_required"] is True
    assert result["files_installed"] is True
    assert result["installed"] is False


def test_failed_apply_cannot_be_verified_as_installed(tmp_path):
    flow = _install(tmp_path)
    flow.check_prerequisites()
    result = flow.apply(lambda _: {"installed": False, "error": "conflict"})
    assert result["step"] == "apply"
    assert result["files_installed"] is False
    with pytest.raises(RuntimeError, match="cannot verify"):
        flow.verify(runner=lambda _: (True, "not called"))


def test_manual_prerequisites_require_confirmation(tmp_path):
    flow = _install(tmp_path, {"requirements": [{"kind": "account", "name": "service"}]})
    assert flow.check_prerequisites()["step"] == "setup"
    with pytest.raises(RuntimeError, match="cannot apply"):
        flow.apply(lambda _: {"installed": True})


def test_missing_safety_service_is_not_success():
    notice = render_share(editorial_name="Skill", description="Description", count_7d=3,
                          specific_work="work", audience="team", reason="helpful")
    assert "Checks unavailable" in notice.text
    assert "✓" not in notice.text
    assert notice.actions[-1].id == "share"


def test_policy_cannot_enable_proactive_review_in_fixed_mode(monkeypatch):
    monkeypatch.setattr("hermes_wisdom.mediation.delivery_mode", lambda: "fixed")
    assert load_policy(local={"enabled": True}).enabled is False


def test_zero_recommendation_cap_remains_zero(monkeypatch):
    monkeypatch.setattr("hermes_wisdom.mediation.delivery_mode", lambda: "agent")
    assert load_policy(local={"max_candidates": 0}).max_candidates == 0
