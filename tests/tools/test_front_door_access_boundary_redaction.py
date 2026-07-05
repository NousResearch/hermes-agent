"""Task 11 front-door/provider evidence redaction contract tests."""

from __future__ import annotations

import json

import pytest


PROVIDER_EVIDENCE_FIXTURES = {
    "set_cookie": "Set-Cookie: CF_Authorization=fixture_cf_access_cookie_value; Path=/; Secure; HttpOnly",
    "cookie": "Cookie: CF_Authorization=fixture_cf_cookie_value; session=fixture_session_cookie_value",
    "authorization": "Authorization: Bearer fixture_authorization_bearer_value",
    "cf_signed_redirect": (
        "https://access.example.invalid/cdn-cgi/access/login/google?"
        "kid=fixture-key&redirect_url=https%3A%2F%2Fcrm.example.invalid%2Fdashboard"
        "&sig=fixture-cloudflare-access-signature"
    ),
    "magic_link": (
        "https://crm.example.invalid/login?magic_link_token=fixture_magic_link_token"
        "&next=%2Fdashboard"
    ),
    "jwt_meta": "meta=eyJmaXh0dXJlIjoiand0LWhlYWRlciJ9.eyJzdWIiOiJmaXh0dXJlIn0.signatureFixture",
    "private_provider_redirect": (
        "https://provider.example.invalid/oauth/callback?"
        "private_redirect_url=https%3A%2F%2Fcrm.internal.invalid%2Fadmin"
        "&provider_token=fixture_provider_redirect_token"
    ),
}


def _serialized(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _assert_no_raw_provider_material(value: object) -> None:
    haystack = _serialized(value)
    for label, fixture in PROVIDER_EVIDENCE_FIXTURES.items():
        if fixture in haystack:
            pytest.fail(f"unredacted provider fixture leaked: {label}")
    forbidden_fragments = {
        "cf_cookie_value": "Cloudflare Access cookie fragment",
        "session_cookie_value": "session cookie fragment",
        "authorization_bearer_value": "Authorization value fragment",
        "cloudflare-access-signature": "Cloudflare signed URL fragment",
        "magic_link_token": "magic-link query fragment",
        "provider_redirect_token": "provider redirect token fragment",
        "crm.internal.invalid": "private provider redirect host",
    }
    for fragment, label in forbidden_fragments.items():
        if fragment in haystack:
            pytest.fail(f"unredacted provider fixture fragment leaked: {label}")


def test_terminal_provider_evidence_is_redacted_before_return_and_evidence_summary(
    tmp_path, monkeypatch
):
    from agent.verification_evidence import record_terminal_result
    from agent.redact import redact_terminal_output

    raw_output = "\n".join(PROVIDER_EVIDENCE_FIXTURES.values())

    returned_output = redact_terminal_output(raw_output, "printf fixture-provider-evidence", force=True)
    _assert_no_raw_provider_material({"output": returned_output})

    project = tmp_path / "project"
    project.mkdir()
    (project / "package.json").write_text(
        '{"scripts":{"test":"node fixture-test.js"}}\n', encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    evidence = record_terminal_result(
        command="npm test",
        cwd=project,
        session_id="task-11-redaction-fixture",
        exit_code=1,
        output=returned_output,
    )

    assert evidence is not None
    _assert_no_raw_provider_material({"output_summary": evidence["output_summary"]})


def test_browser_and_cdp_provider_evidence_is_redacted_before_tool_return():
    from tools.browser_tool import _redact_browser_output
    from tools.browser_cdp_tool import _redact_cdp_output

    raw_browser_payload = {
        "snapshot": "\n".join(PROVIDER_EVIDENCE_FIXTURES.values()),
        "console_messages": [
            {"type": "log", "text": PROVIDER_EVIDENCE_FIXTURES["authorization"]},
            {"type": "warn", "text": PROVIDER_EVIDENCE_FIXTURES["cf_signed_redirect"]},
        ],
        "meta": PROVIDER_EVIDENCE_FIXTURES["jwt_meta"],
    }

    _assert_no_raw_provider_material(_redact_browser_output(raw_browser_payload))
    _assert_no_raw_provider_material(_redact_cdp_output(raw_browser_payload))


def test_front_door_access_boundary_evidence_is_value_free():
    from tools.browser_tool import build_front_door_access_boundary_evidence

    evidence = build_front_door_access_boundary_evidence(
        status_code=302,
        expected_auth_provider="cloudflare_access",
        title="Cloudflare Access",
        url="https://crm.example.invalid/",
        final_url=PROVIDER_EVIDENCE_FIXTURES["cf_signed_redirect"],
        headers={
            "Set-Cookie": PROVIDER_EVIDENCE_FIXTURES["set_cookie"],
            "Cookie": PROVIDER_EVIDENCE_FIXTURES["cookie"],
            "Authorization": PROVIDER_EVIDENCE_FIXTURES["authorization"],
            "Location": PROVIDER_EVIDENCE_FIXTURES["private_provider_redirect"],
        },
        body_text=(
            "Sign in with Google to continue. "
            + PROVIDER_EVIDENCE_FIXTURES["magic_link"]
            + " "
            + PROVIDER_EVIDENCE_FIXTURES["jwt_meta"]
        ),
    )

    assert evidence["status_class"] == "3xx"
    assert evidence["expected_auth_provider"] == "cloudflare_access"
    assert evidence["boundary_result"] == "access_boundary"
    assert evidence["coarse_title"] == "Cloudflare Access"
    assert "headers" not in evidence
    assert "cookies" not in evidence
    assert "url" not in evidence
    assert "final_url" not in evidence
    assert evidence.get("authenticated_ui_acceptance") is False
    _assert_no_raw_provider_material(evidence)
