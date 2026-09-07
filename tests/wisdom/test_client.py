import json
import base64
import hashlib

import pytest

from hermes_wisdom.client import (
    Draft,
    WisdomClient,
    WisdomError,
    WisdomNotFound,
    WisdomValidationError,
)
from hermes_wisdom.package import verify_content_files


def _draft(**overrides):
    value = {
        "id": "d1",
        "orgId": "o1",
        "ownerUserId": "u1",
        "slug": "my-skill",
        "draftCommit": "sha256:" + "a" * 64,
        "contentHash": "sha256:" + "b" * 64,
        "authorDescription": "Does a task.",
        "authorDescriptionHash": "sha256:" + "c" * 64,
        "state": "ready",
        "packageManifestHash": "sha256:" + "d" * 64,
        "packageManifestSchemaVersion": 1,
        "systemSpec": None,
        "scan": None,
        "scanVerdict": "pass",
        "explanation": None,
        "updatedAt": "now",
    }
    value.update(overrides)
    return value


def test_changes_requested_requires_complete_moderator_metadata():
    with pytest.raises(ValueError, match="moderator return metadata"):
        Draft.model_validate(_draft(state="changes_requested"))
    returned = Draft.model_validate(
        _draft(
            state="changes_requested",
            moderationNote="Remove the hostname.",
            moderationDeciderUserId="moderator-1",
            moderationDecidedAt="2026-08-25T00:00:00Z",
        )
    )
    assert returned.moderationNote == "Remove the hostname."


class Response:
    def __init__(self, status: int, body):
        self.status_code = status
        self._body = body
        self.content = json.dumps(body).encode() if body is not None else b""

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code < 200 or self.status_code >= 300:
            raise RuntimeError(f"HTTP {self.status_code}")


class Session:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.response

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self.response


def client(response):
    value = WisdomClient.__new__(WisdomClient)
    value.base = "https://gateway.example"
    value.timeout = 7
    value.session = Session(response)
    return value


def test_capability_uses_gateway_features_field():
    value = client(Response(200, {"features": ["personal", "org", "wisdom"]}))

    assert value.capability()["features"][-1] == "wisdom"


def _agent_led_policy(**overrides):
    return {
        "org_id": "o1",
        "usage_evidence_window_days": 7,
        "min_aggregate_invocations": 3,
        "consecutive_day_usage_counts": True,
        "repeated_edits_count": True,
        "max_recommendations_per_user_per_week": 3,
        "publication_mode": "moderated",
        "install_popularity_threshold": 10,
        "notification_defaults": {
            "skill_ready_to_share": True,
            "teammate_published": False,
            "update_available": True,
        },
        "manager_review_email_cadence": "daily",
        "not_now_suppression_days": 30,
        "version": 1,
        "updated_by_user_id": None,
        **overrides,
    }


def test_agent_led_policy_reads_member_endpoint_with_bounded_request():
    value = client(Response(200, _agent_led_policy()))
    value.identity = {"claims": {"org_id": "o1"}}
    result = value.agent_led_policy()
    assert result.notification_defaults.teammate_published is False
    assert value.session.calls == [
        (
            "GET",
            "https://gateway.example/v1/sync/wisdom/agent-led/policy",
            {
                "json": None,
                "params": None,
                "timeout": 7,
            },
        )
    ]


@pytest.mark.parametrize(
    "override",
    [
        {"org_id": "other-org"},
        {"min_aggregate_invocations": "3"},
        {"notification_defaults": {"teammate_published": True}},
        {"max_recommendations_per_user_per_week": -1},
    ],
)
def test_agent_led_policy_rejects_wrong_org_or_invalid_response(override):
    value = client(Response(200, _agent_led_policy(**override)))
    value.identity = {"claims": {"org_id": "o1"}}
    with pytest.raises(WisdomError):
        value.agent_led_policy()


@pytest.mark.parametrize("status", [401, 403, 404, 503])
def test_unavailable_agent_led_policy_defers_instead_of_enabling_defaults(
    monkeypatch, status
):
    from hermes_wisdom.agent_led.policy import load_policy

    monkeypatch.setattr("hermes_wisdom.mediation.delivery_mode", lambda: "agent")
    value = client(Response(status, {"error": "unavailable"}))
    value.identity = {"claims": {"org_id": "o1"}}
    result = load_policy(client=value, local={"enabled": True})
    assert result.enabled is False
    assert result.source == "server_unavailable"


def test_fixed_mode_does_not_fetch_agent_led_policy(monkeypatch):
    from hermes_wisdom.agent_led.policy import load_policy

    monkeypatch.setattr("hermes_wisdom.mediation.delivery_mode", lambda: "fixed")
    value = client(Response(200, _agent_led_policy()))
    value.identity = {"claims": {"org_id": "o1"}}
    assert load_policy(client=value, local={"enabled": True}).enabled is False
    assert not value.session.calls


def test_suppression_wire_contains_only_opaque_keys_and_empty_write_body():
    key = "sha256:" + "a" * 64
    value = client(
        Response(
            200, {"org_id": "o1", "key": key, "suppress_until": "2026-10-01T00:00:00Z"}
        )
    )
    value.identity = {"claims": {"org_id": "o1"}}
    assert value.suppress_recommendation(key).key == key
    assert value.session.calls[0][2]["json"] == {}
    assert value.session.calls[0][1].endswith("/suppressions/sha256%3A" + "a" * 64)
    value.session.response = Response(200, {"org_id": "o1", "suppressions": []})
    assert value.recommendation_suppressions([key]) == []
    assert value.session.calls[-1][2]["json"] == {"keys": [key]}


def test_suppression_response_cannot_inject_unrequested_or_foreign_keys():
    key = "sha256:" + "a" * 64
    value = client(
        Response(
            200,
            {
                "org_id": "o1",
                "suppressions": [
                    {
                        "key": "sha256:" + "b" * 64,
                        "suppress_until": "2026-10-01T00:00:00Z",
                    }
                ],
            },
        )
    )
    value.identity = {"claims": {"org_id": "o1"}}
    with pytest.raises(WisdomError):
        value.recommendation_suppressions([key])
    value.session.response = Response(200, {"org_id": "other", "suppressions": []})
    with pytest.raises(WisdomError):
        value.recommendation_suppressions([key])


def test_mute_wire_preserves_native_duration_vocabulary():
    value = client(
        Response(
            200,
            {
                "org_id": "o1",
                "muted": True,
                "duration": "1_week",
                "muted_until": "2026-10-01T00:00:00Z",
                "forever": False,
            },
        )
    )
    value.identity = {"claims": {"org_id": "o1"}}
    assert value.set_recommendation_mute("1_week").muted
    assert value.session.calls[-1][2]["json"] == {"duration": "1_week"}
    with pytest.raises(WisdomValidationError):
        value.set_recommendation_mute("arbitrary")


def test_submit_body_has_no_local_candidate_or_activity_signals():
    body = {
        "draft": {
            "id": "d1",
            "orgId": "o1",
            "ownerUserId": "u1",
            "slug": "my-skill",
            "draftCommit": "sha256:" + "a" * 64,
            "contentHash": "sha256:" + "b" * 64,
            "authorDescription": "Does a task.",
            "authorDescriptionHash": "sha256:" + "c" * 64,
            "state": "ready",
            "packageManifestHash": "sha256:" + "d" * 64,
            "packageManifestSchemaVersion": 1,
            "systemSpec": None,
            "scan": None,
            "scanVerdict": "pass",
            "explanation": None,
            "updatedAt": "now",
        }
    }
    value = client(Response(201, body))
    value.submit_draft(
        slug="my-skill",
        commit="sha256:" + "a" * 64,
        content_hash="sha256:" + "b" * 64,
        description="Does a task.",
    )
    payload = value.session.calls[0][2]["json"]
    assert payload == {
        "slug": "my-skill",
        "draft_commit": "sha256:" + "a" * 64,
        "content_hash": "sha256:" + "b" * 64,
        "author_description": "Does a task.",
    }
    assert not (
        {"usage", "refinement", "candidate", "ranking", "stability", "dismissal"}
        & payload.keys()
    )


def test_not_found_is_opaque():
    value = client(Response(404, {"error": "not_found"}))
    with pytest.raises(WisdomNotFound, match="item not found"):
        value._request("GET", "skills/secret")


def test_approve_exact_three_hash_body():
    value = client(Response(200, {"draft": {"id": "invalid"}}))
    with pytest.raises(Exception):
        value.approve("d1", content_hash="c", description_hash="d", manifest_hash="m")
    payload = value.session.calls[0][2]["json"]
    assert payload == {
        "content_hash": "c",
        "author_description_hash": "d",
        "package_manifest_hash": "m",
    }


def test_revise_binds_predecessor_hashes_and_new_private_commit():
    value = client(Response(201, {"draft": _draft(id="d2")}))
    professionalism_review = {
        "schema_version": 1,
        "content_hash": "sha256:" + "f" * 64,
        "author_description_hash": "sha256:" + "9" * 64,
        "status": "pass",
        "summary": "No language or conduct concerns detected.",
        "checks": [],
        "provenance": {
            "kind": "agent_assessed",
            "provider": "codex",
            "model": "gpt-5.6-sol",
        },
        "assessed_at": "2026-09-02T00:00:00Z",
    }

    revised = value.revise_draft(
        "d1",
        commit="sha256:" + "e" * 64,
        content_hash="sha256:" + "f" * 64,
        description="Updated owner copy.",
        expected_content_hash="sha256:" + "b" * 64,
        expected_description_hash="sha256:" + "c" * 64,
        expected_manifest_hash="sha256:" + "d" * 64,
        professionalism_review=professionalism_review,
    )

    assert revised.id == "d2"
    method, url, request = value.session.calls[0]
    assert (method, url) == (
        "POST",
        "https://gateway.example/v1/sync/wisdom/drafts/d1/revise",
    )
    assert request["json"] == {
        "draft_commit": "sha256:" + "e" * 64,
        "content_hash": "sha256:" + "f" * 64,
        "author_description": "Updated owner copy.",
        "expected_content_hash": "sha256:" + "b" * 64,
        "expected_author_description_hash": "sha256:" + "c" * 64,
        "expected_package_manifest_hash": "sha256:" + "d" * 64,
        "professionalism_review": professionalism_review,
    }


def test_content_fetch_is_bound_to_installation_identity_and_takedown_generation():
    skill = b"# Skill\n"
    manifest = b'{"schema_version":1,"name":"skill","requirements":{"hermes":{"minimum_version":"0.1.0"}}}'
    files = [
        ("SKILL.md", "file", skill),
        ("skill.manifest.json", "file", manifest),
    ]
    _records, content_hash = verify_content_files(files)
    value = client(
        Response(
            200,
            {
                "commit": "sha256:" + "a" * 64,
                "content_hash": content_hash,
                "files": [
                    {
                        "path": path,
                        "mode": mode,
                        "hash": "sha256:" + hashlib.sha256(body).hexdigest(),
                        "content_base64": base64.b64encode(body).decode("ascii"),
                    }
                    for path, mode, body in files
                ],
            },
        )
    )

    value.content(
        "skill-1",
        2,
        installation_id="hwi_1234567890123456",
        takedown_generation=7,
    )

    method, url, request = value.session.calls[0]
    assert (method, url) == (
        "GET",
        "https://gateway.example/v1/sync/wisdom/skills/skill-1/versions/2/content",
    )
    assert request["params"] == {
        "installation_id": "hwi_1234567890123456",
        "takedown_generation": 7,
    }


def test_raw_copy_fetches_hash_verified_published_bytes_without_install_state():
    skill = b"# Skill\n"
    manifest = b'{"schema_version":1,"name":"skill","requirements":{"hermes":{"minimum_version":"0.1.0"}}}'
    files = [
        ("SKILL.md", "file", skill),
        ("skill.manifest.json", "file", manifest),
    ]
    _records, content_hash = verify_content_files(files)
    value = client(
        Response(
            200,
            {
                "commit": "sha256:" + "a" * 64,
                "content_hash": content_hash,
                "copy_semantics": "unmanaged_fork",
                "files": [
                    {
                        "path": path,
                        "mode": mode,
                        "hash": "sha256:" + hashlib.sha256(body).hexdigest(),
                        "content_base64": base64.b64encode(body).decode("ascii"),
                    }
                    for path, mode, body in files
                ],
            },
        )
    )

    _response, decoded = value.raw_copy("skill-1", 2)

    assert decoded == files
    method, url, request = value.session.calls[0]
    assert (method, url) == (
        "GET",
        "https://gateway.example/v1/sync/wisdom/skills/skill-1/versions/2/raw",
    )
    assert request["params"] is None


def test_installation_reconciliation_uses_identity_path_and_owned_delete():
    value = client(Response(200, {"installations": []}))
    assert value.installations("hwi_1234567890123456") == []
    method, url, request = value.session.calls[0]
    assert (method, url) == (
        "GET",
        "https://gateway.example/v1/sync/wisdom/installations/hwi_1234567890123456",
    )
    assert request["params"] is None

    value.session.response = Response(
        200,
        {
            "skill_id": "skill-1",
            "installation_id": "hwi_1234567890123456",
            "state": "inactive",
        },
    )
    result = value.deactivate_install("hwi_1234567890123456", "skill-1")
    assert result.state == "inactive"
    assert value.session.calls[1][0:2] == (
        "DELETE",
        "https://gateway.example/v1/sync/wisdom/installations/hwi_1234567890123456/skills/skill-1",
    )


def test_installation_and_feed_responses_are_typed_fail_closed():
    value = client(
        Response(
            200,
            {
                "installations": [
                    {
                        "skill_id": "skill-1",
                        "installed_version": 1,
                        "latest_version": 2,
                        "update_mode": "UNKNOWN",
                        "skill_state": "active",
                        "takedown_generation": 0,
                    }
                ]
            },
        )
    )
    with pytest.raises(WisdomError, match="schema validation"):
        value.installations("hwi_1234567890123456")

    value.session.response = Response(
        200,
        {
            "events": [
                {
                    "event_id": "event-1",
                    "kind": "invented",
                    "skill_id": "skill-1",
                    "version": 2,
                    "takedown_generation": 0,
                    "installation_id": "hwi_1234567890123456",
                    "update_mode": "MANUAL",
                    "occurred_at": "2026-08-24T00:00:00Z",
                }
            ],
            "next_cursor": "cursor-1",
            "has_more": False,
        },
    )
    with pytest.raises(WisdomError, match="schema validation"):
        value.feed(installation_id="hwi_1234567890123456")
