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
