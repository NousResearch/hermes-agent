from __future__ import annotations

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from hermes_cli import web_server
from hermes_cli.web_models import (
    WisdomCandidateEventRequest,
    WisdomEditedFile,
    WisdomCandidateDismissRequest,
    WisdomInstallApplyRequest,
    WisdomInstallPlanRequest,
    WisdomPreparedSaveRequest,
    WisdomReviseRequest,
    WisdomSetupRequest,
    WisdomSuggestRequest,
    WisdomUpdateApplyRequest,
)


def test_setup_bff_forwards_explicit_disclosure_with_profile_scope(monkeypatch):
    calls = []
    monkeypatch.setattr(
        web_server, "_profile_cli_args", lambda profile: ["-p", str(profile)]
    )

    def spawn(command, name):
        calls.append((command, name))
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(web_server, "_spawn_hermes_action", spawn)
    result = asyncio.run(
        web_server.post_wisdom_setup(
            WisdomSetupRequest(accept_disclosure=True, profile="research")
        )
    )
    assert result["ok"] is True
    assert result["pid"] == 123
    assert calls == [
        (
            [
                "-p",
                "research",
                "wisdom",
                "setup",
                "--accept-disclosure",
                "--json",
            ],
            result["name"],
        )
    ]

    with pytest.raises(HTTPException) as rejected:
        asyncio.run(
            web_server.post_wisdom_setup(
                WisdomSetupRequest(accept_disclosure=False, profile="research")
            )
        )
    assert rejected.value.status_code == 422


def test_skill_detail_bff_resolves_command_slug_with_profile_scope(monkeypatch):
    calls = []

    class Service:
        def resolve_skill(self, reference):
            calls.append(("resolve", reference))
            return {"skill": {"id": "skill-1", "slug": reference}}

    async def run(profile, fn):
        calls.append(("profile", profile))
        return fn(Service())

    monkeypatch.setattr(web_server, "_run_wisdom", run)

    result = asyncio.run(
        web_server.get_wisdom_skill("collective-wisdom-canary", profile="research")
    )

    assert result == {"skill": {"id": "skill-1", "slug": "collective-wisdom-canary"}}
    assert calls == [
        ("profile", "research"),
        ("resolve", "collective-wisdom-canary"),
    ]


def test_version_detail_bff_preserves_skill_version_and_profile_scope(monkeypatch):
    calls = []

    class Service:
        def version_detail(self, reference, version):
            calls.append(("version", reference, version))
            return {
                "skill": {"id": "skill-1", "slug": reference},
                "version": {"version": version, "explanation": "Reviewed release"},
            }

    async def run(profile, fn):
        calls.append(("profile", profile))
        return fn(Service())

    monkeypatch.setattr(web_server, "_run_wisdom", run)

    result = asyncio.run(
        web_server.get_wisdom_version("collective-wisdom-canary", 2, profile="research")
    )

    assert result["version"] == {"version": 2, "explanation": "Reviewed release"}
    assert calls == [
        ("profile", "research"),
        ("version", "collective-wisdom-canary", 2),
    ]


def test_suggest_bff_preserves_profile_and_owner_approved_fields(monkeypatch) -> None:
    calls: list[tuple[str | None, tuple[object, ...], dict[str, object]]] = []

    class Service:
        def suggest(self, *args, **kwargs):
            calls.append(("work", args, kwargs))
            return {"network_submission": True}

    async def run(profile, fn):
        calls.append((profile, (), {}))
        return fn(Service())

    monkeypatch.setattr(web_server, "_run_wisdom", run)
    body = WisdomSuggestRequest(
        skill="work",
        local_skill_id="local-1",
        description="Outcome-oriented owner copy",
        system_specification={"auto_install": False},
        send_for_owner_only_server_review=True,
        profile="customer-a",
    )

    result = asyncio.run(web_server.post_wisdom_suggest(body))

    assert result == {"network_submission": True}
    assert calls == [
        ("customer-a", (), {}),
        (
            "work",
            ("work",),
            {
                "description": "Outcome-oriented owner copy",
                "system_specification": {"auto_install": False},
                "allow_private_secret_review": True,
                "local_skill_id": "local-1",
            },
        ),
    ]


def test_revise_bff_forwards_complete_content_and_hash_preconditions(
    monkeypatch,
) -> None:
    calls = []

    class Service:
        def revise(self, draft_id, **kwargs):
            calls.append((draft_id, kwargs))
            return {"draft": {"id": "draft-2"}}

    async def run(profile, fn):
        calls.append(profile)
        return fn(Service())

    monkeypatch.setattr(web_server, "_run_wisdom", run)
    body = WisdomReviseRequest(
        draft_id="draft-1",
        author_description="Updated owner copy",
        files=[
            WisdomEditedFile(path="SKILL.md", content_utf8="# Updated\n"),
            WisdomEditedFile(
                path="skill.manifest.json",
                content_utf8='{"schema_version":1}',
            ),
        ],
        expected_content_hash="sha256:content",
        expected_author_description_hash="sha256:description",
        expected_package_manifest_hash="sha256:manifest",
        profile="research",
    )

    result = asyncio.run(web_server.post_wisdom_revise(body))

    assert result == {"draft": {"id": "draft-2"}}
    assert calls == [
        "research",
        (
            "draft-1",
            {
                "author_description": "Updated owner copy",
                "files": [
                    {"path": "SKILL.md", "content_utf8": "# Updated\n"},
                    {
                        "path": "skill.manifest.json",
                        "content_utf8": '{"schema_version":1}',
                    },
                ],
                "expected_content_hash": "sha256:content",
                "expected_description_hash": "sha256:description",
                "expected_manifest_hash": "sha256:manifest",
                "allow_private_secret_review": False,
            },
        ),
    ]


def test_prepared_save_and_candidate_actions_remain_profile_scoped(monkeypatch) -> None:
    calls = []

    class Service:
        def save_prepared(self, draft_id, **kwargs):
            calls.append(("save", draft_id, kwargs))
            return {"local_draft_id": draft_id}

        def dismiss_local_candidate(self, local_skill_id, content_hash):
            calls.append(("dismiss", local_skill_id, content_hash))
            return {"dismissed": True}

        def defer_candidate_prompt(self, event_id, *, surface):
            calls.append(("defer", event_id, surface))
            return {"event_id": event_id, "state": "deferred"}

        def approve_candidate(self, event_id):
            calls.append(("approve", event_id))
            return {"event_id": event_id, "state": "pending_moderation"}

    async def run(profile, fn):
        calls.append(("profile", profile))
        return fn(Service())

    monkeypatch.setattr(web_server, "_run_wisdom", run)
    monkeypatch.setattr(
        web_server,
        "_schedule_wisdom_professionalism_reviews",
        lambda profile: calls.append(("schedule", profile)),
    )
    files = [
        WisdomEditedFile(path="SKILL.md", content_utf8="# Edited\n"),
        WisdomEditedFile(
            path="skill.manifest.json", content_utf8='{"schema_version":1}'
        ),
    ]
    saved = asyncio.run(
        web_server.post_wisdom_prepared_save(
            WisdomPreparedSaveRequest(
                draft_id="local:1",
                author_description="Owner copy",
                files=files,
                profile="research",
            )
        )
    )
    dismissed = asyncio.run(
        web_server.post_wisdom_candidate_dismiss(
            WisdomCandidateDismissRequest(
                local_skill_id="skill-1",
                content_hash="sha256:content",
                profile="research",
            )
        )
    )
    deferred = asyncio.run(
        web_server.post_wisdom_candidate_defer(
            WisdomCandidateEventRequest(event_id="event-1", profile="research")
        )
    )
    approved = asyncio.run(
        web_server.post_wisdom_candidate_approve(
            WisdomCandidateEventRequest(event_id="event-1", profile="research")
        )
    )

    assert saved == {"local_draft_id": "local:1"}
    assert dismissed == {"dismissed": True}
    assert deferred == {"event_id": "event-1", "state": "deferred"}
    assert approved == {"event_id": "event-1", "state": "pending_moderation"}
    assert calls == [
        ("profile", "research"),
        (
            "save",
            "local:1",
            {
                "author_description": "Owner copy",
                "files": [item.model_dump() for item in files],
            },
        ),
        ("schedule", "research"),
        ("profile", "research"),
        ("dismiss", "skill-1", "sha256:content"),
        ("profile", "research"),
        ("defer", "event-1", "desktop"),
        ("profile", "research"),
        ("approve", "event-1"),
    ]


def test_candidate_event_feed_is_scoped_to_undelivered_desktop_events(
    monkeypatch,
) -> None:
    calls = []

    class Service:
        def pending_candidate_events(self, *, session_id, surface):
            calls.append((session_id, surface))
            return [{"id": "event-1"}]

    async def run(profile, fn):
        calls.append(profile)
        return fn(Service())

    monkeypatch.setattr(web_server, "_run_wisdom", run)

    result = asyncio.run(
        web_server.get_wisdom_events(profile="research", session_id="session-1")
    )

    assert result == {"events": [{"id": "event-1"}]}
    assert calls == ["research", ("session-1", "desktop")]


def test_install_apply_bff_requires_a_plan_receipt(monkeypatch) -> None:
    calls: list[tuple[str | None, str, bool]] = []

    class Service:
        def install_apply(self, receipt: str, *, accept_partial: bool):
            calls.append((None, receipt, accept_partial))
            return {"state": "installed"}

    async def run(profile, fn):
        result = fn(Service())
        calls[0] = (profile, calls[0][1], calls[0][2])
        return result

    monkeypatch.setattr(web_server, "_run_wisdom", run)
    body = WisdomInstallApplyRequest(
        receipt="receipt-123", accept_partial=True, profile="customer-b"
    )

    result = asyncio.run(web_server.post_wisdom_install_apply(body))

    assert result == {"state": "installed"}
    assert calls == [("customer-b", "receipt-123", True)]


def test_install_plan_bff_forwards_validated_update_mode(monkeypatch) -> None:
    calls: list[tuple[str | None, str, str | None]] = []

    class Service:
        def install_plan(self, reference: str, *, update_mode: str | None):
            calls.append((None, reference, update_mode))
            return {"state": "planned", "update_mode": update_mode}

    async def run(profile, fn):
        result = fn(Service())
        calls[0] = (profile, calls[0][1], calls[0][2])
        return result

    monkeypatch.setattr(web_server, "_run_wisdom", run)
    body = WisdomInstallPlanRequest(
        reference="skill-1", update_mode="AUTO_WITH_NOTICE", profile="customer-b"
    )

    result = asyncio.run(web_server.post_wisdom_install_plan(body))

    assert result == {"state": "planned", "update_mode": "AUTO_WITH_NOTICE"}
    assert calls == [("customer-b", "skill-1", "AUTO_WITH_NOTICE")]


def test_install_plan_request_rejects_unknown_update_mode() -> None:
    with pytest.raises(ValueError):
        WisdomInstallPlanRequest(reference="skill-1", update_mode="SILENT")


def test_wisdom_error_mapping_is_opaque_and_bounded() -> None:
    from hermes_wisdom.client import WisdomError, WisdomNotFound

    assert web_server._wisdom_http_error(WisdomNotFound("not found")).status_code == 404
    assert web_server._wisdom_http_error(WisdomError("retry")).status_code == 503


def test_profile_bff_requires_setup_before_running_an_operation(monkeypatch) -> None:
    calls: list[str] = []

    class Service:
        def require_setup(self):
            calls.append("setup")

    monkeypatch.setattr(web_server, "_profile_scope", lambda _profile: nullcontext())
    monkeypatch.setattr("hermes_wisdom.service.WisdomService", Service)

    result = asyncio.run(
        web_server._run_wisdom("research", lambda _service: calls.append("work"))
    )

    assert result is None
    assert calls == ["setup", "work"]


def test_status_bff_is_available_before_setup(monkeypatch) -> None:
    async def run(profile, fn, *, require_setup=True):
        assert profile == "research"
        assert require_setup is False
        return {"configured": False}

    monkeypatch.setattr(web_server, "_run_wisdom", run)

    assert asyncio.run(web_server.get_wisdom_status(profile="research")) == {
        "configured": False
    }


def test_update_bff_forwards_only_explicit_confirmation_flags(monkeypatch) -> None:
    calls = []

    class Service:
        def update_apply(self, receipt, **kwargs):
            calls.append((receipt, kwargs))
            return {"updated": True}

    async def run(profile, fn):
        calls.append(profile)
        return fn(Service())

    monkeypatch.setattr(web_server, "_run_wisdom", run)
    body = WisdomUpdateApplyRequest(
        receipt="wup_123",
        accept_sensitive=True,
        accept_partial=False,
        preserve_modified=True,
        profile="research",
    )
    result = asyncio.run(web_server.post_wisdom_update_apply(body))
    assert result == {"updated": True}
    assert calls == [
        "research",
        (
            "wup_123",
            {
                "accept_sensitive": True,
                "accept_partial": False,
                "preserve_modified": True,
            },
        ),
    ]
