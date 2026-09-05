"""Tests for the in-process hosted room session adapter."""

from __future__ import annotations

from pathlib import Path
import base64

import threading
from types import SimpleNamespace

import pytest

from gateway.hosted_room_driver import TaskIdentity
from tui_gateway.hosted_room_server_rpc import (
    HostedRoomServerRPC,
    HostedRoomSessionError,
)


def _server():
    sessions = {
        "runtime": {
            "history_lock": threading.Lock(),
            "attached_images": [],
        }
    }
    calls = []

    def method(name, result):
        def handler(rid, params):
            calls.append((name, params))
            value = result(params) if callable(result) else result
            return {"id": rid, **value}

        return handler

    methods = {
        "session.list": method(
            "session.list",
            {"result": {"sessions": [{"id": "stored", "resolved_id": "tip", "title": "Group: room"}]}},
        ),
        "session.create": method("session.create", {"result": {"session_id": "runtime"}}),
        "session.resume": method("session.resume", {"result": {"session_id": "runtime"}}),
        "session.history": method("session.history", {"result": {"messages": [{"role": "assistant"}]}}),
        "session.interrupt": method("session.interrupt", {"result": {"interrupted": True}}),
        "approval.respond": method("approval.respond", {"result": {"resolved": 1}}),
        "prompt.submit": method("prompt.submit", {"result": {"status": "streaming"}}),
        "image.attach_bytes": method(
            "image.attach_bytes", {"result": {"attached": True}}
        ),
        "pdf.attach": method("pdf.attach", {"result": {"attached": True}}),
        "file.attach": method(
            "file.attach",
            {"result": {"attached": True, "ref_text": "@file:attachments/notes.txt"}},
        ),
    }
    server = SimpleNamespace(
        _methods=methods,
        _sessions=sessions,
        _sessions_lock=threading.Lock(),
        _pending_approval_request_payload=lambda _session_key: None,
    )
    return server, calls


def test_routes_exact_hidden_session_and_internal_task_proof():
    server, calls = _server()
    rpc = HostedRoomServerRPC(server)
    task = TaskIdentity("room", "task", "thread", "turn")
    callback = lambda _receipt: None

    assert rpc.resolve_exact(profile="ops", title="Group: room", source="bot_room")["session_id"] == "tip"
    assert rpc.create(profile="ops", title="Group: room", source="bot_room")["session_id"] == "runtime"
    rpc.bind_artifact_scope(
        task=task,
        execution_generation=2,
        member_id="member-ops",
        authority_gateway_id="gateway-a",
        authority_epoch=1,
        profile="ops",
    )
    rpc.submit(
        profile="ops",
        session_id="runtime",
        prompt="Do the work",
        source="bot_room",
        task=task,
        execution_generation=2,
        on_terminal=callback,
    )

    create = next(params for method, params in calls if method == "session.create")
    submit = next(params for method, params in calls if method == "prompt.submit")
    assert create["hidden"] is True
    assert create["room_plumbing"] is True
    assert create["follow_profile_config"] is True
    assert create["close_on_disconnect"] is False
    assert submit["_hosted_task"] == {
        "room_id": "room",
        "task_id": "task",
        "thread_id": "thread",
        "turn_id": "turn",
        "execution_generation": 2,
        "member_id": "member-ops",
        "target_profile": "ops",
        "home_install_id": submit["_hosted_task"]["home_install_id"],
        "target_install_id": submit["_hosted_task"]["target_install_id"],
        "authority_gateway_id": "gateway-a",
        "authority_epoch": 1,
    }
    assert submit["_hosted_task"]["home_install_id"] == submit["_hosted_task"][
        "target_install_id"
    ]
    assert submit["_hosted_terminal_callback"] is not callback
    assert callable(submit["_hosted_terminal_callback"])

    rpc.resume(profile="ops", session_id="stored", source="bot_room")
    resume = next(params for method, params in calls if method == "session.resume")
    assert resume["source"] == "bot_room"


def test_info_and_interrupt_are_exact_task_scoped():
    server, calls = _server()
    lock = threading.Lock()
    server._sessions["runtime"] = {
        "history_lock": lock,
        "running": True,
        "_hosted_room_task": {"task_id": "task-a"},
    }
    rpc = HostedRoomServerRPC(server)

    assert rpc.info(profile="ops", session_id="runtime", source="bot_room") == {
        "active": True,
        "task_id": "task-a",
    }
    rpc.interrupt(
        profile="ops",
        session_id="runtime",
        source="bot_room",
        expected_task_id="task-a",
    )
    params = next(params for method, params in calls if method == "session.interrupt")
    assert params["expected_hosted_task_id"] == "task-a"


def test_local_approval_snapshot_and_response_use_exact_request():
    server, calls = _server()
    server._pending_approval_request_payload = lambda session_key: {
        "request_id": "approval-1",
        "command": "pytest -q tests/focused",
        "choices": ["once", "deny"],
    } if session_key == "stored-session" else None
    server._sessions["runtime"] = {
        "history_lock": threading.Lock(),
        "running": True,
        "session_key": "stored-session",
        "_hosted_room_task": {"task_id": "task-a"},
    }
    rpc = HostedRoomServerRPC(server)

    info = rpc.info(profile="ops", session_id="runtime", source="bot_room")
    assert info["status"] == "waiting_for_approval"
    assert info["pending_approval"]["request_id"] == "approval-1"
    assert rpc.approve(
        session_id="runtime",
        request_id="approval-1",
        choice="once",
    ) == {"resolved": 1}
    params = next(params for method, params in calls if method == "approval.respond")
    assert params == {
        "session_id": "runtime",
        "request_id": "approval-1",
        "choice": "once",
        "all": False,
    }


def test_rpc_errors_are_typed(caplog):
    server, _calls = _server()
    server._methods["session.list"] = lambda rid, _params: {
        "id": rid,
        "error": {"code": 4007, "message": "private_error_canary"},
    }
    rpc = HostedRoomServerRPC(server)

    with pytest.raises(HostedRoomSessionError) as exc:
        rpc.resolve_exact(profile="ops", title="Group: room", source="bot_room")
    assert exc.value.code == 4007
    assert "method=session.list code=4007" in caplog.text
    assert "private_error_canary" not in caplog.text


def test_prompt_rejection_is_proven_not_admitted():
    server, _calls = _server()
    server._methods["prompt.submit"] = lambda rid, _params: {
        "id": rid,
        "error": {"code": 4121, "message": "session is already busy"},
    }
    rpc = HostedRoomServerRPC(server)
    task = TaskIdentity("room", "task", "thread", "turn")
    rpc.bind_artifact_scope(
        task=task,
        execution_generation=1,
        member_id="ops",
        authority_gateway_id="gateway-a",
        authority_epoch=1,
        profile="ops",
    )

    with pytest.raises(HostedRoomSessionError) as exc:
        rpc.submit(
            profile="ops",
            session_id="runtime",
            prompt="Do the work",
            source="bot_room",
            task=task,
            execution_generation=1,
            on_terminal=lambda _receipt: None,
        )

    assert exc.value.code == 4121
    assert exc.value.not_admitted is True


def test_attachment_staging_rollback_deletes_only_attempt_uploaded_files(tmp_path):
    server, calls = _server()
    attachment_dir = tmp_path / "attachments"
    attachment_dir.mkdir()
    uploaded = attachment_dir / "notes.txt"
    uploaded.write_text("release notes", encoding="utf-8")
    preserved = attachment_dir / "existing.txt"
    preserved.write_text("keep", encoding="utf-8")
    server._session_home_dir = lambda _session, name: attachment_dir
    server._methods["file.attach"] = lambda rid, params: {
        "id": rid,
        "result": {
            "attached": True,
            "path": str(uploaded),
            "ref_text": "@file:attachments/notes.txt",
            "uploaded": True,
        },
    }
    rpc = HostedRoomServerRPC(server)
    attachment = {
        "attachment_id": f"att_{'3' * 32}",
        "kind": "file",
        "name": "notes.txt",
        "mime": "text/plain",
    }

    rpc.stage_attachment(
        profile="ops",
        session_id="runtime",
        source="bot_room",
        attachment=attachment,
        data=b"release notes",
        execution_generation=9,
    )
    rpc.rollback_attachment_staging(
        profile="ops",
        session_id="runtime",
        source="bot_room",
        execution_generation=9,
    )

    assert not uploaded.exists()
    assert preserved.read_text(encoding="utf-8") == "keep"
    assert any(method == "file.attach" for method, _params in calls) is False


def test_attachment_staging_rollback_restores_pending_images_and_clears_cache():
    server, calls = _server()
    session = server._sessions["runtime"]

    def attach_image(rid, params):
        calls.append(("image.attach_bytes", params))
        session["attached_images"].append("/images/diagram.png")
        return {
            "id": rid,
            "result": {
                "attached": True,
                "path": "/images/diagram.png",
            },
        }

    def fail_pdf_after_partial_mutation(rid, params):
        calls.append(("pdf.attach", params))
        session["attached_images"].append("/images/brief-page-1.png")
        return {
            "id": rid,
            "error": {"code": 5028, "message": "second attachment failed"},
        }

    server._methods["image.attach_bytes"] = attach_image
    server._methods["pdf.attach"] = fail_pdf_after_partial_mutation
    rpc = HostedRoomServerRPC(server)
    image = {
        "attachment_id": f"att_{'1' * 32}",
        "kind": "image",
        "name": "diagram.png",
        "mime": "image/png",
    }
    pdf = {
        "attachment_id": f"att_{'2' * 32}",
        "kind": "pdf",
        "name": "brief.pdf",
        "mime": "application/pdf",
    }

    rpc.stage_attachment(
        profile="ops",
        session_id="runtime",
        source="bot_room",
        attachment=image,
        data=b"image",
        execution_generation=7,
    )
    with pytest.raises(HostedRoomSessionError, match="second attachment failed"):
        rpc.stage_attachment(
            profile="ops",
            session_id="runtime",
            source="bot_room",
            attachment=pdf,
            data=b"pdf",
            execution_generation=7,
        )

    assert session["attached_images"] == [
        "/images/diagram.png",
        "/images/brief-page-1.png",
    ]
    rpc.rollback_attachment_staging(
        profile="ops",
        session_id="runtime",
        source="bot_room",
        execution_generation=7,
    )
    assert session["attached_images"] == []

    rpc.stage_attachment(
        profile="ops",
        session_id="runtime",
        source="bot_room",
        attachment=image,
        data=b"image",
        execution_generation=7,
    )
    assert len([called for called, _params in calls if called == "image.attach_bytes"]) == 2


@pytest.mark.parametrize(
    ("kind", "name", "mime", "method"),
    [
        ("image", "diagram.png", "image/png", "image.attach_bytes"),
        ("pdf", "brief.pdf", "application/pdf", "pdf.attach"),
        ("file", "notes.txt", "text/plain", "file.attach"),
    ],
)
def test_attachment_staging_uses_existing_attach_rpcs_and_deduplicates_attempt(
    kind, name, mime, method
):
    server, calls = _server()
    rpc = HostedRoomServerRPC(server)
    attachment = {
        "attachment_id": f"att_{'1' * 32}",
        "kind": kind,
        "name": name,
        "size": 5,
        "mime": mime,
    }

    first = rpc.stage_attachment(
        profile="ops",
        session_id="runtime",
        source="bot_room",
        attachment=attachment,
        data=b"bytes",
        execution_generation=1,
    )
    repeated = rpc.stage_attachment(
        profile="ops",
        session_id="runtime",
        source="bot_room",
        attachment=attachment,
        data=b"bytes",
        execution_generation=1,
    )
    rpc.stage_attachment(
        profile="ops",
        session_id="runtime",
        source="bot_room",
        attachment=attachment,
        data=b"bytes",
        execution_generation=2,
    )

    assert first == repeated
    matching = [params for called, params in calls if called == method]
    assert len(matching) == 2
    if kind == "file":
        assert matching[0]["data_url"] == (
            f"data:{mime};base64,{base64.b64encode(b'bytes').decode('ascii')}"
        )
    else:
        assert matching[0]["content_base64"] == base64.b64encode(b"bytes").decode(
            "ascii"
        )


def test_terminal_callback_publishes_task_scoped_artifacts(tmp_path: Path, monkeypatch):
    from gateway.hosted_room_artifacts import RoomArtifactOutbox, RoomArtifactScope

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    server, calls = _server()
    rpc = HostedRoomServerRPC(server)
    task = TaskIdentity("room", "task", "thread", "turn")
    captured = []
    rpc.bind_artifact_scope(
        task=task,
        execution_generation=2,
        member_id="member-ops",
        authority_gateway_id="gateway-a",
        authority_epoch=1,
        profile="ops",
    )
    rpc.submit(
        profile="ops",
        session_id="runtime",
        prompt="Do the work",
        source="bot_room",
        task=task,
        execution_generation=2,
        on_terminal=captured.append,
    )
    submit = next(params for method, params in calls if method == "prompt.submit")
    scope = RoomArtifactScope.from_mapping({
        key: submit["_hosted_task"][key]
        for key in (
            "room_id",
            "task_id",
            "execution_generation",
            "member_id",
            "target_profile",
            "home_install_id",
            "target_install_id",
            "authority_gateway_id",
            "authority_epoch",
        )
    })
    output = tmp_path / "handoff.md"
    output.write_text("# Handoff\n", encoding="utf-8")
    RoomArtifactOutbox(home / "state.db").put_path(scope=scope, path=output)

    submit["_hosted_terminal_callback"]({"status": "settled", "text": "Done"})

    assert captured[0]["status"] == "settled"
    assert captured[0]["artifacts"]["items"][0]["name"] == "handoff.md"


def test_terminal_callback_fails_closed_when_manifest_cannot_finalize(
    tmp_path: Path, monkeypatch
):
    from gateway import hosted_room_artifacts

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(
        hosted_room_artifacts,
        "terminal_artifact_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("disk fault")),
    )
    server, calls = _server()
    rpc = HostedRoomServerRPC(server)
    task = TaskIdentity("room", "task", "thread", "turn")
    captured = []
    rpc.bind_artifact_scope(
        task=task,
        execution_generation=1,
        member_id="member-ops",
        authority_gateway_id="gateway-a",
        authority_epoch=1,
        profile="ops",
    )
    rpc.submit(
        profile="ops",
        session_id="runtime",
        prompt="Do the work",
        source="bot_room",
        task=task,
        execution_generation=1,
        on_terminal=captured.append,
    )
    submit = next(params for method, params in calls if method == "prompt.submit")

    submit["_hosted_terminal_callback"]({"status": "settled", "text": "Done"})

    assert captured == [{
        "status": "failed",
        "text": "Done",
        "error": "A Group Chat file could not be finalized.",
    }]
