"""Behavioral regression tests for trusted attachment lifecycle hooks.

Stdlib-only by design: the production image does not ship pytest.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace


def _runtime_modules():
    import hermes_cli.plugins as plugins
    import tui_gateway.server as server
    import tui_gateway.methods_prompt as methods_prompt
    import tui_gateway.session_lifecycle as session_lifecycle
    import tui_gateway.session_compression as session_compression

    return (
        server,
        plugins,
        methods_prompt,
        session_lifecycle,
        session_compression,
    )


class _Patches:
    def __init__(self):
        self._items = []

    def set(self, obj, name, value):
        self._items.append(
            (obj, name, getattr(obj, name))
        )
        setattr(obj, name, value)

    def restore(self):
        for obj, name, old in reversed(self._items):
            setattr(obj, name, old)


def test_file_attach_emits_exact_trusted_payload():
    server, plugins, methods_prompt, _lifecycle, _compression = (
        _runtime_modules()
    )

    handler = server._methods["file.attach"]

    session = {
        "agent": SimpleNamespace(
            session_id="canon-live",
        ),
        "session_key": "canon-old",
        "profile_home": "/trusted/profile",
    }

    stored = Path(
        "/trusted/profile/attachments/input.txt"
    )

    calls = []

    def capture(hook_name, **kwargs):
        calls.append(
            (hook_name, kwargs)
        )
        return []

    patches = _Patches()

    try:
        patches.set(
            server,
            "_sess_building",
            lambda params, rid: (session, None),
        )

        patches.set(
            server,
            "_stage_session_file_attachment",
            lambda session, raw_path, data_url, name: (
                stored,
                True,
            ),
        )

        patches.set(
            server,
            "_attachment_ref_path",
            lambda session, path: "attachments/input.txt",
        )

        patches.set(
            server,
            "_format_ref_value",
            lambda value: value,
        )

        patches.set(
            plugins,
            "invoke_hook",
            capture,
        )

        response = handler(
            "rid-attach",
            {
                "session_id": "runtime-A",
                "path": "/client/input.txt",
                "data_url": "data:text/plain;base64,eA==",
                "name": "input.txt",
            },
        )

    finally:
        patches.restore()

    result = response["result"]

    assert result == {
        "attached": True,
        "name": "input.txt",
        "path": str(stored),
        "ref_path": "attachments/input.txt",
        "ref_text": "@file:attachments/input.txt",
        "uploaded": True,
    }

    assert len(calls) == 1

    hook_name, payload = calls[0]

    assert hook_name == "on_file_attachment_staged"

    assert payload == {
        "runtime_session_id": "runtime-A",
        "canonical_session_id": "canon-live",
        "profile_home": "/trusted/profile",
        "stored_path": str(stored),
        "uploaded": True,
        "ref_path": "attachments/input.txt",
        "ref_text": "@file:attachments/input.txt",
    }


def test_file_attach_hook_failure_is_fail_open():
    server, plugins, methods_prompt, _lifecycle, _compression = (
        _runtime_modules()
    )

    handler = server._methods["file.attach"]

    session = {
        "agent": None,
        "session_key": "canon-A",
        "profile_home": "/trusted/profile",
    }

    stored = Path(
        "/trusted/profile/attachments/input.txt"
    )

    def broken_hook(*args, **kwargs):
        raise RuntimeError("synthetic hook failure")

    patches = _Patches()

    try:
        patches.set(
            server,
            "_sess_building",
            lambda params, rid: (session, None),
        )

        patches.set(
            server,
            "_stage_session_file_attachment",
            lambda session, raw_path, data_url, name: (
                stored,
                True,
            ),
        )

        patches.set(
            server,
            "_attachment_ref_path",
            lambda session, path: "attachments/input.txt",
        )

        patches.set(
            server,
            "_format_ref_value",
            lambda value: value,
        )

        patches.set(
            plugins,
            "invoke_hook",
            broken_hook,
        )

        response = handler(
            "rid-fail-open",
            {
                "session_id": "runtime-A",
                "path": "/client/input.txt",
            },
        )

    finally:
        patches.restore()

    assert response["result"]["attached"] is True
    assert (
        response["result"]["ref_text"]
        == "@file:attachments/input.txt"
    )


def test_runtime_teardown_notification_is_idempotent():
    server, plugins, _methods_prompt, lifecycle, _compression = (
        _runtime_modules()
    )

    calls = []

    patches = _Patches()

    try:
        patches.set(
            plugins,
            "invoke_hook",
            lambda hook_name, **kwargs: (
                calls.append(
                    (hook_name, kwargs)
                )
                or []
            ),
        )

        session = {
            "_sid": "runtime-T",
            "session_key": "canon-old",
            "agent": SimpleNamespace(
                session_id="canon-live",
            ),
        }

        server._notify_runtime_session_teardown(
            session,
            "tui_close",
        )

        server._notify_runtime_session_teardown(
            session,
            "tui_close",
        )

    finally:
        patches.restore()

    assert len(calls) == 1

    hook_name, payload = calls[0]

    assert hook_name == "on_session_runtime_teardown"

    assert payload == {
        "runtime_session_id": "runtime-T",
        "canonical_session_id": "canon-live",
        "reason": "tui_close",
    }


def test_resume_hydration_identity_guard_preserves_replacement():
    server, plugins, _methods_prompt, _lifecycle, _compression = (
        _runtime_modules()
    )

    class Lease:
        def __init__(self):
            self.count = 0

        def release(self):
            self.count += 1

    old_lease = Lease()

    old = {
        "agent": None,
        "session_key": "canon-old",
        "resume_hydrating": True,
        "resume_history_ready": threading.Event(),
        "agent_ready": threading.Event(),
        "active_session_lease": old_lease,
    }

    replacement = {
        "session_key": "canon-replacement",
    }

    hook_calls = []

    class FailingDB:
        def reopen_session(self, stored_id):
            assert stored_id == "stored-P"

            # Simulate a newer runtime winning the same sid while the stale
            # hydration worker still owns `old` by object identity.
            with server._sessions_lock:
                assert server._sessions.get("runtime-P") is old
                server._sessions["runtime-P"] = replacement

            raise RuntimeError(
                "synthetic stale hydration failure"
            )

    class InlineThread:
        def __init__(self, *, target, daemon=False):
            self._target = target
            self.daemon = daemon

        def start(self):
            self._target()

    patches = _Patches()

    with server._sessions_lock:
        original_registry = dict(server._sessions)
        server._sessions.clear()
        server._sessions["runtime-P"] = old

    try:
        patches.set(
            server.threading,
            "Thread",
            InlineThread,
        )

        patches.set(
            plugins,
            "invoke_hook",
            lambda hook_name, **kwargs: (
                hook_calls.append(
                    (hook_name, kwargs)
                )
                or []
            ),
        )

        server._schedule_resume_hydration(
            "runtime-P",
            "stored-P",
            FailingDB(),
            close_db=False,
        )

        with server._sessions_lock:
            assert (
                server._sessions.get("runtime-P")
                is replacement
            )

        # The stale worker must not release resources or revoke lifecycle
        # state belonging to a runtime it no longer owns.
        assert old_lease.count == 0
        assert hook_calls == []

    finally:
        patches.restore()

        with server._sessions_lock:
            server._sessions.clear()
            server._sessions.update(
                original_registry
            )


def test_compression_rebind_emits_exact_transition_and_is_fail_open():
    server, plugins, _methods_prompt, _lifecycle, compression = (
        _runtime_modules()
    )

    import tools.approval as approval

    calls = []

    patches = _Patches()

    try:
        patches.set(
            server,
            "_transfer_active_session_slot",
            lambda *args, **kwargs: True,
        )

        patches.set(
            server,
            "_restart_slash_worker",
            lambda *args, **kwargs: None,
        )

        patches.set(
            approval,
            "unregister_gateway_notify",
            lambda *args, **kwargs: None,
        )

        patches.set(
            approval,
            "register_gateway_notify",
            lambda *args, **kwargs: None,
        )

        patches.set(
            approval,
            "is_session_yolo_enabled",
            lambda *args, **kwargs: False,
        )

        patches.set(
            approval,
            "enable_session_yolo",
            lambda *args, **kwargs: None,
        )

        patches.set(
            approval,
            "disable_session_yolo",
            lambda *args, **kwargs: None,
        )

        patches.set(
            plugins,
            "invoke_hook",
            lambda hook_name, **kwargs: (
                calls.append(
                    (hook_name, kwargs)
                )
                or []
            ),
        )

        session = {
            "agent": SimpleNamespace(
                session_id="canon-new",
            ),
            "session_key": "canon-old",
            "_queued_prompt_generation": 7,
            "pending_title": "keep-me",
        }

        server._sync_session_key_after_compress(
            "runtime-C",
            session,
            clear_pending_title=False,
            restart_slash_worker=False,
        )

        assert session["session_key"] == "canon-new"
        assert session["_queued_prompt_generation"] == 8
        assert session["pending_title"] == "keep-me"

        assert calls == [
            (
                "on_session_canonical_rebind",
                {
                    "runtime_session_id": "runtime-C",
                    "old_canonical_session_id": "canon-old",
                    "new_canonical_session_id": "canon-new",
                },
            ),
        ]

        def broken_hook(*args, **kwargs):
            raise RuntimeError(
                "synthetic rebind hook failure"
            )

        plugins.invoke_hook = broken_hook

        second = {
            "agent": SimpleNamespace(
                session_id="canon-2-new",
            ),
            "session_key": "canon-2-old",
            "_queued_prompt_generation": 0,
        }

        server._sync_session_key_after_compress(
            "runtime-C2",
            second,
            clear_pending_title=False,
            restart_slash_worker=False,
        )

        # Plugin failure cannot roll back Hermes' canonical transition.
        assert second["session_key"] == "canon-2-new"
        assert second["_queued_prompt_generation"] == 1

    finally:
        patches.restore()


def test_hydration_failure_tears_down_runtime_without_durable_finalize():
    server, plugins, _methods_prompt, _lifecycle, _compression = (
        _runtime_modules()
    )

    class Lease:
        def __init__(self):
            self.count = 0
            self.released = threading.Event()

        def release(self):
            self.count += 1
            self.released.set()

    class FailingDB:
        def reopen_session(self, stored_id):
            raise RuntimeError(
                "synthetic hydration failure"
            )

    lease = Lease()

    session = {
        "agent": None,
        "session_key": "canon-H",
        "resume_hydrating": True,
        "resume_history_ready": threading.Event(),
        "agent_ready": threading.Event(),
        "active_session_lease": lease,
    }

    calls = []
    finalize_calls = []

    patches = _Patches()

    with server._sessions_lock:
        original_registry = dict(server._sessions)
        server._sessions.clear()
        server._sessions["runtime-H"] = session

    try:
        patches.set(
            server,
            "_emit",
            lambda *args, **kwargs: None,
        )

        patches.set(
            server,
            "_finalize_session",
            lambda *args, **kwargs: finalize_calls.append(
                (args, kwargs)
            ),
        )

        patches.set(
            plugins,
            "invoke_hook",
            lambda hook_name, **kwargs: (
                calls.append(
                    (hook_name, kwargs)
                )
                or []
            ),
        )

        server._schedule_resume_hydration(
            "runtime-H",
            "stored-H",
            FailingDB(),
            close_db=False,
        )

        assert lease.released.wait(
            timeout=3.0
        ), "hydration failure did not release lease"

        assert lease.count == 1

        with server._sessions_lock:
            assert "runtime-H" not in server._sessions

        assert session["_sid"] == "runtime-H"

        # Hydration failure destroys only the runtime. It must not force
        # durable conversation finalization.
        assert finalize_calls == []

        teardown_calls = [
            item
            for item in calls
            if item[0]
            == "on_session_runtime_teardown"
        ]

        assert teardown_calls == [
            (
                "on_session_runtime_teardown",
                {
                    "runtime_session_id": "runtime-H",
                    "canonical_session_id": "canon-H",
                    "reason": "resume_hydration_failed",
                },
            ),
        ]

    finally:
        patches.restore()

        with server._sessions_lock:
            server._sessions.clear()
            server._sessions.update(
                original_registry
            )


TESTS = (
    test_file_attach_emits_exact_trusted_payload,
    test_file_attach_hook_failure_is_fail_open,
    test_runtime_teardown_notification_is_idempotent,
    test_resume_hydration_identity_guard_preserves_replacement,
    test_compression_rebind_emits_exact_transition_and_is_fail_open,
    test_hydration_failure_tears_down_runtime_without_durable_finalize,
)
