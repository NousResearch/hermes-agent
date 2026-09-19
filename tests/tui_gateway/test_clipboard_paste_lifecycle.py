"""Clipboard extraction must leave other sessions and profile lifecycle free."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import base64
import shutil
import threading

import pytest

from hermes_cli import clipboard, profile_incarnation
from hermes_constants import clear_named_profile_deleted, mark_named_profile_deleted
from tui_gateway import server


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8A"
    "AwMCAO+yWQAAAABJRU5ErkJggg=="
)
_BARRIER_TIMEOUT = 10


@pytest.fixture
def clipboard_sessions(tmp_path, monkeypatch):
    os_home = tmp_path / "os-home"
    os_home.mkdir()
    hermes_home = os_home / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: os_home))
    monkeypatch.setattr(server, "_hermes_home", hermes_home)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_args: None)
    records = []

    def create(name, *, launch_home=False):
        profile_home = hermes_home if launch_home else hermes_home / "profiles" / name
        profile_home.mkdir(parents=True, exist_ok=True)
        incarnation = profile_incarnation.ensure_profile_incarnation(profile_home)
        sid = f"clipboard-{name}"
        record = {
            "agent": None,
            "agent_ready": threading.Event(),
            "agent_error": None,
            "attached_images": [],
            "cwd": str(profile_home),
            "history": [],
            "history_lock": threading.RLock(),
            "history_version": 0,
            "image_counter": 0,
            "profile_home": str(profile_home),
            "profile_incarnation": incarnation,
            "running": False,
            "session_key": sid,
            "transport": None,
        }
        with server._sessions_lock:
            server._sessions[sid] = record
        records.append(sid)
        return sid, record, profile_home

    yield create

    with server._sessions_lock:
        for sid in records:
            server._sessions.pop(sid, None)


def _assert_staging_cleaned(paths):
    assert paths
    assert all(not path.parent.exists() for path in paths)


def test_other_session_attaches_while_clipboard_extraction_is_waiting(
    clipboard_sessions, monkeypatch,
):
    sid, session, profile_home = clipboard_sessions("paste")
    other_sid, other_session, _ = clipboard_sessions("other")
    entered = threading.Event()
    release = threading.Event()
    staging_paths = []

    def blocked_extract(path, *, create_parent=True):
        staging_paths.append(path)
        assert create_parent is False
        assert not path.is_relative_to(profile_home)
        entered.set()
        assert release.wait(_BARRIER_TIMEOUT), "clipboard release barrier timed out"
        path.write_bytes(PNG_BYTES)
        return True

    monkeypatch.setattr(clipboard, "save_clipboard_image", blocked_extract)
    with ThreadPoolExecutor(max_workers=2) as pool:
        paste = pool.submit(server._methods["clipboard.paste"], "paste", {"session_id": sid})
        try:
            assert entered.wait(_BARRIER_TIMEOUT)
            other = pool.submit(
                server._methods["image.attach_bytes"],
                "other",
                {
                    "session_id": other_sid,
                    "content_base64": base64.b64encode(PNG_BYTES).decode(),
                    "filename": "other.png",
                },
            )
            attached = other.result(timeout=_BARRIER_TIMEOUT)
            assert attached["result"]["attached"] is True
            assert not paste.done()
            assert other_session["image_counter"] == 1
        finally:
            release.set()
        result = paste.result(timeout=_BARRIER_TIMEOUT)["result"]

    assert result["attached"] is True
    assert Path(result["path"]).read_bytes() == PNG_BYTES
    assert session["attached_images"] == [result["path"]]
    assert session["image_counter"] == result["count"] == 1
    assert not session["agent_ready"].is_set()
    _assert_staging_cleaned(staging_paths)


@pytest.mark.parametrize("recreate", [False, True])
def test_clipboard_rejects_profile_deleted_during_extraction(
    clipboard_sessions, monkeypatch, recreate,
):
    sid, session, profile_home = clipboard_sessions("retired")
    generation = session["profile_incarnation"]
    staging_paths = []

    def delete_during_extract(path, *, create_parent=True):
        staging_paths.append(path)
        path.write_bytes(PNG_BYTES)
        with profile_incarnation.profile_incarnation_lease(profile_home, generation):
            mark_named_profile_deleted(profile_home)
            shutil.rmtree(profile_home)
            if recreate:
                profile_home.mkdir()
                replacement = profile_incarnation.write_fresh_profile_incarnation(profile_home)
                assert replacement != generation
                clear_named_profile_deleted(profile_home)
        return True

    monkeypatch.setattr(clipboard, "save_clipboard_image", delete_during_extract)
    response = server._methods["clipboard.paste"]("stale", {"session_id": sid})
    assert response["error"]["code"] == 4041
    assert response["error"]["message"] == "profile incarnation changed during clipboard paste"
    assert session["image_counter"] == 0
    assert session["attached_images"] == []
    assert not (profile_home / "images").exists()
    assert profile_home.exists() is recreate
    _assert_staging_cleaned(staging_paths)

    if recreate:
        fresh_sid, fresh, _ = clipboard_sessions("retired")

        def current_extract(path, *, create_parent=True):
            staging_paths.append(path)
            path.write_bytes(PNG_BYTES)
            return True

        monkeypatch.setattr(clipboard, "save_clipboard_image", current_extract)
        result = server._methods["clipboard.paste"]("fresh", {"session_id": fresh_sid})["result"]
        assert result["attached"] is True
        assert fresh["attached_images"] == [result["path"]]
        assert Path(result["path"]).read_bytes() == PNG_BYTES
        _assert_staging_cleaned(staging_paths)


@pytest.mark.parametrize(
    ("has_image", "message"),
    [(False, "No image found in clipboard"), (True, "Clipboard has image but extraction failed")],
)
def test_clipboard_failure_keeps_counter_and_cleans_partial_staging(
    clipboard_sessions, monkeypatch, has_image, message,
):
    sid, session, profile_home = clipboard_sessions("failure")
    session["image_counter"] = 3
    staging_paths = []

    def fail_extract(path, *, create_parent=True):
        staging_paths.append(path)
        path.write_bytes(b"partial")
        return False

    monkeypatch.setattr(clipboard, "save_clipboard_image", fail_extract)
    monkeypatch.setattr(clipboard, "has_clipboard_image", lambda: has_image)
    response = server._methods["clipboard.paste"]("failed", {"session_id": sid})
    assert response["result"] == {"attached": False, "message": message}
    assert session["image_counter"] == 3
    assert session["attached_images"] == []
    assert not (profile_home / "images").exists()
    _assert_staging_cleaned(staging_paths)


@pytest.mark.parametrize("failure_stage", ["extraction", "publication"])
def test_clipboard_write_errors_preserve_counter_and_cleanup(
    clipboard_sessions, monkeypatch, failure_stage,
):
    sid, session, profile_home = clipboard_sessions("write-error")
    session["image_counter"] = 3
    staging_paths = []
    write_bytes = Path.write_bytes

    def extract(path, *, create_parent=True):
        staging_paths.append(path)
        write_bytes(path, PNG_BYTES)
        if failure_stage == "extraction":
            raise PermissionError("clipboard write denied")
        return True

    def final_write(path, data):
        written = write_bytes(path, data)
        if path.parent == profile_home / "images":
            raise PermissionError("clipboard write denied")
        return written

    monkeypatch.setattr(clipboard, "save_clipboard_image", extract)
    monkeypatch.setattr(Path, "write_bytes", final_write)
    with pytest.raises(PermissionError, match="clipboard write denied"):
        server._methods["clipboard.paste"]("write-error", {"session_id": sid})
    assert session["image_counter"] == 3
    assert session["attached_images"] == []
    assert not list((profile_home / "images").glob("*"))
    _assert_staging_cleaned(staging_paths)


def test_clipboard_rejects_session_rebound_during_extraction(
    clipboard_sessions, monkeypatch,
):
    sid, session, original_home = clipboard_sessions("original")
    _, replacement, replacement_home = clipboard_sessions("replacement")
    staging_paths = []

    def rebind_during_extract(path, *, create_parent=True):
        staging_paths.append(path)
        path.write_bytes(PNG_BYTES)
        session["profile_home"] = replacement["profile_home"]
        session["profile_incarnation"] = replacement["profile_incarnation"]
        return True

    monkeypatch.setattr(clipboard, "save_clipboard_image", rebind_during_extract)
    response = server._methods["clipboard.paste"]("rebound", {"session_id": sid})
    assert response["error"]["code"] == 4041
    assert session["image_counter"] == 0
    assert session["attached_images"] == []
    assert not (original_home / "images").exists()
    assert not (replacement_home / "images").exists()
    _assert_staging_cleaned(staging_paths)


@pytest.mark.parametrize("home_value", [None, ""])
def test_clipboard_without_named_profile_uses_launch_home(
    clipboard_sessions, monkeypatch, home_value,
):
    sid, session, profile_home = clipboard_sessions("launch", launch_home=True)
    session.pop("profile_home")
    session.pop("profile_incarnation")
    if home_value is not None:
        session["profile_home"] = home_value
        session["profile_incarnation"] = ""
    staging_paths = []

    def extract(path, *, create_parent=True):
        staging_paths.append(path)
        path.write_bytes(PNG_BYTES)
        return True

    monkeypatch.setattr(clipboard, "save_clipboard_image", extract)
    response = server._methods["clipboard.paste"]("launch", {"session_id": sid})["result"]
    assert response["attached"] is True
    assert Path(response["path"]).parent == profile_home / "images"
    assert Path(response["path"]).read_bytes() == PNG_BYTES
    assert session["image_counter"] == 1
    _assert_staging_cleaned(staging_paths)


@pytest.mark.parametrize("retirement", ["close", "replace", "finalize"])
def test_clipboard_does_not_publish_after_session_retirement(
    clipboard_sessions, monkeypatch, retirement,
):
    sid, session, profile_home = clipboard_sessions("retirement")
    entered = threading.Event()
    release = threading.Event()
    staging_paths = []
    replacement = None

    def blocked_extract(path, *, create_parent=True):
        staging_paths.append(path)
        entered.set()
        assert release.wait(_BARRIER_TIMEOUT), "clipboard release barrier timed out"
        path.write_bytes(PNG_BYTES)
        return True

    monkeypatch.setattr(clipboard, "save_clipboard_image", blocked_extract)
    with ThreadPoolExecutor(max_workers=2) as pool:
        paste = pool.submit(server._methods["clipboard.paste"], "paste", {"session_id": sid})
        try:
            assert entered.wait(_BARRIER_TIMEOUT)
            if retirement == "close":
                close = pool.submit(server._methods["session.close"], "close", {"session_id": sid})
                assert close.result(timeout=_BARRIER_TIMEOUT)["result"]["closed"] is True
                assert sid not in server._sessions
                assert session["_finalized"] is True
            elif retirement == "replace":
                replacement_sid, replacement, _ = clipboard_sessions("retirement")
                assert replacement_sid == sid
                assert replacement["profile_incarnation"] == session["profile_incarnation"]
                assert not session.get("_finalized")
            else:
                finalized = pool.submit(server._finalize_session, session)
                finalized.result(timeout=_BARRIER_TIMEOUT)
                assert server._sessions[sid] is session
                assert session["_finalized"] is True
            assert not paste.done()
        finally:
            release.set()
        response = paste.result(timeout=_BARRIER_TIMEOUT)

    assert response["error"]["code"] == 4001
    assert response["error"]["message"] == "session not found"
    assert session["image_counter"] == 0
    assert session["attached_images"] == []
    if replacement is not None:
        assert replacement["image_counter"] == 0
        assert replacement["attached_images"] == []
    assert not (profile_home / "images").exists()
    _assert_staging_cleaned(staging_paths)
