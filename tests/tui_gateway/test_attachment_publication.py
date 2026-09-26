"""Attachment publication must not park other sessions behind disk I/O (#93508)."""

import base64
from concurrent.futures import ThreadPoolExecutor
import errno
import io
import os
from pathlib import Path
import threading

import pytest

from hermes_cli import clipboard, profile_incarnation
from tui_gateway import server


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8A"
    "AwMCAO+yWQAAAABJRU5ErkJggg=="
)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(server, "_hermes_home", home)
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_start_agent_build", lambda *_: None)

    def add(sid):
        profile = home / "profiles" / sid
        profile.mkdir(parents=True, exist_ok=True)
        session = {
            "agent": None, "agent_ready": threading.Event(), "agent_error": None,
            "attached_images": [], "image_counter": 0, "cwd": str(workspace),
            "history": [], "history_lock": threading.RLock(), "history_version": 0,
            "running": False, "session_key": sid, "profile_home": str(profile),
            "profile_incarnation": profile_incarnation.ensure_profile_incarnation(profile),
            "transport": None,
        }
        # Real session.create records have no _sid; the RPC id must be captured.
        server._sessions[sid] = session
        return session

    return add, workspace


def request(method, sid="first", **params):
    return server.handle_request({
        "id": method, "method": method, "params": {"session_id": sid, **params},
    })


def attachment_input(kind, tmp_path, monkeypatch, payload=PNG):
    if kind == "file-path":
        path = tmp_path / "outside.txt"
        path.write_bytes(payload)
        return "file.attach", {"path": str(path)}
    if kind == "file-data":
        return "file.attach", {"name": "outside.txt", "data_url": base64.b64encode(payload).decode()}
    if kind == "image-bytes":
        return "image.attach_bytes", {"filename": "shot.png", "data": base64.b64encode(payload).decode()}
    if kind == "clipboard":
        def extract(path):
            path.write_bytes(payload)
            return True
        monkeypatch.setattr(clipboard, "save_clipboard_image", extract)
        return "clipboard.paste", {}
    # Exercise the real PDF renderer and the shared image sink, not a mocked queue.
    import shutil
    if shutil.which("pdftoppm") is None:
        pytest.skip("PDF attachment integration requires pdftoppm")
    path = tmp_path / "input.pdf"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 20 20] /Resources << >> >>",
    ]
    pdf = b"%PDF-1.4\n"
    offsets = [0]
    for index, body in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(pdf)
    pdf += b"xref\n0 4\n0000000000 65535 f \n"
    pdf += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    pdf += f"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(pdf)
    return "pdf.attach", {"path": str(path)}


@pytest.mark.parametrize("kind", ["file-path", "file-data", "image-bytes", "clipboard", "pdf"])
@pytest.mark.parametrize("retirement", ["live", "close", "replace", "rebind", "finalize"])
def test_disk_write_leaves_sessions_available_and_rechecks_owner(
    runtime, tmp_path, monkeypatch, kind, retirement,
):
    add, _ = runtime
    session = add("first")
    other = add("other")
    home = Path(session["profile_home"])
    root = home / ("attachments" if kind.startswith("file-") else "images")
    method, params = attachment_input(kind, tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    real_open = io.open

    def blocking_open(file, mode="r", *args, **kwargs):
        handle = real_open(file, mode, *args, **kwargs)
        if isinstance(file, (str, Path)) and Path(file).parent == root and ("w" in mode or "x" in mode):
            write = handle.write

            def blocked_write(data):
                entered.set()
                assert release.wait(15), "disk write barrier not released"
                return write(data)

            handle.write = blocked_write
        return handle

    monkeypatch.setattr(io, "open", blocking_open)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(request, method, **params)
        try:
            assert entered.wait(10), "attachment did not reach disk write"

            def lookup_and_retire():
                # A real registry-reading handler must return before disk I/O resumes.
                response = request(
                    "image.attach_bytes", sid="other", filename="other.png",
                    data=base64.b64encode(PNG).decode(),
                )
                assert "error" not in response, response
                assert response["result"]["attached"] is True
                with server._sessions_lock:
                    if retirement == "close":
                        server._pop_session_by_id("first")
                    elif retirement == "replace":
                        server._sessions["first"] = {**session, "attached_images": []}
                    elif retirement == "rebind":
                        session["profile_home"] = other["profile_home"]
                        session["profile_incarnation"] = other["profile_incarnation"]
                    elif retirement == "finalize":
                        session["_finalized"] = True

            pool.submit(lookup_and_retire).result(timeout=3)
            assert not pending.done()
            assert session["attached_images"] == []
        finally:
            release.set()
        response = pending.result(timeout=10)

    if retirement == "live":
        assert "error" not in response, response
        result = response["result"]
        target = Path(result["pages"][0]["path"] if kind == "pdf" else result["path"])
        assert target.parent == root
        assert target.read_bytes().startswith(b"\x89PNG")
        assert list(root.iterdir()) == [target]
        if not kind.startswith("file-"):
            assert session["attached_images"] == [str(target)]
            assert session["image_counter"] == result["count"] == 1
    else:
        assert "error" in response, response
        assert session["attached_images"] == []
        assert session["image_counter"] == 0
        assert list(root.iterdir()) == []
        assert len(other["attached_images"]) == other["image_counter"] == 1


@pytest.mark.parametrize("kind", ["file-path", "file-data", "image-bytes", "clipboard", "pdf"])
@pytest.mark.parametrize("extra_bytes", [1, 9])
def test_attachment_caps_bound_actual_input_without_restricting_pdf_pages(
    runtime, tmp_path, monkeypatch, kind, extra_bytes,
):
    add, _ = runtime
    session = add("first")
    root = Path(session["profile_home"]) / ("attachments" if kind.startswith("file-") else "images")
    # Small configured limits exercise the exact-cap and cap+1 boundaries without
    # allocating hundreds of MB. PDF rendered pages intentionally exceed the
    # image upload cap, which applies only to image.attach_bytes.
    monkeypatch.setattr(server, "_ATTACHMENT_MAX_BYTES", len(PNG), raising=False)
    monkeypatch.setattr(server, "_ATTACH_BYTES_MAX_BYTES", len(PNG), raising=False)
    if kind == "pdf":
        method, params = attachment_input(kind, tmp_path, monkeypatch)
        read_bytes = server._read_attachment_bytes
        page_payload = [PNG]

        def rendered_page(path, limit):
            if path.name.startswith("page-"):
                path.write_bytes(page_payload[0])
            return read_bytes(path, limit)

        monkeypatch.setattr(server, "_read_attachment_bytes", rendered_page)
        monkeypatch.setattr(server, "_ATTACH_BYTES_MAX_BYTES", 1)
    else:
        method, params = attachment_input(kind, tmp_path, monkeypatch)
    accepted = request(method, **params)
    assert "error" not in accepted, accepted
    prior_files = set(root.iterdir())
    prior_queue = list(session["attached_images"])
    if kind == "pdf":
        page_payload[0] = PNG + b"x" * extra_bytes
    else:
        method, params = attachment_input(kind, tmp_path, monkeypatch, PNG + b"x" * extra_bytes)
    refused = request(method, **params)
    assert "error" in refused, refused
    assert "too large" in refused["error"]["message"]
    assert set(root.iterdir()) == prior_files
    assert session["attached_images"] == prior_queue


@pytest.mark.parametrize("method", ["file.attach", "image.attach_bytes"])
def test_publish_collision_and_partial_write_never_destroy_existing_bytes(
    runtime, monkeypatch, method,
):
    add, _ = runtime
    session = add("first")
    params = ({"name": "notes.txt", "data_url": base64.b64encode(PNG).decode()}
              if method == "file.attach" else {"filename": "shot.png", "data": base64.b64encode(PNG).decode()})
    # Launch-home publication has no named-profile lease: the atomic operation
    # itself must enforce exclusivity across independent backend processes.
    session["profile_home"] = None
    session["profile_incarnation"] = None
    root = server._hermes_home / ("attachments" if method == "file.attach" else "images")
    real_open, real_link = io.open, os.link
    written_handles = []

    def partial_open(file, mode="r", *args, **kwargs):
        handle = real_open(file, mode, *args, **kwargs)
        if isinstance(file, (str, Path)) and Path(file).parent == root and ("x" in mode or "w" in mode):
            write = handle.write
            written_handles.append(handle)

            def fail_write(data):
                write(data[:3])
                raise OSError("injected partial write")

            handle.write = fail_write
        return handle

    with monkeypatch.context() as patch:
        patch.setattr(io, "open", partial_open)
        failed = request(method, **params)
    assert "injected partial write" in failed["error"]["message"]
    assert written_handles and all(handle.closed for handle in written_handles)
    assert list(root.iterdir()) == []
    assert session["image_counter"] == 0
    assert session["attached_images"] == []

    collisions = []

    def competing_publish(source, target):
        if not collisions:
            # Another process wins after name selection, immediately before the
            # real atomic publish. There is no process-global Python lock there.
            Path(target).write_bytes(b"other publisher")
            collisions.append(Path(target))
        return real_link(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(os, "link", competing_publish)
        result = request(method, **params)["result"]
    target = Path(result["path"])
    assert collisions and collisions[0].read_bytes() == b"other publisher"
    assert target != collisions[0] and target.read_bytes() == PNG
    assert set(root.iterdir()) == {target, collisions[0]}
    if method == "image.attach_bytes":
        assert session["image_counter"] == 1
        assert session["attached_images"] == [str(target)]


@pytest.mark.parametrize("method", ["file.attach", "image.attach_bytes"])
def test_filesystem_refusing_hard_links_still_publishes_without_clobbering(
    runtime, monkeypatch, method,
):
    # FAT/exFAT, Android app data (SELinux) and some SMB/FUSE mounts refuse link().
    add, _ = runtime
    home = server._hermes_home / "profiles" / "first"
    root = home / ("attachments" if method == "file.attach" else "images")
    collisions = []

    def refuse_link(source, target):
        if Path(target).parent == root and not collisions:
            # Another process publishes this name after selection, before our publish.
            Path(target).write_bytes(b"other publisher")
            collisions.append(Path(target))
        raise PermissionError(errno.EPERM, "Operation not permitted", str(target))

    monkeypatch.setattr(os, "link", refuse_link)
    # A pre-marker named profile backfills its incarnation on this filesystem too.
    session = add("first")
    assert profile_incarnation.read_profile_incarnation(home) == session["profile_incarnation"]
    assert profile_incarnation.ensure_profile_incarnation(home) == session["profile_incarnation"]
    params = ({"name": "notes.txt", "data_url": base64.b64encode(PNG).decode()}
              if method == "file.attach" else {"filename": "shot.png", "data": base64.b64encode(PNG).decode()})
    target = Path(request(method, **params)["result"]["path"])
    assert collisions and collisions[0].read_bytes() == b"other publisher"
    assert target != collisions[0] and target.read_bytes() == PNG
    assert set(root.iterdir()) == {target, collisions[0]}


@pytest.mark.parametrize("method", ["file.attach", "image.attach"])
@pytest.mark.parametrize("retire", [False, True])
def test_workspace_paths_remain_no_copy_and_recheck_owner(
    runtime, monkeypatch, method, retire,
):
    import cli
    add, workspace = runtime
    session = add("first")
    source = workspace / "picture.png"
    source.write_bytes(PNG)
    detect = cli._detect_file_drop

    def resolve(raw):
        result = detect(raw)
        if retire:
            server._pop_session_by_id("first")
        return result

    monkeypatch.setattr(cli, "_detect_file_drop", resolve)
    # No-copy refs are not an upload; they must not read or impose the sink cap.
    monkeypatch.setattr(server, "_ATTACHMENT_MAX_BYTES", 1)
    response = request(method, path=str(source))
    if retire:
        assert "error" in response
        assert session["attached_images"] == []
    else:
        assert response["result"]["path"] == str(source)
        if method == "file.attach":
            assert response["result"]["uploaded"] is False
        else:
            assert session["attached_images"] == [str(source)]
    assert source.read_bytes() == PNG
    assert not (Path(session["profile_home"]) / "attachments").exists()
    assert not (Path(session["profile_home"]) / "images").exists()


@pytest.mark.parametrize("method", ["file.attach", "image.attach", "image.attach_bytes", "clipboard.paste", "pdf.attach"])
def test_retired_record_is_rejected_before_attachment_io(runtime, method):
    add, _ = runtime
    session = add("first")
    session["_finalized"] = True
    # The registry can still contain a finalized record awaiting the reaper.
    # Reject it before path/decode validation or native clipboard/PDF work.
    response = request(method)
    assert response["error"]["code"] == 4001
    assert session["attached_images"] == []
