"""File attachment uploads retain their bytes in the shared profile namespace."""

import base64
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import pytest

from tui_gateway import server


@pytest.fixture
def sessions(tmp_path, monkeypatch):
    records = {
        sid: {
            "profile_home": str(tmp_path / "profile"),
            "cwd": str(tmp_path / "workspace"),
            "attached_images": [],
            "running": False,
            "history_lock": threading.Lock(),
        }
        for sid in ("first", "second")
    }
    monkeypatch.setattr(server, "_sessions", records)
    return records


def upload(session_id, name, payload):
    return server._methods["file.attach"](1, {
        "session_id": session_id,
        "name": name,
        "data_url": "data:text/plain;base64," + base64.b64encode(payload).decode(),
    })


@pytest.mark.parametrize("same_session", [True, False])
def test_concurrent_same_name_uploads_keep_distinct_bytes_and_refs(
    sessions, tmp_path, monkeypatch, same_session,
):
    target = tmp_path / "profile" / "attachments" / "report (notes).txt"
    ready = threading.Barrier(2)
    original_open = Path.open

    def concurrent_open(path, mode="r", *args, **kwargs):
        if Path(path) == target and mode in {"wb", "xb"}:
            # Both requests select the same candidate before either can create it.
            ready.wait(timeout=10)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", concurrent_open)
    payloads = (b"first distinct upload", b"second distinct upload")
    owners = ("first", "first" if same_session else "second")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(upload, owner, target.name, data)
                   for owner, data in zip(owners, payloads)]
        responses = [future.result(timeout=10) for future in futures]

    paths = []
    for response, data in zip(responses, payloads):
        assert "error" not in response, response
        result = response["result"]
        path = Path(result["path"])
        paths.append(path)
        assert path.read_bytes() == data
        assert path.parent == target.parent
        assert path.suffix == target.suffix
        assert result["name"] == path.name
        assert result["uploaded"] is True
        assert result["ref_path"] == str(path)
        assert result["ref_text"] == f"@file:`{path}`"
    assert len(set(paths)) == len(payloads)
    assert set(target.parent.iterdir()) == set(paths)


def test_failed_write_removes_only_its_partial_upload(sessions, tmp_path, monkeypatch):
    payload = b"interrupted upload"
    original_open = Path.open
    partial_written = threading.Event()
    successful_write = threading.Event()
    root = tmp_path / "profile" / "attachments"

    class InterruptedWrite:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def write(self, data):
            self.stream.write(data[:5])
            self.stream.flush()
            partial_written.set()
            assert successful_write.wait(timeout=10)
            raise OSError("attachment write interrupted")

    def interrupted_open(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        if Path(path) == root / "partial.txt" and mode in {"wb", "xb"}:
            return InterruptedWrite(stream)
        return stream

    monkeypatch.setattr(Path, "open", interrupted_open)
    with ThreadPoolExecutor(max_workers=1) as pool:
        failed = pool.submit(upload, "first", "partial.txt", payload)
        try:
            assert partial_written.wait(timeout=10)
            success = upload("second", "complete.txt", b"complete upload")
            assert "error" not in success, success
        finally:
            successful_write.set()
        assert failed.result(timeout=10)["error"]["code"] == 5028

    survivor = Path(success["result"]["path"])
    assert survivor.read_bytes() == b"complete upload"
    assert set(root.iterdir()) == {survivor}


def test_cross_session_attached_images_allocate_unique_paths(sessions, tmp_path):
    session_a = sessions["first"]
    session_b = sessions["second"]
    img_bytes_a = b"SESSION_A_IMAGE_PNG"
    img_bytes_b = b"SESSION_B_IMAGE_PNG"

    path_a = server._queue_attached_image(session_a, img_bytes_a, ".png", prefix="upload")
    # Reset image_counter in session_b to 0 to simulate concurrent/fresh session in same profile
    session_b["image_counter"] = 0
    path_b = server._queue_attached_image(session_b, img_bytes_b, ".png", prefix="upload")

    assert path_a != path_b
    assert path_a.read_bytes() == img_bytes_a
    assert path_b.read_bytes() == img_bytes_b
    assert path_a.exists() and path_b.exists()



def test_upload_never_writes_through_a_planted_symlink(sessions, tmp_path):
    """A repo-shipped dangling symlink at the candidate name must not redirect the upload."""
    root = tmp_path / "profile" / "attachments"
    root.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    (root / "notes.txt").symlink_to(outside)

    response = upload("first", "notes.txt", b"user upload")

    assert "error" not in response, response
    stored = Path(response["result"]["path"])
    assert not outside.exists()
    assert stored.parent == root.resolve() and stored.name == "notes-2.txt"
    assert stored.read_bytes() == b"user upload"
