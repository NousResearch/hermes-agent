"""Unit tests for FileSyncManager host-path containment (no fcntl required)."""

from pathlib import Path

from tools.environments.file_sync import FileSyncManager


def _write_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _make_manager(tmp_path: Path, file_mapping=None) -> FileSyncManager:
    mapping = file_mapping or []

    def _get_files():
        return list(mapping)

    return FileSyncManager(
        upload_fn=lambda *_a, **_k: None,
        delete_fn=lambda *_a, **_k: None,
        get_files_fn=_get_files,
        bulk_download_fn=None,
    )


def test_infer_matching_prefix(tmp_path):
    host_file = tmp_path / "host" / "skills" / "a.py"
    _write_file(host_file, b"content")
    mapping = [(str(host_file), "/root/.hermes/skills/a.py")]
    mgr = _make_manager(tmp_path, file_mapping=mapping)
    result = mgr._infer_host_path(
        "/root/.hermes/skills/b.py",
        file_mapping=mapping,
    )
    assert result == str(tmp_path / "host" / "skills" / "b.py")


def test_infer_rejects_traversal_suffix(tmp_path):
    host_file = tmp_path / "host" / "skills" / "a.py"
    _write_file(host_file, b"content")
    mapping = [(str(host_file), "/root/.hermes/skills/a.py")]
    mgr = _make_manager(tmp_path, file_mapping=mapping)
    result = mgr._infer_host_path(
        "/root/.hermes/skills/../../.ssh/authorized_keys",
        file_mapping=mapping,
    )
    assert result is None


def test_host_path_containment_rejects_escape(tmp_path):
    host_file = tmp_path / "host" / "skills" / "a.py"
    _write_file(host_file, b"content")
    mapping = [(str(host_file), "/root/.hermes/skills/a.py")]
    mgr = _make_manager(tmp_path, file_mapping=mapping)
    assert mgr._host_path_is_contained(
        str(tmp_path / "host" / "skills" / "b.py"),
        mapping,
    )
    assert not mgr._host_path_is_contained(
        str(tmp_path / "host" / ".ssh" / "authorized_keys"),
        mapping,
    )
