import json
import os

import pytest


@pytest.fixture(autouse=True)
def session_root_cleanup(monkeypatch):
    from agent.runtime_cwd import clear_session_cwd, clear_session_file_root
    from tools import file_tools

    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.setattr(
        file_tools,
        "_SENSITIVE_PATH_PREFIXES",
        tuple(
            prefix
            for prefix in file_tools._SENSITIVE_PATH_PREFIXES
            if prefix != "/private/var/"
        ),
    )
    clear_session_cwd()
    clear_session_file_root()
    file_tools.clear_file_ops_cache()
    yield
    clear_session_cwd()
    clear_session_file_root()
    file_tools.clear_file_ops_cache()


def _set_file_root(path):
    from agent.runtime_cwd import set_session_cwd, set_session_file_root

    set_session_cwd(str(path))
    set_session_file_root(str(path))


def _payload(raw):
    return json.loads(raw)


def test_session_root_allows_relative_write_and_read(tmp_path):
    from tools.file_tools import read_file_tool, write_file_tool

    root = tmp_path / "user-a"
    root.mkdir()
    _set_file_root(root)

    written = _payload(
        write_file_tool("notes/sentinel.md", "user-a only\n", task_id="root-allow")
    )

    assert "error" not in written
    assert (root / "notes" / "sentinel.md").read_text() == "user-a only\n"

    read = _payload(read_file_tool("notes/sentinel.md", task_id="root-allow"))

    assert "error" not in read
    assert "user-a only" in read["content"]


def test_session_root_rejects_absolute_and_dotdot_escapes(tmp_path):
    from tools.file_tools import patch_tool, read_file_tool, search_tool, write_file_tool

    root = tmp_path / "user-a"
    root.mkdir()
    outside = tmp_path / "user-b-secret.txt"
    outside.write_text("outside\n")
    _set_file_root(root)

    read = _payload(read_file_tool(str(outside), task_id="root-escape"))
    write = _payload(write_file_tool("../user-b-secret.txt", "pwned\n", task_id="root-escape"))
    search = _payload(search_tool("outside", path=str(outside), task_id="root-escape"))
    patch = _payload(
        patch_tool(
            mode="replace",
            path=str(outside),
            old_string="outside",
            new_string="pwned",
            task_id="root-escape",
        )
    )

    assert "current session file root" in read["error"]
    assert "current session file root" in write["error"]
    assert "current session file root" in search["error"]
    assert "current session file root" in patch["error"]
    assert outside.read_text() == "outside\n"


def test_session_root_rejects_symlink_escape(tmp_path):
    from tools.file_tools import read_file_tool, search_tool, write_file_tool

    root = tmp_path / "user-a"
    root.mkdir()
    outside = tmp_path / "user-b-secret.txt"
    outside.write_text("outside\n")
    (root / "link.txt").symlink_to(outside)
    _set_file_root(root)

    read = _payload(read_file_tool("link.txt", task_id="root-symlink"))
    write = _payload(write_file_tool("link.txt", "pwned\n", task_id="root-symlink"))
    search = _payload(search_tool("outside", path=".", task_id="root-symlink"))

    assert "current session file root" in read["error"]
    assert "current session file root" in write["error"]
    assert "symlink" in search["error"]
    assert outside.read_text() == "outside\n"


def test_session_root_rewrites_v4a_patch_paths_to_validated_root(tmp_path):
    from tools.file_tools import patch_tool

    root = tmp_path / "user-a"
    root.mkdir()
    target = root / "app.py"
    target.write_text("print('old')\n")
    _set_file_root(root)

    result = _payload(
        patch_tool(
            mode="patch",
            patch=(
                "*** Begin Patch\n"
                "*** Update File: app.py\n"
                "@@\n"
                "-print('old')\n"
                "+print('new')\n"
                "*** End Patch\n"
            ),
            task_id="root-v4a",
        )
    )

    assert result["success"] is True
    assert target.read_text() == "print('new')\n"
    assert result["files_modified"] == [str(target.resolve())]


def test_session_root_rejects_hardlink_aliases(tmp_path):
    from tools.file_tools import read_file_tool, search_tool, write_file_tool

    root = tmp_path / "user-a"
    root.mkdir()
    outside = tmp_path / "user-b-secret.txt"
    outside.write_text("outside\n")
    link = root / "linked.txt"
    try:
        os.link(outside, link)
    except OSError as exc:
        pytest.skip(f"hardlinks unavailable on this filesystem: {exc}")
    _set_file_root(root)

    read = _payload(read_file_tool("linked.txt", task_id="root-hardlink"))
    write = _payload(write_file_tool("linked.txt", "pwned\n", task_id="root-hardlink"))
    search = _payload(search_tool("outside", path=".", task_id="root-hardlink"))

    assert "multiple filesystem links" in read["error"]
    assert "multiple filesystem links" in write["error"]
    assert "multiple filesystem links" in search["error"]
    assert outside.read_text() == "outside\n"
