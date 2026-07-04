import os
import subprocess
import sys
from pathlib import Path

from tools.cloud_file_materializer import is_cloud_file_path, materialize_for_read
from tools.code_execution_tool import _cloud_file_sitecustomize_source


def test_is_cloud_file_path_recognizes_macos_providers():
    assert is_cloud_file_path(
        "/Users/alice/Library/Mobile Documents/com~apple~CloudDocs/case.pdf"
    )
    assert is_cloud_file_path(
        "/Users/alice/Library/CloudStorage/OneDrive-Example/case.pdf"
    )
    assert not is_cloud_file_path("/Users/alice/project/case.pdf")


def test_materialize_for_read_copies_cloud_file_to_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    source_dir = tmp_path / "Library" / "Mobile Documents" / "com~apple~CloudDocs"
    source_dir.mkdir(parents=True)
    source = source_dir / "note.txt"
    source.write_text("hello from icloud", encoding="utf-8")

    cached = materialize_for_read(source)

    assert cached != source
    assert cached.read_text(encoding="utf-8") == "hello from icloud"
    assert str(cached).startswith(str(tmp_path / "hermes" / "cache" / "cloud_files"))


def test_materialize_for_read_leaves_normal_file_unchanged(tmp_path):
    source = tmp_path / "plain.txt"
    source.write_text("plain", encoding="utf-8")

    assert materialize_for_read(source) == source


def test_execute_code_sitecustomize_intercepts_read_only_cloud_open(tmp_path):
    hermes_home = tmp_path / "hermes"
    cloud_dir = tmp_path / "Library" / "Mobile Documents" / "com~apple~CloudDocs"
    cloud_dir.mkdir(parents=True)
    source = cloud_dir / "note.txt"
    source.write_text("sitecustomize ok", encoding="utf-8")

    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(_cloud_file_sitecustomize_source(), encoding="utf-8")

    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_home)
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), env.get("PYTHONPATH", "")])

    code = (
        "from pathlib import Path\n"
        f"p = Path({str(source)!r})\n"
        "print(open(p).read())\n"
        "print(p.read_text())\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=True,
    )

    assert proc.stdout.splitlines() == ["sitecustomize ok", "sitecustomize ok"]
    assert (hermes_home / "cache" / "cloud_files").exists()

