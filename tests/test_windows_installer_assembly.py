"""The installer builder has one confined, complete, deterministic source graph."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _authoring(root: Path, contents: dict[str, bytes], *, raw_manifest=None, **updates) -> Path:
    source = root / "source"
    source.mkdir(parents=True)
    for name, content in contents.items():
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    manifest = {"version": 1, "entry": "install.ps1", "files": list(contents), "kill_track": {}}
    manifest.update(updates)
    (root / "manifest.json").write_text(raw_manifest or json.dumps(manifest), encoding="utf-8")
    return root


def _cli(root: Path, *args: str):
    return subprocess.run(
        [sys.executable, str(root / "scripts/build_windows_installer.py"), *args],
        cwd=root, capture_output=True, text=True, timeout=20,
    )


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"], ids=["lf", "windows-crlf"])
def test_composition_and_cli_check_preserve_the_declared_bytes(tmp_path: Path, newline: bytes):
    builder = importlib.import_module("scripts.build_windows_installer")
    scripts = tmp_path / "scripts"
    inputs = {
        "install.ps1": b"param([switch]$Manifest)\n# @include topic.ps1\nWrite-Output $Message\n",
        "topic.ps1": b"$Message = 'fixture'\n# @include nested/helper.ps1\n",
        "nested/helper.ps1": b"function Get-Fixture { return $Message }\n",
    }
    authoring = _authoring(scripts / "windows-installer", {
        name: content.replace(b"\n", newline) for name, content in inputs.items()
    })
    expected = (
        b"param([switch]$Manifest)\n$Message = 'fixture'\n"
        b"function Get-Fixture { return $Message }\nWrite-Output $Message\n"
    )
    assert builder.assemble(authoring) == expected
    assert builder.assemble(authoring) == expected
    # Exercise the real CLI from a separate, relocatable checkout layout.
    shutil.copyfile(REPO / "scripts/build_windows_installer.py", scripts / "build_windows_installer.py")
    assert _cli(tmp_path, "--check").returncode != 0
    generated = _cli(tmp_path)
    assert generated.returncode == 0, generated.stdout + generated.stderr
    artifact = scripts / "install.ps1"
    assert artifact.read_bytes() == expected
    assert _cli(tmp_path, "--check").returncode == 0
    artifact.write_bytes(expected + b"# stale local edit\n")
    assert _cli(tmp_path, "--check").returncode != 0
    assert artifact.read_bytes().endswith(b"# stale local edit\n"), "check must never rewrite a stale artifact"
    assert _cli(tmp_path).returncode == 0
    (authoring / "source/nested/helper.ps1").write_bytes(b"function Get-Fixture { return 'changed' }\n")
    assert _cli(tmp_path, "--check").returncode != 0
    assert artifact.read_bytes() == expected


@pytest.mark.parametrize("sources,updates,error", [
    ({"install.ps1": b"# @include topic.ps1\n# @include topic.ps1\n", "topic.ps1": b"# helper\n"}, {}, "more than once"),
    ({"install.ps1": b"# @include topic.ps1\n", "topic.ps1": b"# @include install.ps1\n"}, {}, "cycle"),
    ({"install.ps1": b"# @include missing.ps1\n"}, {}, "not declared"),
    ({"install.ps1": b"# entry\n", "unused.ps1": b"# unused\n"}, {}, "orphan"),
    ({"install.ps1": b"# entry\n", "hidden.txt": b"# undeclared\n"}, {"files": ["install.ps1"]}, "undeclared"),
    ({"install.ps1": b"# entry\n"}, {"files": ["install.ps1", "install.ps1"]}, "duplicate"),
    ({"install.ps1": b"# entry\n"}, {"files": ["install.ps1", "Install.ps1"]}, "case-colliding"),
    ({"install.ps1": b"# entry\n"}, {"files": ["install.ps1", "../outside.ps1"]}, "invalid source path"),
    ({"install.ps1": b"# entry\n"}, {"files": ["install.ps1", "/outside.ps1"]}, "invalid source path"),
    ({"install.ps1": b"# entry\n"}, {"files": ["install.ps1", "nested\\helper.ps1"]}, "invalid source path"),
    ({"install.ps1": b"# @include topic.ps1 extra\n", "topic.ps1": b"# helper\n"}, {}, "malformed"),
    ({"install.ps1": b"\xef\xbb\xbf# entry\n"}, {}, "BOM"),
    ({"install.ps1": b"# entry\r# tail\n"}, {}, "newline"),
    ({"install.ps1": b"# entry"}, {}, "newline"),
    ({"install.ps1": b"# line\n" * 2001}, {}, "ceiling"),
    ({"install.ps1": b"# line\n" * 2002}, {"kill_track": {"install.ps1": 2001}}, "ceiling"),
    ({"install.ps1": b"# entry\n"}, {"kill_track": {"install.ps1": 2001}}, "completed"),
    ({"install.ps1": b"# @include topic.ps1\n", "topic.ps1": b"# helper\n"}, {"files": ["topic.ps1", "install.ps1"]}, "traversal order"),
    ({"install.ps1": b"# entry\n"}, {"raw_manifest": '{"version":1,"version":1,"entry":"install.ps1","files":["install.ps1"],"kill_track":{}}'}, "duplicate manifest key"),
    ({"install.ps1": b"# entry\n"}, {"raw_manifest": '{"version":1,"entry":"install.ps1","files":["install.ps1"],"kill_track":{"install.ps1":2001,"install.ps1":2002}}'}, "duplicate manifest key"),
])
def test_invalid_graph_cannot_publish_an_installer(tmp_path: Path, sources, updates, error):
    builder = importlib.import_module("scripts.build_windows_installer")
    authoring = _authoring(tmp_path / "authoring", sources, **updates)
    with pytest.raises(builder.AssemblyError, match=error):
        builder.assemble(authoring)


@pytest.mark.windows_only
@pytest.mark.parametrize("site", ["source-root", "nested-source"])
def test_windows_junctions_cannot_alias_authored_source(tmp_path: Path, site: str):
    builder = importlib.import_module("scripts.build_windows_installer")
    if site == "source-root":
        authoring = _authoring(tmp_path / "authoring", {"install.ps1": b"# fixture\n"})
        junction = authoring / "source"
        target = authoring / "actual-source"
        junction.rename(target)
    else:
        authoring = _authoring(tmp_path / "authoring", {
            "install.ps1": b"# @include actual/topic.ps1\n# @include linked/topic.ps1\n",
            "actual/topic.ps1": b"# fixture\n",
        }, files=["install.ps1", "actual/topic.ps1", "linked/topic.ps1"])
        junction = authoring / "source/linked"
        target = authoring / "source/actual"
    powershell = shutil.which("powershell.exe")
    assert powershell
    result = subprocess.run(
        [powershell, "-NoProfile", "-Command",
         "$ErrorActionPreference = 'Stop'; New-Item -ItemType Junction -Path $env:HERMES_TEST_LINK -Target $env:HERMES_TEST_TARGET | Out-Null"],
        env={**os.environ, "HERMES_TEST_LINK": str(junction), "HERMES_TEST_TARGET": str(target)},
        capture_output=True, text=True, timeout=15, creationflags=subprocess.CREATE_NO_WINDOW,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    try:
        with pytest.raises(builder.AssemblyError, match="link"):
            builder.assemble(authoring)
    finally:
        junction.rmdir()
    assert target.is_dir()
