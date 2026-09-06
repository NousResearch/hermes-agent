"""A single generated installer retains its scope across real Windows entry points."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pytestmark = pytest.mark.windows_only


@pytest.fixture(params=["powershell.exe", "pwsh.exe"])
def shell(request):
    executable = shutil.which(request.param)
    assert executable, f"Installer delivery must be tested with {request.param} installed"
    return executable


@pytest.fixture
def delivery(tmp_path: Path):
    builder = importlib.import_module("scripts.build_windows_installer")
    cache = tmp_path / "empty-cache"
    cache.mkdir()
    artifact = cache / "install.ps1"
    artifact.write_bytes(builder.assemble())
    home = tmp_path / "home"
    home.mkdir()
    environment = {
        **os.environ,
        "HERMES_HOME": str(home),
        "HERMES_TEST_INSTALLER": str(artifact),
        "HERMES_TEST_INSTALL_DIR": str(home / "hermes-agent"),
        "TEMP": str(tmp_path), "TMP": str(tmp_path),
        "LOCALAPPDATA": str(tmp_path / "localappdata"),
        "APPDATA": str(tmp_path / "appdata"),
        "USERPROFILE": str(tmp_path / "profile"),
        # The clean test runner strips these Windows executable-dispatch vars.
        "OS": "Windows_NT", "PATHEXT": ".COM;.EXE;.BAT;.CMD",
    }
    return artifact, environment


def _run(shell: str, delivery, command: str, **environment):
    artifact, env = delivery
    result = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", command],
        cwd=artifact.parent, env={**env, **environment},
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    return result


def _json(result):
    lines = []
    for line in result.stdout.splitlines():
        try:
            lines.append(json.loads(line))
        except ValueError:
            continue
    assert len(lines) == 1, result.stdout + result.stderr
    return lines[0]


def _manifest(shell, delivery, *, desktop=False):
    command = "& $env:HERMES_TEST_INSTALLER -Manifest"
    if desktop:
        command += " -IncludeDesktop"
    result = _run(shell, delivery, command)
    assert result.returncode == 0, result.stdout + result.stderr
    return _json(result)


@pytest.mark.parametrize("entry", ["manifest", "desktop", "protocol", "paths", "unknown", "empty", "dot-source"])
def test_cached_artifact_preserves_public_metadata_and_scope(shell, delivery, entry):
    artifact, environment = delivery
    manifest = _manifest(shell, delivery)
    names = [stage["name"] for stage in manifest["stages"]]
    desktop_names = [stage["name"] for stage in _manifest(shell, delivery, desktop=True)["stages"]]
    assert [name for name in desktop_names if name != "desktop"] == names
    assert desktop_names.count("desktop") == 1
    commands = {
        "manifest": "& $env:HERMES_TEST_INSTALLER -Manifest",
        "desktop": "& $env:HERMES_TEST_INSTALLER -Manifest -IncludeDesktop",
        "protocol": "& $env:HERMES_TEST_INSTALLER -ProtocolVersion",
        "paths": "& $env:HERMES_TEST_INSTALLER -ShowResolvedPaths",
        "unknown": "& $env:HERMES_TEST_INSTALLER -Stage fixture-unknown-stage; exit $LASTEXITCODE",
        "empty": "& $env:HERMES_TEST_INSTALLER -Stage ''; exit $LASTEXITCODE",
        "dot-source": """
. $env:HERMES_TEST_INSTALLER -IncludeDesktop
$resolved = @($InstallStages | ForEach-Object {
    $command = Get-Command $_.Worker -CommandType Function -ErrorAction Stop
    @{ name = $_.Name; worker = $command.Name; file = $command.ScriptBlock.File }
})
@{ protocol_version = $InstallStageProtocolVersion; resolved = $resolved } | ConvertTo-Json -Depth 5 -Compress
""",
    }
    result = _run(shell, delivery, commands[entry])
    assert result.returncode == (2 if entry in {"unknown", "empty"} else 0), result.stdout + result.stderr
    value = _json(result)
    checks = {
        "manifest": lambda: (
            (value["protocol_version"], [stage["name"] for stage in value["stages"]]),
            (manifest["protocol_version"], names),
        ),
        "desktop": lambda: (
            (value["protocol_version"], [stage["name"] for stage in value["stages"]]),
            (manifest["protocol_version"], desktop_names),
        ),
        "protocol": lambda: (value, manifest["protocol_version"]),
        "paths": lambda: (
            (Path(value["hermes_home"]), Path(value["install_dir"]), Path(value["temp"])),
            (Path(environment["HERMES_HOME"]), Path(environment["HERMES_TEST_INSTALL_DIR"]), artifact.parent.parent),
        ),
        "unknown": lambda: ((value["ok"], value["stage"], "unknown stage" in value["reason"]), (False, "fixture-unknown-stage", True)),
        "empty": lambda: ((value["ok"], value["stage"], "unknown stage" in value["reason"]), (False, "", True)),
        "dot-source": lambda: (
            (value["protocol_version"], [row["name"] for row in value["resolved"]],
             all(row["worker"] and Path(row["file"]) == artifact for row in value["resolved"])),
            (manifest["protocol_version"], desktop_names, True),
        ),
    }
    actual, expected = checks[entry]()
    assert actual == expected
    assert len(names) == len(set(names))
    assert list(artifact.parent.iterdir()) == [artifact], "delivery must not fetch or unpack sibling modules"
    assert not Path(environment["HERMES_TEST_INSTALL_DIR"]).exists(), "metadata entry points must never install"


@pytest.mark.parametrize("transport", ["scriptblock", "irm-iex"])
def test_loopback_delivery_executes_without_a_checkout(shell, delivery, transport):
    artifact, environment = delivery
    baseline = _manifest(shell, delivery)
    payload = artifact.read_bytes()
    # Exercise the documented irm | iex transport with a metadata-only caller.
    # The installer body is unchanged; the wrapper binds its public -Manifest flag.
    if transport == "irm-iex":
        payload = b"& {\n" + payload + b"\n} -Manifest\n"
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            if self.path != "/install.ps1":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        command = {
            "scriptblock": "& ([scriptblock]::Create((Invoke-RestMethod -Uri $env:HERMES_TEST_URL))) -Manifest",
            "irm-iex": "Invoke-RestMethod -Uri $env:HERMES_TEST_URL | Invoke-Expression",
        }[transport]
        result = _run(shell, delivery, command, HERMES_TEST_URL=f"http://127.0.0.1:{server.server_port}/install.ps1")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert result.returncode == 0, result.stdout + result.stderr
    assert _json(result) == baseline
    assert requests == ["/install.ps1"]
    assert list(artifact.parent.iterdir()) == [artifact]
    assert not Path(environment["HERMES_TEST_INSTALL_DIR"]).exists()
