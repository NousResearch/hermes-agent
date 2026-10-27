"""Container exec-read paths stay POSIX when Hermes runs natively on Windows.

Regression for #82757: on a Windows host, ``Path("/workspace/x.png")`` stringifies as
``\\workspace\\x.png``, which the Linux sandbox cannot open. Both ways a path reaches the
exec-read are covered: a container path passed as-is, and a host path under the mounted
workspace that is translated to its container location.
"""

import base64
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


@pytest.fixture(autouse=True)
def _no_real_sandbox_bringup(monkeypatch):
    import tools.terminal_tool as tt
    monkeypatch.setattr(tt, "ensure_task_env", lambda *a, **k: None)


def _reload(monkeypatch, hermes_home: Path):
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    import hermes_constants
    importlib.reload(hermes_constants)
    import tools.image_source as isrc
    importlib.reload(isrc)
    return isrc


async def _exec_read_command(isrc, src: str, **env_attrs) -> str:
    captured = {}

    def fake_execute(cmd, **kw):
        captured["cmd"] = cmd
        return {"returncode": 0, "output": base64.b64encode(PNG).decode()}

    env = SimpleNamespace(execute=fake_execute, **env_attrs)
    with patch("tools.image_source._get_active_env", return_value=env):
        res = await isrc.resolve_image_source(src, isrc.ResolveContext(task_id="t1"))
    assert res.origin == "container"
    assert res.data == PNG
    return captured["cmd"]


@pytest.mark.platforms("windows")
@pytest.mark.asyncio
async def test_posix_container_path_reaches_sandbox_unchanged(tmp_path, monkeypatch):
    isrc = _reload(monkeypatch, tmp_path / "hermes")
    cmd = await _exec_read_command(isrc, "/workspace/input/image.png")
    assert "< /workspace/input/image.png |" in cmd


@pytest.mark.platforms("windows")
@pytest.mark.asyncio
async def test_mounted_host_path_translates_to_posix_container_path(tmp_path, monkeypatch):
    isrc = _reload(monkeypatch, tmp_path / "hermes")
    project = tmp_path / "project"
    cmd = await _exec_read_command(
        isrc, str(project / "input" / "image.png"),
        host_cwd=str(project), host_cwd_mount="/workspace")
    assert "< /workspace/input/image.png |" in cmd
