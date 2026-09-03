import asyncio
import base64
from pathlib import PureWindowsPath

from tools import image_source


# Valid 1x1 PNG. Small enough to keep the test self-contained.
_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class _FakeSandboxEnv:
    """Minimal sandbox environment that records executed commands."""

    def __init__(self):
        self.commands = []

    def execute(self, command):
        self.commands.append(command)
        return {
            "returncode": 0,
            "output": base64.b64encode(_PNG_1X1).decode("ascii"),
        }


def test_container_fallback_preserves_posix_path_on_windows_host(monkeypatch):
    """POSIX sandbox paths must remain POSIX when Hermes runs on Windows."""

    # Simulate pathlib behavior on a native Windows Hermes host.
    monkeypatch.setattr(image_source, "Path", PureWindowsPath)

    # Force this path down the non-local/sandbox resolution path.
    monkeypatch.setattr(
        image_source,
        "_permitted_host_read_target",
        lambda p, ctx: None,
    )
    monkeypatch.setattr(
        image_source,
        "_is_local_terminal_backend",
        lambda: False,
    )

    # Exercise the real _resolve_container_fallback implementation, but replace
    # the actual Docker environment with a deterministic fake.
    sandbox = _FakeSandboxEnv()
    monkeypatch.setattr(
        image_source,
        "_get_active_env",
        lambda task_id: sandbox,
    )

    resolved = asyncio.run(
        image_source.resolve_image_source(
            "/workspace/input/image.png",
            image_source.ResolveContext(task_id="default"),
        )
    )

    assert len(sandbox.commands) == 1

    command = sandbox.commands[0]

    # The Linux container path must cross the host -> sandbox boundary unchanged.
    assert "/workspace/input/image.png" in command

    # Regression check: Path(...) on Windows must not turn it into this.
    assert "\\workspace\\input\\image.png" not in command

    # Verify the real sandbox fallback successfully decoded/finalized the image.
    assert resolved.data == _PNG_1X1
    assert resolved.mime == "image/png"
    assert resolved.origin == "container"