"""Tests for the per-backend file extraction API (issue #466).

Covers BaseEnvironment.fetch_file / fetch_file_size (base64-over-exec
default), the SSH cat-over-ControlMaster override, the Docker bind-mount /
docker cp override, the Daytona SDK override, and the Local passthrough.
"""

import base64
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.environments.base import BaseEnvironment, FileFetchError


# ---------------------------------------------------------------------------
# Base implementation (base64 over the exec channel)
# ---------------------------------------------------------------------------

class _FakeExecEnvironment(BaseEnvironment):
    """BaseEnvironment with execute() stubbed to canned results."""

    def __init__(self, results):
        # Skip BaseEnvironment.__init__ side effects — set the attributes
        # fetch_file/fetch_file_size actually use.
        self.timeout = 60
        self._results = list(results)
        self.commands = []

    def execute(self, command, cwd="", **kwargs):
        self.commands.append(command)
        return self._results.pop(0)

    def cleanup(self):
        pass


class TestBaseFetchFile:
    def test_fetch_file_decodes_marker_fenced_payload(self, tmp_path):
        payload = b"%PDF-1.4 binary\x00\x01 content"
        encoded = base64.b64encode(payload).decode()

        def fake_execute(command, cwd="", **kwargs):
            marker = command.split("echo ")[1].split(" &&")[0]
            return {"output": f"{marker}\n{encoded}\n{marker}\n", "returncode": 0}

        env = _FakeExecEnvironment([])
        env.execute = fake_execute
        dest = tmp_path / "out.pdf"
        env.fetch_file("/workspace/report.pdf", str(dest))
        assert dest.read_bytes() == payload

    def test_fetch_file_ignores_noise_outside_markers(self, tmp_path):
        payload = b"hello world"
        encoded = base64.b64encode(payload).decode()

        def fake_execute(command, cwd="", **kwargs):
            marker = command.split("echo ")[1].split(" &&")[0]
            return {
                "output": f"motd: welcome!\n{marker}\n{encoded}\n{marker}\ntrailing\n",
                "returncode": 0,
            }

        env = _FakeExecEnvironment([])
        env.execute = fake_execute
        dest = tmp_path / "out.txt"
        env.fetch_file("/tmp/hello.txt", str(dest))
        assert dest.read_bytes() == payload

    def test_fetch_file_missing_file_raises(self, tmp_path):
        env = _FakeExecEnvironment([{"output": "", "returncode": 1}])
        with pytest.raises(FileFetchError, match="could not read"):
            env.fetch_file("/nope.txt", str(tmp_path / "out"))

    def test_fetch_file_corrupt_payload_raises(self, tmp_path):
        def fake_execute(command, cwd="", **kwargs):
            marker = command.split("echo ")[1].split(" &&")[0]
            return {"output": f"{marker}\nnot!!valid@@b64\n{marker}\n", "returncode": 0}

        env = _FakeExecEnvironment([])
        env.execute = fake_execute
        with pytest.raises(FileFetchError, match="corrupted"):
            env.fetch_file("/tmp/x.bin", str(tmp_path / "out"))

    def test_fetch_file_size_parses_last_digit_token(self):
        env = _FakeExecEnvironment([{"output": "banner\n  1234\n", "returncode": 0}])
        assert env.fetch_file_size("/tmp/x.bin") == 1234

    def test_fetch_file_size_missing_returns_none(self):
        env = _FakeExecEnvironment([{"output": "", "returncode": 1}])
        assert env.fetch_file_size("/nope") is None

    def test_remote_home_defaults_to_none(self):
        env = _FakeExecEnvironment([])
        assert env.remote_home is None


# ---------------------------------------------------------------------------
# SSH override (cat over the ControlMaster connection)
# ---------------------------------------------------------------------------

class TestSSHFetchFile:
    def _make_env(self):
        from tools.environments.ssh import SSHEnvironment

        env = SSHEnvironment.__new__(SSHEnvironment)
        env.host = "example.com"
        env.user = "worker"
        env.port = 22
        env.key_path = ""
        env.control_socket = Path("/tmp/hermes-ssh/test.sock")
        return env

    def test_fetch_file_streams_cat_to_dest(self, tmp_path):
        env = self._make_env()
        dest = tmp_path / "artifact.bin"
        payload = b"\x00\x01binary"

        def fake_run(cmd, stdin=None, stdout=None, stderr=None, timeout=None):
            assert cmd[-1] == "cat /home/worker/artifact.bin"
            assert any("ControlPath=" in part for part in cmd)
            stdout.write(payload)
            return subprocess.CompletedProcess(cmd, 0, stderr=b"")

        with patch("tools.environments.ssh.subprocess.run", side_effect=fake_run):
            env.fetch_file("/home/worker/artifact.bin", str(dest))
        assert dest.read_bytes() == payload

    def test_fetch_file_failure_raises_and_removes_partial(self, tmp_path):
        env = self._make_env()
        dest = tmp_path / "artifact.bin"

        def fake_run(cmd, stdin=None, stdout=None, stderr=None, timeout=None):
            stdout.write(b"partial")
            return subprocess.CompletedProcess(cmd, 1, stderr=b"cat: no such file")

        with patch("tools.environments.ssh.subprocess.run", side_effect=fake_run):
            with pytest.raises(FileFetchError, match="no such file"):
                env.fetch_file("/home/worker/missing.bin", str(dest))
        assert not dest.exists()

    def test_fetch_file_quotes_hostile_remote_path(self, tmp_path):
        env = self._make_env()
        seen = {}

        def fake_run(cmd, stdin=None, stdout=None, stderr=None, timeout=None):
            seen["remote_cmd"] = cmd[-1]
            return subprocess.CompletedProcess(cmd, 0, stderr=b"")

        with patch("tools.environments.ssh.subprocess.run", side_effect=fake_run):
            env.fetch_file("/tmp/$(rm -rf ~)/x.txt", str(tmp_path / "x"))
        assert "$(rm" in seen["remote_cmd"]
        assert seen["remote_cmd"].startswith("cat '")


# ---------------------------------------------------------------------------
# Docker override (bind-mount view, docker cp fallback)
# ---------------------------------------------------------------------------

class TestDockerFetchFile:
    def _make_env(self, home_dir=None, workspace_dir=None):
        from tools.environments.docker import DockerEnvironment

        env = DockerEnvironment.__new__(DockerEnvironment)
        env._home_dir = home_dir
        env._workspace_dir = workspace_dir
        env._container_id = "cafebabe1234"
        env._docker_exe = "docker"
        return env

    def test_bind_mounted_root_path_copies_from_host(self, tmp_path):
        home = tmp_path / "home"
        home.mkdir()
        (home / "report.pdf").write_bytes(b"%PDF data")
        env = self._make_env(home_dir=str(home))
        dest = tmp_path / "out.pdf"

        with patch("tools.environments.docker.subprocess.run") as run_mock:
            env.fetch_file("/root/report.pdf", str(dest))
        run_mock.assert_not_called()
        assert dest.read_bytes() == b"%PDF data"

    def test_traversal_out_of_mount_does_not_map(self, tmp_path):
        home = tmp_path / "home"
        home.mkdir()
        env = self._make_env(home_dir=str(home))
        # /root/../etc/passwd normalizes to /etc/passwd -> no host mapping,
        # falls through to docker cp.
        assert env._host_path_for("/root/../etc/passwd") is None

    def test_unmounted_path_uses_docker_cp(self, tmp_path):
        env = self._make_env()
        dest = tmp_path / "out.txt"

        def fake_run(cmd, capture_output=None, text=None, timeout=None, stdin=None):
            assert cmd[:3] == ["docker", "cp", "-L"]
            assert cmd[3] == "cafebabe1234:/var/tmp/out.txt"
            Path(cmd[4]).write_bytes(b"copied")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("tools.environments.docker.subprocess.run", side_effect=fake_run):
            env.fetch_file("/var/tmp/out.txt", str(dest))
        assert dest.read_bytes() == b"copied"

    def test_docker_cp_failure_raises(self, tmp_path):
        env = self._make_env()

        def fake_run(cmd, capture_output=None, text=None, timeout=None, stdin=None):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="no such file")

        with patch("tools.environments.docker.subprocess.run", side_effect=fake_run):
            with pytest.raises(FileFetchError, match="no such file"):
                env.fetch_file("/nope.txt", str(tmp_path / "out"))

    def test_docker_cp_directory_result_rejected(self, tmp_path):
        env = self._make_env()
        dest = tmp_path / "out"

        def fake_run(cmd, capture_output=None, text=None, timeout=None, stdin=None):
            Path(cmd[4]).mkdir()
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("tools.environments.docker.subprocess.run", side_effect=fake_run):
            with pytest.raises(FileFetchError, match="not a regular file"):
                env.fetch_file("/some/dir", str(dest))
        assert not dest.exists()


# ---------------------------------------------------------------------------
# Daytona override (SDK download)
# ---------------------------------------------------------------------------

class TestDaytonaFetchFile:
    def _make_env(self):
        from tools.environments.daytona import DaytonaEnvironment

        env = DaytonaEnvironment.__new__(DaytonaEnvironment)
        env._sandbox = MagicMock()
        return env

    def test_fetch_file_uses_sdk_download(self, tmp_path):
        env = self._make_env()
        dest = tmp_path / "out.csv"
        env._sandbox.fs.download_file.side_effect = (
            lambda remote, local: Path(local).write_bytes(b"a,b\n1,2")
        )
        env.fetch_file("/home/daytona/data.csv", str(dest))
        env._sandbox.fs.download_file.assert_called_once_with(
            "/home/daytona/data.csv", str(dest)
        )
        assert dest.read_bytes() == b"a,b\n1,2"

    def test_fetch_file_sdk_error_raises(self, tmp_path):
        env = self._make_env()
        env._sandbox.fs.download_file.side_effect = RuntimeError("sandbox stopped")
        with pytest.raises(FileFetchError, match="sandbox stopped"):
            env.fetch_file("/home/daytona/data.csv", str(tmp_path / "out"))


# ---------------------------------------------------------------------------
# Local passthrough
# ---------------------------------------------------------------------------

class TestLocalFetchFile:
    def _make_env(self):
        from tools.environments.local import LocalEnvironment

        return LocalEnvironment.__new__(LocalEnvironment)

    def test_fetch_file_copies(self, tmp_path):
        env = self._make_env()
        src = tmp_path / "src.txt"
        src.write_text("hello")
        dest = tmp_path / "dest.txt"
        env.fetch_file(str(src), str(dest))
        assert dest.read_text() == "hello"

    def test_fetch_file_same_path_is_noop(self, tmp_path):
        env = self._make_env()
        src = tmp_path / "src.txt"
        src.write_text("hello")
        env.fetch_file(str(src), str(src))
        assert src.read_text() == "hello"

    def test_fetch_file_missing_raises(self, tmp_path):
        env = self._make_env()
        with pytest.raises(FileFetchError):
            env.fetch_file(str(tmp_path / "nope.txt"), str(tmp_path / "out"))

    def test_fetch_file_size(self, tmp_path):
        env = self._make_env()
        src = tmp_path / "src.bin"
        src.write_bytes(b"12345")
        assert env.fetch_file_size(str(src)) == 5
        assert env.fetch_file_size(str(tmp_path / "nope")) is None
        assert env.fetch_file_size(str(tmp_path)) is None
