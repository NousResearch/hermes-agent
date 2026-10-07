"""Tests for the verify environment manifest and the smoke runner."""

import contextlib
import http.server
import json
import os
import subprocess
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from agent.verify.environment import (
    load_manifest,
    load_or_detect,
    manifest_path,
    save_manifest,
)
from agent.verify.recipes import Recipe
from agent.verify.runner import run_verify


class TestManifest:
    def test_roundtrip(self, tmp_path):
        recipe = Recipe(
            name="Next.js",
            kind="nextjs",
            bootstrap=["npm install"],
            build=["npm run build"],
            test=["npm test"],
            start="npm run dev",
            port=3000,
            readiness_path="/health",
        )
        path = save_manifest(tmp_path, recipe)
        assert path == manifest_path(tmp_path)
        payload = json.loads(path.read_text())
        assert payload["version"] == 1
        assert "updatedAt" in payload
        assert load_manifest(tmp_path) == recipe

    def test_missing_file(self, tmp_path):
        assert load_manifest(tmp_path) is None

    def test_malformed_json_tolerated(self, tmp_path):
        path = manifest_path(tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text("{oops", encoding="utf-8")
        assert load_manifest(tmp_path) is None

    def test_non_dict_tolerated(self, tmp_path):
        path = manifest_path(tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text("[1, 2, 3]", encoding="utf-8")
        assert load_manifest(tmp_path) is None

    def test_bare_recipe_shape_accepted(self, tmp_path):
        path = manifest_path(tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"name": "Custom", "test": ["true"]}), encoding="utf-8")
        recipe = load_manifest(tmp_path)
        assert recipe.name == "Custom"
        assert recipe.test == ["true"]

    def test_manifest_wins_over_detection(self, tmp_path):
        (tmp_path / "go.mod").write_text("module x\n", encoding="utf-8")
        save_manifest(tmp_path, Recipe(name="Custom", kind="custom", test=["true"]))
        recipe, source = load_or_detect(tmp_path)
        assert source == "manifest"
        assert recipe.name == "Custom"

    def test_detection_fallback(self, tmp_path):
        (tmp_path / "go.mod").write_text("module x\n", encoding="utf-8")
        recipe, source = load_or_detect(tmp_path)
        assert source == "detected"
        assert recipe.kind == "go"


class TestRunner:
    def test_all_phases_pass(self, tmp_path):
        recipe = Recipe(name="x", bootstrap=["true"], build=["true"], test=["true"])
        result = run_verify(tmp_path, recipe, skip_start=True)
        assert result.ok
        assert [p.phase for p in result.phases] == ["bootstrap", "build", "test"]
        assert all(p.exit_code == 0 for p in result.phases)
        assert all(p.duration >= 0 for p in result.phases)

    def test_failure_stops_pipeline(self, tmp_path):
        recipe = Recipe(name="x", build=["false"], test=["true"])
        result = run_verify(tmp_path, recipe, skip_start=True)
        assert not result.ok
        assert len(result.phases) == 1
        assert result.phases[0].exit_code == 1

    def test_output_captured(self, tmp_path):
        recipe = Recipe(name="x", test=["echo hello-verify"])
        result = run_verify(tmp_path, recipe, skip_start=True)
        assert "hello-verify" in result.phases[0].output_tail

    def test_phase_selection(self, tmp_path):
        recipe = Recipe(name="x", bootstrap=["true"], build=["true"], test=["true"])
        result = run_verify(tmp_path, recipe, phases=("test",))
        assert [p.phase for p in result.phases] == ["test"]

    def test_phase_timeout(self, tmp_path):
        recipe = Recipe(name="x", test=["sleep 5"])
        result = run_verify(tmp_path, recipe, phase_timeout=0.3, skip_start=True)
        assert not result.ok
        assert result.phases[0].timed_out
        assert result.phases[0].exit_code is None

    def test_commands_run_in_project_root(self, tmp_path):
        (tmp_path / "marker.txt").write_text("here", encoding="utf-8")
        recipe = Recipe(name="x", test=["cat marker.txt"])
        result = run_verify(tmp_path, recipe, skip_start=True)
        assert result.ok


class TestComposeGuard:
    """#103567: a compose recipe must not build/up over a live deployment."""

    def _compose_recipe(self):
        return Recipe(
            name="docker-compose project", kind="compose",
            build=["docker compose build"], start="docker compose up",
            evidence=["Detected docker-compose.yml"],
        )

    @pytest.mark.parametrize(
        "probe",
        [
            pytest.param(MagicMock(returncode=0, stdout="myproject-db-1\nmyproject-web-1\n"), id="containers-running"),
            pytest.param(subprocess.TimeoutExpired(cmd="docker", timeout=15), id="probe-timed-out"),
            pytest.param(MagicMock(returncode=1, stdout="", stderr="permission denied on docker.sock"), id="probe-failed"),
        ],
    )
    def test_refuses_and_spawns_nothing_mutating_when_live_state_cannot_be_ruled_out(
        self, tmp_path, monkeypatch, probe
    ):
        calls: list[list[str] | str] = []

        def tracking_run(cmd, *args, **kwargs):
            calls.append(cmd)
            if isinstance(probe, BaseException):
                raise probe
            return probe

        monkeypatch.setattr("subprocess.run", tracking_run)

        result = run_verify(tmp_path, self._compose_recipe(), skip_start=False)

        assert not result.ok
        assert result.phases[0].exit_code == 1
        # Only the read-only probe ran -- never build or up.
        assert len(calls) == 1
        assert calls[0][:3] == ["docker", "compose", "ps"]

    @pytest.mark.parametrize(
        "probe",
        [
            pytest.param(MagicMock(returncode=0, stdout=""), id="none-running"),
            pytest.param(FileNotFoundError("docker not found"), id="docker-absent"),
        ],
    )
    def test_proceeds_when_no_live_containers_or_docker_is_absent(self, tmp_path, monkeypatch, probe):
        monkeypatch.setattr(
            "subprocess.run",
            MagicMock(side_effect=probe) if isinstance(probe, BaseException) else MagicMock(return_value=probe),
        )

        with patch("agent.verify.runner._run_phase_command") as mock_phase:
            mock_phase.return_value = MagicMock(ok=True, phase="build")
            run_verify(tmp_path, self._compose_recipe(), phases=("build",))

        assert mock_phase.called

    def test_guard_skipped_when_no_mutating_phase_selected(self, tmp_path, monkeypatch):
        probe = MagicMock()
        monkeypatch.setattr("agent.verify.runner._compose_live_state_reason", probe)

        with patch("agent.verify.runner._run_phase_command") as mock_phase:
            mock_phase.return_value = MagicMock(ok=True, phase="test")
            run_verify(tmp_path, Recipe(name="x", kind="compose", test=["true"]), phases=("test",))

        probe.assert_not_called()

    def test_result_to_dict(self, tmp_path):
        recipe = Recipe(name="x", test=["true"])
        payload = run_verify(tmp_path, recipe, skip_start=True).to_dict()
        assert payload["ok"] is True
        assert payload["recipe"] == "x"
        assert payload["phases"][0]["command"] == "true"
        assert payload["readiness"] is None


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestReadiness:
    @pytest.mark.platforms("linux")
    def test_readiness_against_live_server(self, tmp_path):
        port = _free_port()
        recipe = Recipe(
            name="x",
            start=f"python3 -m http.server {port} --bind 127.0.0.1",
            port=port,
        )
        result = run_verify(tmp_path, recipe, phases=("start",), ready_timeout=15)
        assert result.readiness is not None
        assert result.readiness.ready
        assert result.readiness.status_code == 200
        assert result.readiness.url == f"http://127.0.0.1:{port}/"
        assert result.ok

    def test_readiness_timeout_when_nothing_listens(self, tmp_path):
        port = _free_port()
        recipe = Recipe(name="x", start="sleep 30", port=port)
        result = run_verify(tmp_path, recipe, phases=("start",), ready_timeout=1.5)
        assert result.readiness is not None
        assert not result.readiness.ready
        assert not result.ok

    def test_skip_start(self, tmp_path):
        recipe = Recipe(name="x", test=["true"], start="sleep 30", port=1)
        result = run_verify(tmp_path, recipe, skip_start=True)
        assert result.readiness is None
        assert result.ok

    def test_start_skipped_after_phase_failure(self, tmp_path):
        recipe = Recipe(name="x", test=["false"], start="sleep 30", port=1)
        result = run_verify(tmp_path, recipe, stop_on_failure=False)
        assert result.readiness is None
        assert not result.ok

    def test_port_override(self, tmp_path):
        port = _free_port()

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(204)
                self.end_headers()

            def log_message(self, *a):
                pass

        server = http.server.HTTPServer(("127.0.0.1", port), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        time.sleep(0.05)
        try:
            recipe = Recipe(name="x", start="sleep 30", port=1)
            result = run_verify(
                tmp_path, recipe, phases=("start",), ready_timeout=10, port_override=port
            )
            assert result.readiness.ready
            assert result.readiness.status_code == 204
        finally:
            server.shutdown()
            thread.join(timeout=5)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group semantics")
class TestTerminateProcessGroupGuard:
    """_terminate_process_group must never killpg a group the child does not
    lead: a child spawned without start_new_session shares OUR process group,
    and the group signal would take the verify runner down with it."""

    def test_real_shared_group_child_does_not_signal_us(self):
        """Real child in our own group: if killpg fired, this test process would
        be dead before the assertion. The direct child still dies."""
        import os as _os

        from agent.verify.runner import _terminate_process_group

        proc = subprocess.Popen(
            ["sleep", "60"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            assert _os.getpgid(proc.pid) == _os.getpgid(0)  # shared group precondition
            _terminate_process_group(proc)
            proc.wait(timeout=5)
            assert proc.returncode is not None
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_real_group_leader_child_is_group_killed(self):
        """Control: a start_new_session child leads its own group and is killed
        through the group signal path."""
        import os as _os

        from agent.verify.runner import _terminate_process_group

        proc = subprocess.Popen(
            ["sleep", "60"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            process_group=0,
        )
        try:
            assert _os.getpgid(proc.pid) == proc.pid  # leader precondition
            _terminate_process_group(proc)
            proc.wait(timeout=5)
            assert proc.returncode is not None
        finally:
            if proc.poll() is None:
                proc.kill()


@pytest.mark.platforms("windows")
class TestStartPhaseShellGrandchild:
    """``shell=True`` interposes ``cmd.exe``, so the app the recipe starts is a GRANDCHILD.
    ``start_new_session=True`` is a POSIX-only no-op and Windows has no ``os.killpg``, so the
    runner used to stop that ``cmd.exe`` alone. The app survived: it kept the readiness port
    bound, so the next ``hermes verify`` could not bind it, and it held the stdout write end it
    inherited, so reading the pipe to EOF never returned -- verify hung forever and printed
    nothing, not even the JSON report (#134525)."""

    @staticmethod
    def _free_port() -> int:
        import socket
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            return probe.getsockname()[1]

    def test_grandchild_is_stopped_and_the_phase_returns(self, tmp_path):
        import sys

        import psutil

        from agent.verify.runner import _run_start_phase

        port, pid_file = self._free_port(), tmp_path / "server.pid"
        server = tmp_path / "server.py"
        server.write_text(
            "import os, sys\n"
            "from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer\n"
            "class H(BaseHTTPRequestHandler):\n"
            "    def do_GET(self):\n"
            "        self.send_response(200); self.end_headers(); self.wfile.write(b'ok')\n"
            "    def log_message(self, *a): pass\n"
            f"open({str(pid_file)!r}, 'w', encoding='utf-8').write(str(os.getpid()))\n"
            "print('SERVER-UP', flush=True)\n"
            "ThreadingHTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()\n",
            encoding="utf-8",
        )
        recipe = Recipe(name="grandchild app", kind="test", port=port, readiness_path="/health",
                        start=f'"{sys.executable}" "{server}" {port}')

        result: list = []
        worker = threading.Thread(
            target=lambda: result.append(_run_start_phase(recipe, tmp_path, 30.0)), daemon=True)
        worker.start()
        # Bounded join: unfixed, the phase never returns at all, and a hanging test tells nobody why.
        worker.join(timeout=90)

        server_pid = int(pid_file.read_text(encoding="utf-8")) if pid_file.is_file() else None
        try:
            assert result, "the start phase never returned (the grandchild still held the stdout pipe)"
            assert server_pid is not None, "the recipe's server never started"
            assert result[0].ready is True, result[0]
            assert not psutil.pid_exists(server_pid) or not psutil.Process(server_pid).is_running(), (
                "the app survived teardown and still holds the readiness port")
            # The pipe reached EOF, so the phase reports what the app actually printed.
            assert "SERVER-UP" in result[0].output_tail
        finally:
            if server_pid is not None:  # never leak the server into the rest of the run
                with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                    psutil.Process(server_pid).kill()
