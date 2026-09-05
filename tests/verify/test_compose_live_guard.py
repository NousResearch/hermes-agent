"""The verify runner must never rebuild or recreate a live docker compose stack.

Every docker call is monkeypatched: these tests never talk to a real daemon.
"""

import subprocess

import pytest

from agent.verify import runner
from agent.verify.recipes import Recipe
from agent.verify.runner import ReadinessResult, _is_mutating_compose_command, run_verify

COMPOSE_RECIPE = Recipe(
    name="docker-compose project",
    kind="compose",
    build=["docker compose build"],
    start="docker compose up",
)


class _Recorder:
    """Stands in for ``subprocess.run``: list argv = the compose probe, str = a phase command."""

    def __init__(self, probe_stdout: str = "", probe_returncode: int = 0, probe_error=None):
        self.probe_stdout = probe_stdout
        self.probe_returncode = probe_returncode
        self.probe_error = probe_error
        self.probes: list[list[str]] = []
        self.commands: list[str] = []

    def __call__(self, command, **kwargs):
        if isinstance(command, list):
            self.probes.append(command)
            if self.probe_error is not None:
                raise self.probe_error
            return subprocess.CompletedProcess(
                command, self.probe_returncode, stdout=self.probe_stdout, stderr=""
            )
        self.commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="ran\n", stderr=None)


@pytest.fixture
def recorder(monkeypatch):
    def install(**kwargs):
        rec = _Recorder(**kwargs)
        monkeypatch.setattr(runner.subprocess, "run", rec)
        monkeypatch.setattr(
            runner.subprocess, "Popen",
            lambda *a, **kw: pytest.fail(f"start command was launched: {a!r}"),
        )
        return rec

    return install


class TestMutatingCommandDetection:
    @pytest.mark.parametrize(
        "command",
        [
            "docker compose build",
            "docker compose up",
            "docker-compose up -d",
            "docker compose -f compose.yml up --force-recreate",
            "docker compose down",
            "/usr/local/bin/docker-compose build web",
        ],
    )
    def test_mutating(self, command):
        assert _is_mutating_compose_command(command)

    @pytest.mark.parametrize(
        "command",
        [
            "docker compose ps",
            "docker compose config",
            "npm run build",
            "make up-to-date",
            "docker build .",
        ],
    )
    def test_not_mutating(self, command):
        assert not _is_mutating_compose_command(command)


class TestLiveComposeGuard:
    def test_running_containers_block_build_and_start(self, tmp_path, recorder):
        rec = recorder(probe_stdout="abc123def456\n789ghi012jkl\n")
        result = run_verify(tmp_path, COMPOSE_RECIPE)

        assert rec.commands == []
        assert rec.probes == [["docker", "compose", "ps", "-q"]]
        assert not result.ok
        assert result.readiness is None
        refused = result.phases[0]
        assert refused.phase == "build"
        assert refused.command == "docker compose build"
        assert refused.exit_code == 1
        assert "left untouched" in refused.output_tail
        assert "abc123def456" in refused.output_tail

    def test_refusal_reported_for_start_phase_alone(self, tmp_path, recorder):
        rec = recorder(probe_stdout="abc123def456\n")
        result = run_verify(tmp_path, COMPOSE_RECIPE, phases=("start",))

        assert rec.commands == []
        assert not result.ok
        assert [p.phase for p in result.phases] == ["start"]
        assert result.phases[0].command == "docker compose up"
        assert "left untouched" in result.phases[0].output_tail

    def test_refusal_streamed_to_output_callback(self, tmp_path, recorder):
        recorder(probe_stdout="abc123def456\n")
        chunks: list[str] = []
        run_verify(tmp_path, COMPOSE_RECIPE, on_output=chunks.append)
        assert any("Refused to run" in chunk for chunk in chunks)

    def test_probe_failure_fails_closed(self, tmp_path, recorder):
        rec = recorder(probe_returncode=1)
        result = run_verify(tmp_path, COMPOSE_RECIPE)
        assert rec.commands == []
        assert not result.ok
        assert "docker compose ps -q` failed" in result.phases[0].output_tail

    def test_missing_docker_fails_closed(self, tmp_path, recorder):
        rec = recorder(probe_error=FileNotFoundError("docker"))
        result = run_verify(tmp_path, COMPOSE_RECIPE)
        assert rec.commands == []
        assert not result.ok
        assert "could not probe" in result.phases[0].output_tail

    def test_no_running_containers_runs_build_and_start(self, tmp_path, monkeypatch):
        rec = _Recorder(probe_stdout="\n")
        monkeypatch.setattr(runner.subprocess, "run", rec)
        started: list[str] = []

        def fake_start(recipe, root, ready_timeout, port_override=None):
            started.append(recipe.start)
            return ReadinessResult("http://127.0.0.1:8000/", True, 200, 0.0)

        monkeypatch.setattr(runner, "_run_start_phase", fake_start)
        result = run_verify(tmp_path, COMPOSE_RECIPE)

        assert rec.commands == ["docker compose build"]
        assert started == ["docker compose up"]
        assert result.ok

    def test_probe_runs_once_per_verify(self, tmp_path, recorder):
        rec = recorder(probe_stdout="abc123def456\n")
        recipe = Recipe(
            name="compose", kind="compose",
            build=["docker compose build"], test=["docker compose up -d"],
        )
        run_verify(tmp_path, recipe, skip_start=True, stop_on_failure=False)
        assert rec.commands == []
        assert len(rec.probes) == 1

    def test_non_compose_recipe_is_unaffected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            runner, "_compose_project_is_live",
            lambda root: pytest.fail("probed docker for a non-compose recipe"),
        )
        recipe = Recipe(name="x", build=["true"], test=["echo hello-verify"])
        result = run_verify(tmp_path, recipe, skip_start=True)
        assert result.ok
        assert "hello-verify" in result.phases[1].output_tail
