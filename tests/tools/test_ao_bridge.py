import os
import signal
import subprocess

from tools.ao_bridge import AOBridge, AOSession


def test_bridge_env_prepends_codex_shim(tmp_path, monkeypatch):
    home = tmp_path / "home"
    shim_dir = tmp_path / "shims"
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    bridge = AOBridge(
        home=str(home),
        codex_shim_dir=shim_dir,
        codex_real_bin="/opt/test/bin/codex",
    )

    env = bridge._bridge_env()

    assert env["PATH"].split(os.pathsep)[:2] == [str(home / "bin"), str(shim_dir)]
    assert env["CODEX_REAL_BIN"] == "/opt/test/bin/codex"
    assert env["HOME"] == str(home)
    assert env["HERMES_AO_VERIFICATION_CODEX_HOME"] == str(home / ".codex-verification")
    assert env["HERMES_AO_CODEX_AUTH_HOME"].endswith("/.codex")


def test_bridge_env_allows_verification_codex_home_override(tmp_path, monkeypatch):
    override = tmp_path / "codex-verify"
    monkeypatch.setenv("HERMES_AO_VERIFICATION_CODEX_HOME", str(override))
    auth_home = tmp_path / "codex-auth"
    monkeypatch.setenv("HERMES_AO_CODEX_AUTH_HOME", str(auth_home))

    bridge = AOBridge(codex_real_bin="/opt/test/bin/codex")

    env = bridge._bridge_env()
    assert env["HERMES_AO_VERIFICATION_CODEX_HOME"] == str(override)
    assert env["HERMES_AO_CODEX_AUTH_HOME"] == str(auth_home)


def test_bridge_env_does_not_use_shim_as_real_codex(tmp_path, monkeypatch):
    home = tmp_path / "home"
    user_bin = home / "bin"
    shim_dir = tmp_path / "shims"
    user_bin.mkdir(parents=True)
    shim_dir.mkdir()
    shim = shim_dir / "codex"
    shim.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    user_codex = user_bin / "codex"
    user_codex.symlink_to(shim)
    monkeypatch.setenv("PATH", f"{user_bin}{os.pathsep}/usr/bin")
    monkeypatch.delenv("CODEX_REAL_BIN", raising=False)

    bridge = AOBridge(home=str(home), codex_shim_dir=shim_dir)

    assert bridge._bridge_env()["CODEX_REAL_BIN"] == "/opt/homebrew/bin/codex"


def test_ensure_codex_shim_on_user_path_is_non_destructive(tmp_path):
    home = tmp_path / "home"
    shim_dir = tmp_path / "shims"
    shim_dir.mkdir()
    shim = shim_dir / "codex"
    shim.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    bridge = AOBridge(
        home=str(home),
        codex_shim_dir=shim_dir,
        codex_real_bin="/opt/test/bin/codex",
    )
    bridge._ensure_codex_shim_on_user_path()

    user_codex = home / "bin" / "codex"
    assert user_codex.is_symlink()
    assert user_codex.resolve() == shim

    user_codex.unlink()
    user_codex.write_text("existing", encoding="utf-8")
    bridge._ensure_codex_shim_on_user_path()

    assert user_codex.read_text(encoding="utf-8") == "existing"


def test_spawn_forwards_minimal_worker_prompt_to_node_bridge():
    bridge = AOBridge(codex_real_bin="/opt/test/bin/codex")
    calls = []
    bridge._call = lambda command, payload, timeout: calls.append((command, payload, timeout)) or {
        "session": {"id": "oryn-workspace-42", "project_id": "OrynWorkspace", "status": "working"}
    }

    session = bridge.spawn(
        project_id="OrynWorkspace",
        prompt="Read-only benchmark.",
        minimal_worker_prompt=True,
    )

    assert session.id == "oryn-workspace-42"
    assert calls == [(
        "spawn",
        {
            "project_id": "OrynWorkspace",
            "prompt": "Read-only benchmark.",
            "issue_id": None,
            "branch": None,
            "agent": "codex",
            "minimal_worker_prompt": True,
        },
        180,
    )]


def test_run_codex_exec_benchmark_uses_noninteractive_cli(tmp_path, monkeypatch):
    config = tmp_path / "ao.yaml"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config.write_text(
        f"""
defaults:
  agent: codex
projects:
  OrynWorkspace:
    path: {workspace}
    agentConfig:
      model: gpt-test
      reasoningEffort: medium
""",
        encoding="utf-8",
    )
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        output_file = args[args.index("--output-last-message") + 1]
        with open(output_file, "w", encoding="utf-8") as handle:
            handle.write("BENCHMARK_RESULT\nFINAL_MARKER: BENCH_DONE\n")
        return subprocess.CompletedProcess(
            args,
            0,
            "BENCHMARK_RESULT\nFINAL_MARKER: BENCH_DONE\n",
            "tokens used\n1,234\n",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    bridge = AOBridge(config_path=str(config), codex_real_bin="/opt/test/bin/codex")

    result = bridge.run_codex_exec_benchmark(
        project_id="OrynWorkspace",
        prompt="Run benchmark.",
        timeout_seconds=12,
    )

    args, kwargs = calls[0]
    assert args[:2] == ["/opt/test/bin/codex", "exec"]
    assert "--sandbox" in args and "read-only" in args
    assert "--cd" in args and str(workspace) in args
    assert "--model" in args and "gpt-test" in args
    assert args[-1] == "-"
    assert kwargs["input"] == "Run benchmark."
    assert result["status"] == "completed"
    assert result["summary"] == "BENCHMARK_RESULT\nFINAL_MARKER: BENCH_DONE"
    assert result["token_total"] == 1234
    assert result["workspace_path"] == str(workspace)


def test_kill_forces_tmux_and_session_process_cleanup(monkeypatch):
    bridge = AOBridge(codex_real_bin="/opt/test/bin/codex")
    calls = []
    signals = []

    bridge._call = lambda command, payload, timeout: calls.append((command, payload, timeout)) or {"ok": True}

    def fake_run(args, **kwargs):
        if args[:2] == ["tmux", "kill-session"]:
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[:2] == ["pgrep", "-f"]:
            pattern = args[-1]
            output = "123\n" if pattern in {"oryn-workspace-42", "tmux-oryn-workspace-42"} else ""
            return subprocess.CompletedProcess(args, 0 if output else 1, output, "")
        raise AssertionError(f"unexpected subprocess: {args}")

    def fake_kill(pid, sig):
        signals.append((pid, sig))
        if sig == 0:
            raise ProcessLookupError()

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(os, "kill", fake_kill)

    bridge.kill(
        "oryn-workspace-42",
        session=AOSession(id="oryn-workspace-42", tmux_name="tmux-oryn-workspace-42"),
    )

    assert calls == [("kill", {"session_id": "oryn-workspace-42"}, 60)]
    assert (123, signal.SIGTERM) in signals


def test_runtime_health_marks_running_session_stale(monkeypatch):
    bridge = AOBridge(codex_real_bin="/opt/test/bin/codex")

    def fake_run(args, **kwargs):
        if args[:2] == ["tmux", "has-session"]:
            return subprocess.CompletedProcess(args, 1, "", "missing")
        if args[:2] == ["pgrep", "-f"]:
            return subprocess.CompletedProcess(args, 1, "", "")
        raise AssertionError(f"unexpected subprocess: {args}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    health = bridge.runtime_health(
        AOSession(
            id="oryn-workspace-42",
            status="working",
            tmux_name="tmux-oryn-workspace-42",
        )
    )

    assert health["runtime_health"] == "stale"
    assert "runtime is gone" in health["runtime_warning"]


def test_runtime_health_keeps_terminal_session_ok_when_tmux_is_gone(monkeypatch):
    bridge = AOBridge(codex_real_bin="/opt/test/bin/codex")

    def fake_run(args, **kwargs):
        if args[:2] == ["tmux", "has-session"]:
            return subprocess.CompletedProcess(args, 1, "", "missing")
        if args[:2] == ["pgrep", "-f"]:
            return subprocess.CompletedProcess(args, 1, "", "")
        raise AssertionError(f"unexpected subprocess: {args}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    health = bridge.runtime_health(
        AOSession(
            id="oryn-workspace-42",
            status="done",
            tmux_name="tmux-oryn-workspace-42",
        )
    )

    assert health["runtime_health"] == "ok"
    assert health["runtime_warning"] is None


def test_archived_sessions_fill_agent_and_model_from_project_config(tmp_path):
    home = tmp_path / "home"
    config = tmp_path / "agent-orchestrator.yaml"
    config.write_text(
        """
defaults:
  agent: codex
projects:
  OrynWorkspace:
    agentConfig:
      model: gpt-5.5
      reasoningEffort: medium
""",
        encoding="utf-8",
    )
    archive = home / ".agent-orchestrator" / "hash-oryn-workspace" / "sessions" / "archive"
    archive.mkdir(parents=True)
    (archive / "oryn-workspace-12_2026-05-27T11-02-59-567Z").write_text(
        "\n".join([
            "worktree=/tmp/oryn-workspace-12",
            "branch=session/oryn-workspace-12",
            "status=spawning",
            "tmuxName=hash-oryn-workspace-12",
            "project=OrynWorkspace",
        ]),
        encoding="utf-8",
    )
    bridge = AOBridge(
        config_path=str(config),
        home=str(home),
        codex_real_bin="/opt/test/bin/codex",
    )
    bridge._call = lambda command, payload, timeout: {"sessions": []}

    sessions = bridge.list(project_id="OrynWorkspace")

    assert sessions[0].agent == "codex"
    assert sessions[0].model == "gpt-5.5"
    assert sessions[0].reasoning_effort == "medium"
    assert sessions[0].event_fields()["agent"] == "codex"
    assert sessions[0].event_fields()["model"] == "gpt-5.5"
    assert sessions[0].event_fields()["reasoning_effort"] == "medium"


def test_spawn_persists_resolved_launch_metadata_from_project_config(tmp_path):
    config = tmp_path / "agent-orchestrator.yaml"
    config.write_text(
        """
defaults:
  agent: codex
projects:
  OrynWorkspace:
    agentConfig:
      model: gpt-5.5
      reasoningEffort: high
""",
        encoding="utf-8",
    )
    bridge = AOBridge(
        config_path=str(config),
        codex_real_bin="/opt/test/bin/codex",
    )
    calls = []

    def fake_call(command, payload, timeout):
        calls.append((command, payload, timeout))
        return {
            "session": {
                "id": "oryn-workspace-15",
                "project_id": "OrynWorkspace",
                "status": "spawning",
            }
        }

    bridge._call = fake_call

    session = bridge.spawn(project_id="OrynWorkspace", prompt="Smoke")

    assert calls[0][1]["agent"] == "codex"
    assert session.agent == "codex"
    assert session.model == "gpt-5.5"
    assert session.reasoning_effort == "high"


def test_codex_shim_translates_ao_approval_mode(tmp_path):
    real_codex = tmp_path / "real-codex"
    real_codex.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    real_codex.chmod(0o755)

    shim = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "tools",
        "ao_shims",
        "codex",
    )
    proc = subprocess.run(
        [
            shim,
            "--approval-mode",
            "full-auto",
            "--model",
            "gpt-5.5",
            "--",
            "hello",
        ],
        text=True,
        capture_output=True,
        env={"CODEX_REAL_BIN": str(real_codex), "PATH": os.environ.get("PATH", "")},
        check=True,
    )

    assert proc.stdout.splitlines() == [
        "--ask-for-approval",
        "never",
        "--sandbox",
        "danger-full-access",
        "--model",
        "gpt-5.5",
        "--",
        "hello",
    ]


def test_codex_shim_isolates_dev_verification_codex_home(tmp_path):
    real_codex = tmp_path / "real-codex"
    real_codex.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'CODEX_HOME=%s\\n' \"${CODEX_HOME:-}\"\n"
        "printf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    real_codex.chmod(0o755)
    verification_home = tmp_path / "codex-verification"
    auth_home = tmp_path / "codex-auth"
    auth_home.mkdir()
    (auth_home / "auth.json").write_text('{"token":"test"}\n', encoding="utf-8")
    workspace = tmp_path / "worktree"
    workspace.mkdir()

    shim = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "tools",
        "ao_shims",
        "codex",
    )
    proc = subprocess.run(
        [
            shim,
            "--model",
            "gpt-5.5",
            "--",
            "Run fixed verification commands and return DEV_VERIFICATION_RESULTS.",
        ],
        text=True,
        capture_output=True,
        cwd=workspace,
        env={
            "CODEX_REAL_BIN": str(real_codex),
            "HERMES_AO_VERIFICATION_CODEX_HOME": str(verification_home),
            "HERMES_AO_CODEX_AUTH_HOME": str(auth_home),
            "HOME": str(tmp_path / "home"),
            "PATH": os.environ.get("PATH", ""),
        },
        check=True,
    )

    assert proc.stdout.splitlines()[0] == f"CODEX_HOME={verification_home}"
    config = verification_home / "config.toml"
    assert config.exists()
    assert f'[projects."{workspace}"]' in config.read_text(encoding="utf-8")
    assert (verification_home / "auth.json").read_text(encoding="utf-8") == '{"token":"test"}\n'


def test_codex_shim_copies_auth_for_lab_worker_home(tmp_path):
    real_codex = tmp_path / "real-codex"
    real_codex.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'CODEX_HOME=%s\\n' \"${CODEX_HOME:-}\"\n"
        "printf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    real_codex.chmod(0o755)
    lab_home = tmp_path / ".oryn-lab"
    workspace = lab_home / ".worktrees" / "HermesAgentLab" / "lab-hermes-agent-1"
    workspace.mkdir(parents=True)
    auth_home = tmp_path / "codex-auth"
    auth_home.mkdir()
    (auth_home / "auth.json").write_text('{"token":"lab-test"}\n', encoding="utf-8")

    shim = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "tools",
        "ao_shims",
        "codex",
    )
    subprocess.run(
        [
            shim,
            "--model",
            "gpt-5.5",
            "--",
            "Append a docs note.",
        ],
        text=True,
        capture_output=True,
        cwd=workspace,
        env={
            "CODEX_REAL_BIN": str(real_codex),
            "HERMES_AO_CODEX_AUTH_HOME": str(auth_home),
            "HOME": str(lab_home),
            "PATH": os.environ.get("PATH", ""),
        },
        check=True,
    )

    codex_home = lab_home / ".codex"
    assert (codex_home / "auth.json").read_text(encoding="utf-8") == '{"token":"lab-test"}\n'
    assert f'[projects."{workspace}"]' in (codex_home / "config.toml").read_text(encoding="utf-8")
