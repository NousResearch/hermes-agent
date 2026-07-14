import json
from pathlib import Path
from types import SimpleNamespace

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import codex_exec_tool as tool


def _fake_completed(stdout="", stderr="", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def test_resolve_cwd_requires_allowed_root(tmp_path):
    allowed = tmp_path / "allowed"
    blocked = tmp_path / "blocked"
    allowed.mkdir()
    blocked.mkdir()

    assert tool._resolve_cwd(str(allowed), [allowed]) == allowed.resolve()

    try:
        tool._resolve_cwd(str(blocked), [allowed])
    except PermissionError as exc:
        assert "outside codex_exec.allowed_roots" in str(exc)
    else:
        raise AssertionError("expected PermissionError")


def test_scrub_env_drops_secret_like_values():
    clean = tool._scrub_env(
        {
            "PATH": "/bin",
            "HOME": "/tmp/home",
            "OPENAI_API_KEY": "sk-secret",
            "RANDOM_TOKEN": "secret",
            "CODEX_HOME": "/tmp/codex",
            "GIT_CONFIG_GLOBAL": "/tmp/gitconfig",
        }
    )

    assert clean["PATH"] == "/bin"
    assert clean["CODEX_HOME"] == "/tmp/codex"
    assert clean["GIT_CONFIG_GLOBAL"] == "/tmp/gitconfig"
    assert "OPENAI_API_KEY" not in clean
    assert "RANDOM_TOKEN" not in clean


def test_codex_exec_blocks_danger_without_opt_in(tmp_path, monkeypatch):
    monkeypatch.setattr(tool.shutil, "which", lambda name: "/usr/bin/codex")
    monkeypatch.setattr(tool, "_codex_auth_available", lambda env=None: True)
    monkeypatch.setattr(tool, "_configured_allowed_roots", lambda config=None: [tmp_path])

    result = json.loads(
        tool.codex_exec_tool(
            task="diagnose",
            cwd=str(tmp_path),
            sandbox="danger-full-access",
        )
    )

    assert "error" in result
    assert "allow_danger" in result["error"]


def test_codex_exec_writes_artifacts_and_summary(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    hermes_home = tmp_path / "hermes-home"
    token = set_hermes_home_override(hermes_home)
    calls = []

    stdout = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "command_execution", "command": "ls"},
                }
            ),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "done"},
                }
            ),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                }
            ),
        ]
    )

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if cmd[:2] == ["git", "-C"]:
            return _fake_completed(stdout="")
        return _fake_completed(stdout=stdout, stderr="progress", returncode=0)

    try:
        monkeypatch.setattr(tool.shutil, "which", lambda name: "/usr/bin/codex")
        monkeypatch.setattr(tool, "_codex_auth_available", lambda env=None: True)
        monkeypatch.setattr(tool, "_configured_allowed_roots", lambda config=None: [repo])
        monkeypatch.setattr(tool.subprocess, "run", fake_run)

        result = json.loads(
            tool.codex_exec_tool(
                task="summarize",
                cwd=str(repo),
                sandbox="read-only",
                timeout_seconds=30,
            )
        )
    finally:
        reset_hermes_home_override(token)

    assert result["success"] is True
    assert result["thread_id"] == "thread-1"
    assert result["usage"] == {"input_tokens": 10, "output_tokens": 2}
    assert result["command_count"] == 1
    assert result["final_message"] == "done"

    artifact_dir = Path(result["artifact_dir"])
    assert (artifact_dir / "stdout.jsonl").read_text() == stdout
    assert (artifact_dir / "stderr.txt").read_text() == "progress"
    metadata = json.loads((artifact_dir / "metadata.json").read_text())
    assert metadata["cwd"] == str(repo.resolve())
    assert metadata["sandbox"] == "read-only"

    codex_cmd = [call for call, _kw in calls if call and call[0] == "codex"][0]
    codex_kwargs = [_kw for call, _kw in calls if call and call[0] == "codex"][0]
    assert "--json" in codex_cmd
    assert "--ephemeral" in codex_cmd
    assert codex_kwargs["stdin"] is tool.subprocess.DEVNULL
    sandbox_idx = codex_cmd.index("--sandbox")
    assert codex_cmd[sandbox_idx:sandbox_idx + 2] == ["--sandbox", "read-only"]


def test_codex_exec_requires_output_schema_inside_cwd(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    other = tmp_path / "other"
    repo.mkdir()
    other.mkdir()
    schema = other / "schema.json"
    schema.write_text("{}")

    monkeypatch.setattr(tool.shutil, "which", lambda name: "/usr/bin/codex")
    monkeypatch.setattr(tool, "_codex_auth_available", lambda env=None: True)
    monkeypatch.setattr(tool, "_configured_allowed_roots", lambda config=None: [repo])

    result = json.loads(
        tool.codex_exec_tool(
            task="summarize",
            cwd=str(repo),
            output_schema=str(schema),
        )
    )

    assert "error" in result
    assert "output_schema must resolve inside cwd" in result["error"]


def test_codex_exec_fails_closed_without_codex_auth(tmp_path, monkeypatch):
    monkeypatch.setattr(tool.shutil, "which", lambda name: "/usr/bin/codex")
    monkeypatch.setattr(tool, "_codex_auth_available", lambda env=None: False)

    result = json.loads(tool.codex_exec_tool(task="diagnose", cwd=str(tmp_path)))

    assert "error" in result
    assert "does not fall back" in result["error"]
