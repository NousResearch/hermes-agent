"""The standalone DM runner must activate PM dependencies after env scrubbing (#122487)."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


@pytest.mark.parametrize("stdin_file", [False, True])
def test_delivery_runner_uses_committed_dependencies(tmp_path, monkeypatch, stdin_file):
    from pm.environments import runtime_facts_path, site_packages
    from tools import bot_mode_dm

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    root = Path(bot_mode_dm.__file__).resolve().parents[1]
    record = runtime_facts_path(root)
    selected = record.parent / "environments" / "selected" / "venv"
    packages = site_packages(selected)
    packages.mkdir(parents=True)
    (selected / "pyvenv.cfg").write_text("home = fixture\n", encoding="utf-8")
    # Reuse the test interpreter's installed dependencies without installing anything.
    # -S below removes them from the runner; only the committed generation restores them.
    installed = [p for p in sys.path if Path(p).name in ("site-packages", "dist-packages")]
    assert installed
    (packages / "test-dependencies.pth").write_text("\n".join(installed) + "\n", encoding="utf-8")
    record.write_text(json.dumps({"schema": 1, "packages": {"venv": {"environment": str(selected)}}}), encoding="utf-8")
    payload = tmp_path / "message.txt"
    payload.write_text("hello from fixture", encoding="utf-8")
    observed = tmp_path / "received.json"
    transport = tmp_path / "transport.py"
    transport.write_text(
        "import json, os, pathlib, sys\n"
        "text = sys.stdin.read() if sys.argv[1] == '-' else pathlib.Path(sys.argv[1]).read_text(encoding='utf-8')\n"
        "pathlib.Path(sys.argv[2]).write_text(json.dumps({'text': text, 'author': json.loads(os.environ['HERMES_TURN_AUTHOR'])}), encoding='utf-8')\n",
        encoding="utf-8",
    )
    author = {"id": "bot:sender", "name": "sender", "is_bot": True}
    command = bot_mode_dm._delivery_command(
        [sys.executable, str(transport), "-" if stdin_file else str(payload), str(observed)],
        str(payload), stdin_file=stdin_file, author=author,
    )
    argv = shlex.split(command)
    argv.insert(1, "-S")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("VIRTUAL_ENV", None)
    result = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(observed.read_text(encoding="utf-8")) == {"text": "hello from fixture", "author": author}
    assert not payload.exists()


@pytest.mark.parametrize("mode", ["--run-delivery", "--wait-reply", "--invalid"])
def test_unavailable_generation_preserves_payload_and_stdlib_paths(tmp_path, monkeypatch, mode):
    from pm.environments import runtime_facts_path
    from tools import bot_mode_dm

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    record = runtime_facts_path(Path(bot_mode_dm.__file__).resolve().parents[1])
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"packages": {"venv": {"environment": str(tmp_path / "missing")}}}), encoding="utf-8")
    payload = tmp_path / "message.txt"
    payload.write_text("do not consume", encoding="utf-8")
    reply = tmp_path / "reply.json"
    reply.write_text(json.dumps({"reply": "already delivered"}), encoding="utf-8")
    args = {
        "--run-delivery": [mode, "stdin", str(payload), "must-not-launch"],
        "--wait-reply": [mode, str(reply), "recipient", "1"],
        "--invalid": [mode],
    }[mode]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-S", bot_mode_dm.__file__, *args], cwd=tmp_path,
                            env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == {"--run-delivery": 1, "--wait-reply": 0, "--invalid": 2}[mode], result.stderr
    if mode == "--wait-reply":
        assert "already delivered" in result.stdout
    elif mode == "--run-delivery":
        assert "dependency environment" in result.stderr
    assert payload.read_text(encoding="utf-8") == "do not consume"
