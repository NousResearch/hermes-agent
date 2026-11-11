"""Tests for the ``hermes verify`` CLI command implementation."""

import argparse
import json

from hermes_cli.verify_cmd import run_verify_command


def make_args(path, **overrides):
    defaults = dict(
        path=str(path),
        detect_only=False,
        save=False,
        skip_start=False,
        phase=None,
        port=None,
        timeout=60.0,
        ready_timeout=5.0,
        json=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_detect_only_json(tmp_path, capsys):
    (tmp_path / "go.mod").write_text("module x\n", encoding="utf-8")
    code = run_verify_command(make_args(tmp_path, detect_only=True, json=True))
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "detected"
    assert payload["recipe"]["kind"] == "go"
    assert payload["recipe"]["build"] == ["go build ./..."]


def test_no_recipe_found(tmp_path, capsys):
    code = run_verify_command(make_args(tmp_path, detect_only=True))
    assert code == 1
    assert "No recognizable project" in capsys.readouterr().err


def test_save_writes_manifest(tmp_path):
    (tmp_path / "go.mod").write_text("module x\n", encoding="utf-8")
    code = run_verify_command(make_args(tmp_path, detect_only=True, save=True, json=True))
    assert code == 0
    manifest = tmp_path / ".hermes" / "environment.json"
    assert manifest.exists()
    payload = json.loads(manifest.read_text())
    assert payload["version"] == 1
    assert payload["recipe"]["kind"] == "go"


def test_run_phases_json(tmp_path, capsys):
    manifest = tmp_path / ".hermes"
    manifest.mkdir()
    (manifest / "environment.json").write_text(
        json.dumps({"recipe": {"name": "Fake", "test": ["echo ok"]}}),
        encoding="utf-8",
    )
    code = run_verify_command(make_args(tmp_path, json=True, skip_start=True))
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["source"] == "manifest"
    assert payload["phases"][0]["command"] == "echo ok"


def test_failing_phase_exit_code(tmp_path, capsys):
    manifest = tmp_path / ".hermes"
    manifest.mkdir()
    (manifest / "environment.json").write_text(
        json.dumps({"recipe": {"name": "Fake", "test": ["false"]}}),
        encoding="utf-8",
    )
    code = run_verify_command(make_args(tmp_path, json=True))
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False




def test_bad_path(tmp_path, capsys):
    code = run_verify_command(make_args(tmp_path / "nope"))
    assert code == 2


def _toy_manifest(tmp_path, command="echo ok"):
    manifest = tmp_path / ".hermes"
    manifest.mkdir(exist_ok=True)
    (manifest / "environment.json").write_text(
        json.dumps({"recipe": {"name": "Toy", "test": [command]}}),
        encoding="utf-8",
    )


def _fake_credential_sources(monkeypatch, names, values=None):
    """Pretend the CLI loaded *names* into os.environ; return only those really present."""
    import hermes_cli.env_loader as env_loader

    for name, value in (values or {}).items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(env_loader, "launch_dotenv_keys", lambda: frozenset(names))
    monkeypatch.setattr(env_loader, "secret_source_names", lambda: ())


def test_child_env_names_are_filtered_ordered_and_value_free(monkeypatch):
    from hermes_cli.verify_cmd import _child_env_block

    _fake_credential_sources(
        monkeypatch,
        ["zz_TOKEN", "AA_API_KEY", "HERMES_SESSION_ID", "NOT_PRESENT_SECRET"],
        {"zz_TOKEN": "value-one", "AA_API_KEY": "value-two"},
    )

    block = _child_env_block()

    # Credential-ish suffixes only, names actually present in the child env, sorted.
    assert block["credentialCount"] == 2
    assert block["credentialNames"] == ["AA_API_KEY", "zz_TOKEN"]
    serialised = json.dumps(block)
    assert "value-one" not in serialised
    assert "value-two" not in serialised


def test_child_env_block_is_absent_when_the_sources_break(monkeypatch):
    import hermes_cli.env_loader as env_loader
    from hermes_cli.verify_cmd import _child_env_block

    def boom():
        raise RuntimeError("env loader unavailable")

    monkeypatch.setattr(env_loader, "launch_dotenv_keys", boom)
    monkeypatch.setattr(env_loader, "secret_source_names", boom)

    assert _child_env_block() is None


def test_json_discloses_inherited_credentials_without_values(tmp_path, capsys, monkeypatch):
    _toy_manifest(tmp_path)
    _fake_credential_sources(monkeypatch, ["TOY_API_KEY"], {"TOY_API_KEY": "sk-toy-secret"})

    code = run_verify_command(make_args(tmp_path, json=True, skip_start=True))

    out = capsys.readouterr().out
    assert code == 0
    payload = json.loads(out)
    assert payload["childEnv"] == {
        "credentialCount": 1,
        "credentialNames": ["TOY_API_KEY"],
        "valuesIncluded": False,
    }
    assert "sk-toy-secret" not in out


def test_human_report_counts_inherited_credentials(tmp_path, capsys, monkeypatch):
    _toy_manifest(tmp_path)
    _fake_credential_sources(monkeypatch, ["TOY_API_KEY"], {"TOY_API_KEY": "sk-toy-secret"})

    code = run_verify_command(make_args(tmp_path, skip_start=True))

    out = capsys.readouterr().out
    assert code == 0
    assert "Result: OK" in out
    assert "Credentials inherited by verify children: 1" in out
    assert "sk-toy-secret" not in out

