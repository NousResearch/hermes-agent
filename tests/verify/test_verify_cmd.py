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


def test_human_report(tmp_path, capsys):
    manifest = tmp_path / ".hermes"
    manifest.mkdir()
    (manifest / "environment.json").write_text(
        json.dumps({"recipe": {"name": "Fake", "test": ["echo ok"]}}),
        encoding="utf-8",
    )
    code = run_verify_command(make_args(tmp_path, skip_start=True))
    out = capsys.readouterr().out
    assert code == 0
    assert "Recipe: Fake" in out
    assert "PASS" in out
    assert "Result: OK" in out


def test_bad_path(tmp_path, capsys):
    code = run_verify_command(make_args(tmp_path / "nope"))
    assert code == 2
