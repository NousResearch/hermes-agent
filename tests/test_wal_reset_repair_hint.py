"""Table-driven tests for the WAL-reset repair hint's install-type wording.

The hint must never promise a repair path the install cannot deliver (#79179): ``git``/``unknown``
describe the *code layout*, not who owns the running interpreter. The ``hermes update`` wording
additionally requires the interpreter to live in the project's own ``venv``/``.venv`` — a checkout
driven by a system Python or an externally managed venv keeps its linked SQLite whatever
``hermes update`` does to the code tree, and has to be told to upgrade the interpreter instead.
"""

import sys

import pytest

import hermes_cli.config
from hermes_state_wal import _EXTERNAL_RUNTIME_REPAIR_HINT, _wal_reset_repair_hint


def _hint_for_install(monkeypatch, project_root, method, *, interpreter_prefix=None):
    """Run the hint with a faked install method, project root, and interpreter location."""
    monkeypatch.setattr(hermes_cli.config, "detect_install_method", lambda root: method)
    monkeypatch.setattr(hermes_cli.config, "get_project_root", lambda: project_root)
    if interpreter_prefix is not None:
        monkeypatch.setattr(sys, "prefix", str(interpreter_prefix))
    return _wal_reset_repair_hint()


@pytest.mark.parametrize("venv_name", ["venv", ".venv"])
@pytest.mark.parametrize("method", ["git", "unknown"])
def test_project_venv_interpreter_keeps_managed_wording(tmp_path, monkeypatch, method, venv_name):
    venv_dir = tmp_path / venv_name
    venv_dir.mkdir()
    hint = _hint_for_install(monkeypatch, tmp_path, method, interpreter_prefix=venv_dir)
    assert hint == "Hermes-managed installs can repair the embedded runtime with `hermes update`"


@pytest.mark.parametrize("method", ["git", "unknown"])
def test_external_interpreter_gets_runtime_upgrade_wording(tmp_path, monkeypatch, method):
    # The project venv exists, but the interpreter runs from a venv outside the project tree:
    # `hermes update` cannot replace the SQLite linked into it (#79179 reproduction).
    (tmp_path / "venv").mkdir()
    external = tmp_path.parent / "hermes-external-venv"
    external.mkdir(exist_ok=True)
    hint = _hint_for_install(monkeypatch, tmp_path, method, interpreter_prefix=external)
    assert hint == _EXTERNAL_RUNTIME_REPAIR_HINT
    assert "Hermes-managed installs" not in hint


def test_git_checkout_without_any_venv_gets_runtime_upgrade_wording(tmp_path, monkeypatch):
    # System-Python checkout: no project venv at all, `hermes update` has nothing to rebuild.
    hint = _hint_for_install(monkeypatch, tmp_path, "git", interpreter_prefix=tmp_path.parent)
    assert hint == _EXTERNAL_RUNTIME_REPAIR_HINT


def test_docker_and_nix_wording_is_unchanged(tmp_path, monkeypatch):
    hint = _hint_for_install(monkeypatch, tmp_path, "docker", interpreter_prefix=tmp_path)
    assert hint == "update the container image with `docker pull nousresearch/hermes-agent:latest`"
    hint = _hint_for_install(monkeypatch, tmp_path, "nix", interpreter_prefix=tmp_path)
    assert hint.startswith("Update Hermes through the Nix source")


def test_probe_failure_keeps_generic_sqlite_guidance(tmp_path, monkeypatch):
    def _explode(root):
        raise RuntimeError("install probe unavailable")

    monkeypatch.setattr(hermes_cli.config, "detect_install_method", _explode)
    monkeypatch.setattr(hermes_cli.config, "get_project_root", lambda: tmp_path)
    assert _wal_reset_repair_hint() == (
        "install a Python build bundled with SQLite 3.51.3+ (or backports 3.50.7 / 3.44.6) and restart Hermes"
    )
