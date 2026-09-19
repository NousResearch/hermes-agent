"""Table-driven tests for the WAL-reset repair hint's install-type wording.

The hint must never promise a repair path the install cannot deliver (#79179): ``git``/``unknown``
describe the *code layout*, not who owns the running interpreter. ``hermes update``'s runtime
repair cuts over the checkout's live venv, so the gate follows the repair target's own
resolution — interpreter-file presence with the managed ``venv`` winning over ``.venv``,
matching ``hermes_cli.managed_uv._default_live_venv`` — rather than ``project_venv_dir``'s
directory-presence rule.
"""

import sys

import pytest

import hermes_cli.config
from hermes_constants import venv_python_path
from hermes_state_wal import _EXTERNAL_RUNTIME_REPAIR_HINT, _HYBRID_RUNTIME_REPAIR_HINT, _wal_reset_repair_hint


def _hint_for_install(monkeypatch, project_root, method, *, interpreter_prefix=None):
    """Run the hint with a faked install method, project root, and interpreter location."""
    monkeypatch.setattr(hermes_cli.config, "detect_install_method", lambda root: method)
    monkeypatch.setattr(hermes_cli.config, "get_project_root", lambda: project_root)
    if interpreter_prefix is not None:
        monkeypatch.setattr(sys, "prefix", str(interpreter_prefix))
    return _wal_reset_repair_hint()


def _make_live_venv(root, name):
    """Create a venv skeleton whose interpreter file exists (platform-correct layout)."""
    venv = root / name
    venv.mkdir(exist_ok=True)
    python = venv_python_path(venv)
    python.parent.mkdir(parents=True, exist_ok=True)
    python.touch()
    return venv


@pytest.mark.parametrize("venv_name", ["venv", ".venv"])
@pytest.mark.parametrize("method", ["git", "unknown"])
def test_project_venv_interpreter_keeps_managed_wording(tmp_path, monkeypatch, method, venv_name):
    venv_dir = _make_live_venv(tmp_path, venv_name)
    hint = _hint_for_install(monkeypatch, tmp_path, method, interpreter_prefix=venv_dir)
    assert hint == "Hermes-managed installs can repair the embedded runtime with `hermes update`"


@pytest.mark.parametrize("method", ["git", "unknown"])
def test_empty_venv_dir_falls_back_to_live_dot_venv(tmp_path, monkeypatch, method):
    # Gate primitive must match the repair target: an empty `venv/` next to a live `.venv`
    # makes `project_venv_dir` (directory presence, `venv` wins) and the repair target
    # `_default_live_venv` (interpreter file, `venv` wins) disagree — the hint follows the
    # repair target, so this install stays repair-capable (#79179 review).
    (tmp_path / "venv").mkdir()
    live = _make_live_venv(tmp_path, ".venv")
    hint = _hint_for_install(monkeypatch, tmp_path, method, interpreter_prefix=live)
    assert hint == "Hermes-managed installs can repair the embedded runtime with `hermes update`"


@pytest.mark.parametrize("method", ["git", "unknown"])
def test_hybrid_checkout_gets_rebuild_and_relaunch_wording(tmp_path, monkeypatch, method):
    # A live project venv exists but an external interpreter runs: `hermes update` still
    # rebuilds that venv, so the hint points at update + relaunching from it instead of
    # foreclosing the zero-upgrade repair path (#79179 review).
    live = _make_live_venv(tmp_path, "venv")
    external = tmp_path.parent / "hermes-external-venv"
    external.mkdir(exist_ok=True)
    hint = _hint_for_install(monkeypatch, tmp_path, method, interpreter_prefix=external)
    assert hint == _HYBRID_RUNTIME_REPAIR_HINT.format(cmd="hermes update", venv=venv_python_path(live))
    assert "restart Hermes from" in hint


@pytest.mark.parametrize("method", ["git", "unknown"])
def test_venv_without_interpreter_gets_runtime_upgrade_wording(tmp_path, monkeypatch, method):
    # A leftover venv directory with no interpreter file is not a repair target — there is
    # nothing for `hermes update` to cut over, so the wording points at the external runtime.
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
