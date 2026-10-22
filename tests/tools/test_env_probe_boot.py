"""Backends start the host toolchain probe at boot, so a new chat's first prompt build finds it cached."""

import pytest

from tools import env_probe


@pytest.fixture(autouse=True)
def _probe_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    env_probe._reset_cache_for_tests()
    yield
    env_probe._reset_cache_for_tests()


def _config(tmp_path, text: str) -> None:
    (tmp_path / "config.yaml").write_text(text, encoding="utf-8")


def test_boot_warm_runs_the_probe_once_for_the_first_prompt(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(env_probe, "_build_probe_line", lambda: calls.append(1) or "Python toolchain: fixture.")
    _config(tmp_path, "agent: {}\n")

    env_probe.warm_environment_probe_at_boot()
    assert env_probe._PROBE_DONE.wait(timeout=5)

    assert env_probe.get_environment_probe_line() == "Python toolchain: fixture."
    assert calls == [1]


def test_boot_warm_honours_environment_probe_off(tmp_path, monkeypatch):
    monkeypatch.setattr(env_probe, "_build_probe_line", lambda: pytest.fail("host inspected"))
    _config(tmp_path, "agent:\n  environment_probe: false\n")

    env_probe.warm_environment_probe_at_boot()

    assert env_probe._PROBE_THREAD is None


def test_boot_warm_never_raises(monkeypatch):
    def broken_config():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", broken_config)

    env_probe.warm_environment_probe_at_boot()

    assert env_probe._PROBE_THREAD is None
