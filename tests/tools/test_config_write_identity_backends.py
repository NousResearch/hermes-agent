"""The config.yaml file-identity check (#117815) is a HOST check: it applies only where the write lands
on this host's disk. On a remote backend the same path string names a file on that backend."""

import os

import pytest

import tools.file_tools_write_guards as guards
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def config_alias(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("approvals:\n  mode: manual\n", encoding="utf-8")
    alias = tmp_path / "alias.yaml"
    os.link(home / "config.yaml", alias)
    tok = set_hermes_home_override(str(home))
    try:
        yield str(alias)
    finally:
        reset_hermes_home_override(tok)


def test_a_host_alias_of_config_is_refused_on_the_local_backend(config_alias, monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    err = guards._check_sensitive_path(config_alias, "default")
    assert err is not None and "Refusing to write to Hermes config file" in err


def test_a_remote_backend_write_is_not_judged_by_a_host_file_at_that_path(config_alias, monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    assert guards._check_sensitive_path(config_alias, "default") is None
