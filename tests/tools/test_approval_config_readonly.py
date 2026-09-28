"""Regression tests: the approval guard path reads config via
load_config_readonly() (no per-call deepcopy).

The guard path runs per terminal command. load_config() pays a defensive
deepcopy on every call (~356us of the ~376us warm-cache cost, measured on
a real config.yaml) and the guard path loaded config 2-3x per command.
Every swapped call site was audited read-only (all callers take scalar
reads or iterate; none mutate the returned dict or any nested structure),
so they now use load_config_readonly() — the API built for exactly this
(hermes_cli/config.py docstring; precedent: #74211, #74322).

These tests drive the REAL functions against a temp HERMES_HOME config
(AGENTS.md: E2E with real imports), not mocks of the seam under test.
"""
import pytest

import hermes_cli.config as hc
from tools.approval import check_all_command_guards, load_permanent_allowlist
from tools.approval_context import _get_approval_config
from tools.approval_context import _get_cron_approval_mode
from tools.tirith_security import _load_security_config


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  default: test-model\n"
        "approvals:\n  mode: manual\n  timeout: 300\n  cron_mode: deny\n"
        "command_allowlist: []\n"
        "security:\n  tirith_enabled: false\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    hc._LOAD_CONFIG_CACHE.clear()
    yield home
    hc._LOAD_CONFIG_CACHE.clear()


def test_readers_return_live_cache_without_corrupting_it(
        config_home, monkeypatch):
    """Guard-population check for the readonly swap: repeated reads return
    the same cached object and the cache stays intact — no swapped site
    may mutate what it returns."""
    first = _get_approval_config()
    second = _get_approval_config()
    assert first is second  # live cache object, no deepcopy
    # a full guard pass must leave the cache values untouched
    before = dict(first)
    check_all_command_guards("ls -la", "local")
    _get_cron_approval_mode()
    load_permanent_allowlist()
    _load_security_config()
    assert _get_approval_config() == before
    assert hc.load_config_readonly()["approvals"]["mode"] == "manual"
