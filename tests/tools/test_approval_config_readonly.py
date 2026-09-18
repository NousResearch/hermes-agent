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
from tools.approval_context import _get_approval_config, _get_approval_mode
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


def _patched_loaders(monkeypatch):
    """Count BOTH loader variants. (A boom on load_config is useless here —
    every call site wraps the load in try/except and would swallow it; a
    pass-through counter is the robust form. The pins are: legacy
    load_config == 0 calls, load_config_readonly == the expected count,
    and cache identity — none satisfiable by the pre-fix code.)"""
    calls = {"readonly": 0, "legacy": 0}

    real_ro = hc.load_config_readonly
    real_legacy = hc.load_config

    def counting_ro():
        calls["readonly"] += 1
        return real_ro()

    def counting_legacy():
        calls["legacy"] += 1
        return real_legacy()

    monkeypatch.setattr(hc, "load_config_readonly", counting_ro)
    monkeypatch.setattr(hc, "load_config", counting_legacy)
    return calls


def test_guard_never_calls_deepcopy_variant(config_home, monkeypatch):
    """Pin: a full guard pass must not pay one deepcopying load_config.
    Fails pre-fix (the guard called load_config 2x per invocation)."""
    calls = _patched_loaders(monkeypatch)
    check_all_command_guards("ls -la", "local")
    assert calls["legacy"] == 0, (
        f"guard path called deepcopying load_config "
        f"{calls['legacy']}x — regression reintroduces the deepcopy cost")
    assert calls["readonly"] >= 1


def test_config_readers_never_call_deepcopy_variant(config_home, monkeypatch):
    calls = _patched_loaders(monkeypatch)
    assert _get_approval_mode() == "manual"
    assert _get_approval_config().get("timeout") == 300
    assert _get_cron_approval_mode() == "deny"
    assert load_permanent_allowlist() == set()
    sec = _load_security_config()
    assert sec["tirith_enabled"] is False
    assert calls["legacy"] == 0
    assert calls["readonly"] == 5  # one readonly load per function


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
