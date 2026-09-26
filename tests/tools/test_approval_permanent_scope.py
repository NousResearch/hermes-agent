"""``approvals.allow_permanent: false`` (issue #90217) removes the permanent approval scope from
every surface and refuses an ``always`` a stale client sends anyway.

Drives the real gate against a temp ``HERMES_HOME`` config, like
``tests/tools/test_approval_config_readonly.py`` — no mocks of the seam under test.
"""

import pytest

import hermes_cli.config as hc
from tools import approval as approval_module
from tools.approval import check_all_command_guards, detect_dangerous_command

DANGEROUS = "rm -rf /var/data"


@pytest.fixture(autouse=True)
def _clean_approval_state():
    def _clear():
        for name in ("_session_approved", "_permanent_approved", "_permanent_approved_by_home",
                     "_permanent_baseline_by_home", "_pending"):
            getattr(approval_module, name).clear()
    _clear()
    yield
    _clear()


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    """Writer for a temp ``HERMES_HOME``; ``allow_permanent`` is set per test."""

    def _make(allow_permanent: bool | None = None):
        home = tmp_path / "hermes"
        home.mkdir()
        approvals = "approvals:\n  mode: manual\n  timeout: 300\n"
        if allow_permanent is not None:
            approvals += f"  allow_permanent: {'true' if allow_permanent else 'false'}\n"
        (home / "config.yaml").write_text(
            "model:\n  default: test-model\n" + approvals
            + "command_allowlist: []\n"
            "security:\n  tirith_enabled: false\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(home))
        # Only an interactive CLI has a human to ask, so the gate prompts instead of auto-denying.
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        hc._LOAD_CONFIG_CACHE.clear()
        return home

    yield _make
    hc._LOAD_CONFIG_CACHE.clear()


def _answering(choice: str, seen: dict | None = None):
    def cb(command, description, *, allow_permanent=True, **kwargs):
        if seen is not None:
            seen["allow_permanent"] = allow_permanent
        return choice
    return cb


def test_allow_permanent_false_hides_the_always_scope(config_home):
    """The gate stops advertising ``always``, so no surface can render it."""
    config_home(allow_permanent=False)
    seen = {}

    result = check_all_command_guards(DANGEROUS, "local", approval_callback=_answering("deny", seen))

    assert seen["allow_permanent"] is False
    assert result["approved"] is False


def test_allow_permanent_false_refuses_a_stale_always(config_home):
    """A client answering ``always`` anyway gets one command, not a permanent allowlist key."""
    config_home(allow_permanent=False)
    pattern_key = detect_dangerous_command(DANGEROUS)[1]

    result = check_all_command_guards(DANGEROUS, "local", approval_callback=_answering("always"))

    assert result["approved"] is True  # the human did approve this one command
    assert approval_module._is_permanently_approved(pattern_key) is False
    assert hc.load_config_readonly().get("command_allowlist") == []


def test_always_still_persists_when_permanent_scope_is_allowed(config_home):
    """Control: with the key absent (default true) ``always`` writes the allowlist as before."""
    config_home()
    pattern_key = detect_dangerous_command(DANGEROUS)[1]

    result = check_all_command_guards(DANGEROUS, "local", approval_callback=_answering("always"))

    assert result["approved"] is True
    assert approval_module._is_permanently_approved(pattern_key) is True
    assert pattern_key in hc.load_config_readonly().get("command_allowlist")
