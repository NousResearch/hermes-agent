"""Multiplexed-gateway parity for what a TURN sees: a profile served by the default multiplexer
(``_profile_runtime_scope``) must observe the same tool-side policy its standalone gateway
(``HERMES_HOME=<profile>``) would — never the launch profile's values frozen into process caches or
read from the process env.

Every test warms the site under launch home A, then reads under routed profile B with different
config (real temp homes, real config.yaml, real terminal scope; no mocks of the thing under test).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def two_homes(tmp_path, monkeypatch):
    """Launch home A (HERMES_HOME) and routed profile B, differing in every setting under test."""
    a = tmp_path / ".hermes"
    b = a / "profiles" / "b"
    b.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(a))
    for name, home in (("a", a), ("b", b)):
        (home / f"cred_{name}.txt").write_text("x", encoding="utf-8")
        (home / "config.yaml").write_text(yaml.safe_dump({
            "terminal": {"backend": "local" if name == "a" else "docker",
                         "credential_files": [f"cred_{name}.txt"]},
            "command_allowlist": [f"{name}-only-cmd *"],
            "security": {"redact_secrets": name == "b"},
            "browser": {"engine": "chrome" if name == "a" else "lightpanda", "headed": name == "b"},
            "lsp": {"enabled": name == "b"},
        }), encoding="utf-8")
    return a, b


def _under(home: Path, fn):
    token = set_hermes_home_override(str(home))
    try:
        return fn()
    finally:
        reset_hermes_home_override(token)


def test_routed_local_profile_cwd_matches_standalone_gateway(tmp_path, two_homes):
    """A standalone gateway resolves an unset ``terminal.cwd`` on a local backend to ``$HOME`` at
    import; the routed profile's terminal scope must yield the same cwd, not the multiplexer's
    process cwd — otherwise the system prompt, context files and the terminal all start in
    wherever ``hermes gateway`` happened to be launched from."""
    from tools.terminal_scope import build_profile_terminal_scope, install_and_reset_profile_terminal_scope
    from agent.runtime_cwd import resolve_agent_cwd

    a, _ = two_homes
    assert build_profile_terminal_scope(a)["TERMINAL_CWD"] == str(tmp_path)
    with install_and_reset_profile_terminal_scope(a):
        assert resolve_agent_cwd() == tmp_path
    # Non-local backends stay unset (sandbox default), exactly like gateway/run.py's placeholder rule.
    assert "TERMINAL_CWD" not in build_profile_terminal_scope(two_homes[1])


def test_terminal_backend_consumers_read_the_routed_scope(two_homes):
    """Every ``TERMINAL_ENV`` reader that shapes a turn (image-source locality, credential-file path
    translation, image-gen cache base, skill readiness) resolves the ROUTED profile's backend."""
    from gateway.run import _profile_runtime_scope
    from tools import credential_files, image_generation_tool, image_source

    a, b = two_homes

    def observe():
        return (
            image_source._is_local_terminal_backend(),
            image_generation_tool._agent_cache_base_for_env(None),
            credential_files.to_agent_visible_cache_path("/host/.hermes/x.png", "/root/.hermes"),
            sorted(Path(m["host_path"]).name for m in credential_files.get_credential_file_mounts()),
        )

    with _profile_runtime_scope(a):
        assert observe() == (True, None, "/host/.hermes/x.png", ["cred_a.txt"])
    with _profile_runtime_scope(b):
        local, cache_base, translated, mounts = observe()
    assert local is False
    assert cache_base == "/root/.hermes"
    assert translated != "/host/.hermes/x.png" or mounts == ["cred_b.txt"]
    assert mounts == ["cred_b.txt"]


def test_permanent_allowlist_is_per_profile(two_homes):
    """Profile A's ``command_allowlist`` must not pre-approve commands for routed profile B, and
    B's own 'always' approvals must not be folded into A's set."""
    from tools import approval

    a, b = two_homes
    approval.load_permanent_allowlist()  # launch-profile startup load (A)
    assert approval.is_approved("s", "a-only-cmd *")
    assert _under(b, lambda: approval.is_approved("s", "a-only-cmd *")) is False
    assert _under(b, lambda: approval.is_approved("s", "b-only-cmd *")) is True
    _under(b, lambda: approval.approve_permanent("b-always *"))
    assert not approval.is_approved("s", "b-always *")
    assert _under(a, lambda: approval.is_approved("s", "a-only-cmd *"))


def test_profile_scoped_process_caches_follow_routed_home(two_homes, monkeypatch):
    """Config-derived singletons (redaction switch, aux unhealthy marks, LSP service, browser
    engine/headed flags, MCP stderr log path) are keyed by the routed profile home."""
    from agent import auxiliary_client, lsp, redact
    from tools import browser_tool_cloud, mcp_tool_config
    from tools.browser_tool_lifecycle import cleanup_all_browsers

    a, b = two_homes
    monkeypatch.setattr(redact, "_REDACT_ENABLED", False)  # launch profile opted out
    redact._REDACT_ENABLED_BY_HOME.clear()
    token = "sk-abcdefghijklmnopqrstuvwxyz0123456789"
    assert redact.redact_sensitive_text(token) == token
    assert _under(b, lambda: redact.redact_sensitive_text(token)) != token

    auxiliary_client._reset_aux_unhealthy_cache()
    _under(a, lambda: auxiliary_client._mark_provider_unhealthy("openrouter"))
    assert _under(a, lambda: auxiliary_client._is_provider_unhealthy("openrouter")) is True
    assert _under(b, lambda: auxiliary_client._is_provider_unhealthy("openrouter")) is False

    lsp.shutdown_service()
    try:
        assert _under(a, lsp.get_service) is None
        assert _under(b, lsp.get_service) is not None
    finally:
        lsp.shutdown_service()

    cleanup_all_browsers()
    assert _under(a, browser_tool_cloud._get_browser_engine) == "chrome"
    assert _under(b, browser_tool_cloud._get_browser_engine) == "lightpanda"
    assert _under(a, browser_tool_cloud._is_headed_mode) is False
    assert _under(b, browser_tool_cloud._is_headed_mode) is True

    mcp_tool_config._mcp_stderr_log_fh.clear()
    try:
        assert Path(_under(a, mcp_tool_config._get_mcp_stderr_log).name).parent == a / "logs"
        assert Path(_under(b, mcp_tool_config._get_mcp_stderr_log).name).parent == b / "logs"
    finally:
        for fh in mcp_tool_config._mcp_stderr_log_fh.values():
            fh.close()
        mcp_tool_config._mcp_stderr_log_fh.clear()


# --- approval-required review-policy revocation under the multiplexer ------------------------------

_PATTERN = "kubectl *--context[= ]admin*"
_KUBECTL = "kubectl --context admin -n vault exec vault-0 -- du -sh /vault/data"
_UNRELATED = "ls -la /tmp"


@pytest.fixture
def two_homes_with_rule(two_homes, monkeypatch):
    """``two_homes`` plus a live ``command_approval_required`` rule in EACH home, and a
    ``set_review(home, review)`` that rewrites one home's rule with a strictly increasing mtime
    (the config cache is keyed on ``(mtime_ns, size)`` and ``smart`` -> ``human`` keeps the size)."""
    import os
    import time

    import hermes_cli.config as hc

    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    clock = {"ns": time.time_ns()}

    def set_review(home: Path, review: str):
        path = home / "config.yaml"
        current = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        current["approvals"] = {"mode": "smart", "command_approval_required": [
            {"pattern": _PATTERN, "description": "kubectl on admin", "review": review}]}
        current["security"] = dict(current.get("security") or {}, tirith_enabled=False)
        path.write_text(yaml.safe_dump(current), encoding="utf-8")
        clock["ns"] += 1_000_000_000
        os.utime(path, ns=(clock["ns"], clock["ns"]))

    a, b = two_homes
    for home in (a, b):
        set_review(home, "smart")
    hc._LOAD_CONFIG_CACHE.clear()
    yield set_review
    hc._LOAD_CONFIG_CACHE.clear()


def _allowlist_on_disk(home: Path) -> list:
    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")).get("command_allowlist") or []


def test_review_policy_revocation_is_scoped_to_the_routed_profile(two_homes_with_rule, two_homes, monkeypatch):
    """An Always grant taken in routed profile B is revoked from B's own permanent set and B's own
    ``config.yaml`` when B's rule tightens ``smart -> human`` — never from the launch profile's
    (PR #106779 review, F-001). Permanent state is per profile since the profile-isolation change,
    so a revocation that reads the module-level set fixes the wrong profile in both directions.
    """
    from tools import approval, approval_floors, approval_smart
    from tools.approval import check_all_command_guards
    from tools.approval_context import reset_current_session_key, set_current_session_key

    set_review = two_homes_with_rule
    a, b = two_homes
    smart_key = approval_floors._RequiredRule(_PATTERN, "kubectl on admin", "smart").key
    guardian = []
    monkeypatch.setattr(approval_smart, "_smart_approve",
                        lambda command, description: guardian.append(command) or "escalate")
    prompts = []
    answers = iter(["always", "deny"])
    monkeypatch.setattr(approval, "prompt_dangerous_approval",
                        lambda command, description, **kw: prompts.append(kw.get("allow_permanent")) or next(answers))
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval, "_permanent_approved", set())
    monkeypatch.setattr(approval, "_permanent_approved_by_home", {})
    monkeypatch.setattr(approval, "_session_approved", {})
    approval_floors._observed_review.clear()

    session = "test:multiplex:approval-required"
    token = set_current_session_key(session)
    try:
        approval.load_permanent_allowlist()  # launch-profile (A) startup load
        a_before = _allowlist_on_disk(a)

        # 1. Always granted in B, under B's smart rule: persisted into B's set and B's config only.
        assert _under(b, lambda: check_all_command_guards(_KUBECTL, "local"))["approved"] is True
        assert smart_key in _allowlist_on_disk(b)
        assert smart_key not in _allowlist_on_disk(a)
        assert _under(b, lambda: smart_key in approval._permanent_set())
        assert smart_key not in approval._permanent_approved

        # 2. B's rule tightens; an unrelated guarded command in B observes the transition.
        set_review(b, "human")
        approval_floors._observed_review.clear()  # a fresh sighting, as in a new process
        assert _under(b, lambda: check_all_command_guards(_UNRELATED, "local"))["approved"] is True
        assert smart_key not in _allowlist_on_disk(b), "B's persisted grant was revoked"
        assert _under(b, lambda: smart_key not in approval._permanent_set())
        assert _allowlist_on_disk(a) == a_before, "the launch profile's config was not touched"

        # 3. Back to smart in B: the restored policy needs a fresh guardian AND person decision.
        set_review(b, "smart")
        approval_floors._observed_review.clear()
        assert _under(b, lambda: check_all_command_guards(_KUBECTL, "local"))["approved"] is False
        assert guardian == [_KUBECTL, _KUBECTL]
        assert prompts == [True, True]
    finally:
        reset_current_session_key(token)
        approval.clear_session(session)
