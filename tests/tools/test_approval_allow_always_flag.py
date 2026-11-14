"""``approvals.allow_always: false`` hides the permanent scope and never writes command_allowlist.

Drives the real functions against a temp HERMES_HOME (positive control: with the flag absent the
same ``_persist_choice`` call DOES change config.yaml)."""
import hashlib

import pytest

import hermes_cli.config as hc
import tools.approval as ap

KEY = "script execution via heredoc"
KEY_OFF = "allow-always-off test key"  # distinct: the positive control leaves KEY approved in process memory


def _home(tmp_path, monkeypatch, approvals_extra=""):
    home = tmp_path / "hermes"
    home.mkdir()
    cfg = home / "config.yaml"
    cfg.write_text(f"approvals:\n  mode: manual\n{approvals_extra}command_allowlist: []\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    hc._LOAD_CONFIG_CACHE.clear()
    return cfg


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_positive_control_always_persists_by_default(tmp_path, monkeypatch):
    cfg = _home(tmp_path, monkeypatch)
    before = _sha(cfg)
    ap._persist_choice("s", "always", [KEY])
    assert _sha(cfg) != before and KEY in cfg.read_text()


@pytest.mark.parametrize("raw", ["false", "0", '"off"', '"no"', "False"])
def test_flag_off_never_writes_config(tmp_path, monkeypatch, raw):
    cfg = _home(tmp_path, monkeypatch, f"  allow_always: {raw}\n")
    before = _sha(cfg)
    assert ap._allow_always_enabled() is False
    ap._persist_choice("s", "always", [KEY_OFF])
    assert _sha(cfg) == before
    assert not ap.is_approved("s", KEY_OFF)  # downgraded to once: not even session-scoped


def test_flag_off_keeps_session_and_once(tmp_path, monkeypatch):
    cfg = _home(tmp_path, monkeypatch, "  allow_always: false\n")
    before = _sha(cfg)
    ap._persist_choice("s2", "session", ["k1"])
    ap._persist_choice("s2", "once", ["k2"])
    assert ap.is_approved("s2", "k1") and not ap.is_approved("s2", "k2")
    assert _sha(cfg) == before


@pytest.mark.parametrize("raw,offered", [("", True), ("  allow_always: true\n", True), ("  allow_always: false\n", False)])
def test_mcp_consent_card_offers_always_only_when_enabled(tmp_path, monkeypatch, raw, offered):
    import threading

    import tools.approval_context as ctx
    import tools.approval_prompt as pr
    _home(tmp_path, monkeypatch, "  timeout: 2\n" + raw)
    monkeypatch.setattr(ctx, "_is_gateway_approval_context", lambda: True)
    monkeypatch.setattr(ctx, "_is_single_query_approval_context", lambda: False)
    monkeypatch.setattr(ctx, "get_current_session_key", lambda: "mcp-t")
    seen = {}

    def notify(data):
        seen.update(data)
        threading.Timer(0.2, ap.resolve_gateway_approval, ("mcp-t", "deny")).start()

    ap.register_gateway_notify("mcp-t", notify)
    try:
        pr.request_elicitation_consent("ok?", "test", remember_key="mcp:test")
    finally:
        ap.unregister_gateway_notify("mcp-t")
    assert seen["allow_permanent"] is offered and seen["allow_session"] is True


def _typed(monkeypatch, answer):
    import tools.approval_prompt as pr
    monkeypatch.setattr(pr, "_read_choice", lambda prompt, timeout: answer)
    return pr


@pytest.mark.parametrize("answer", ["a", "always"])
def test_cli_stdin_typed_always_is_once_when_flag_off(tmp_path, monkeypatch, answer):
    # The CLI stdin fallback used to turn a typed "a" into "session" when [a]lways was hidden: that
    # would still approve every later command of the session. With the flag off it must stay "once".
    cfg = _home(tmp_path, monkeypatch, "  allow_always: false\n")
    before = _sha(cfg)
    pr = _typed(monkeypatch, answer)
    choice = pr._ask_human("cmd", "desc", 5, allow_permanent=ap._allow_always_enabled(), approval_callback=None,
                           allow_session=True, smart_denied=False)
    assert choice == "once"
    ap._persist_choice("s3", choice, [KEY_OFF])
    assert not ap.is_approved("s3", KEY_OFF) and _sha(cfg) == before


def test_cli_stdin_typed_always_unchanged_when_flag_on(tmp_path, monkeypatch):
    _home(tmp_path, monkeypatch)
    pr = _typed(monkeypatch, "a")
    assert pr._ask_human("cmd", "desc", 5, allow_permanent=True, approval_callback=None,
                         allow_session=True, smart_denied=False) == "always"
