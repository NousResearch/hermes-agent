"""Tests: skills.manage install path (tui_gateway/methods_tools.py::_skills_install).

The RPC runs `do_install(skip_confirm=True)` with a sink console. Contracts:
- a successful install answers `installed: true`;
- a failed install (unresolved name, fetch failure, security-scan block) must answer
  an error envelope whose message is the tail the CLI user would have read — before
  this, every failure returned `installed: true` and the verdict was discarded
  (#63307 Part B).
"""

import tui_gateway.server as srv


def _install(params):
    return srv._methods["skills.manage"](1, params)


def _stub_do_install(monkeypatch, verdict, lines):
    """Patch do_install at the module the RPC resolves it from."""
    import hermes_cli.skills_hub as cli_hub

    class _Sink:
        def __init__(self):
            self.lines = []

        def print(self, *args, **kwargs):
            self.lines.append(" ".join(str(a) for a in args))

    def fake_do_install(identifier, **kwargs):
        console = kwargs.get("console")
        for line in lines:
            console.print(line)
        return verdict

    monkeypatch.setattr(cli_hub, "do_install", fake_do_install)
    return _Sink()


def test_install_success_reports_installed(monkeypatch):
    _stub_do_install(monkeypatch, True, ["Fetching: official/x", "Installed: x"])

    out = _install({"action": "install", "query": "official/x"})

    assert "error" not in out
    assert out["result"] == {"installed": True, "name": "official/x"}


def test_install_failure_is_an_error_not_success(monkeypatch):
    """A scan-blocked install answers an error envelope carrying the scan verdict's
    last line; the old shape (`installed: true` on every path) hid the block."""
    _stub_do_install(
        monkeypatch,
        False,
        [
            "Fetching: clawhub/org/evil",
            "Running security scan...",
            "Not installed: the security scan found 2 high-risk pattern(s) in "
            "'org/evil' (listed above).",
        ],
    )

    out = _install({"action": "install", "query": "clawhub/org/evil"})

    assert "error" in out
    assert out["error"]["code"] == 5031
    assert "security scan found" in out["error"]["message"]
    assert out["error"]["data"]["installed"] is False
    assert "Not installed: the security scan found" in out["error"]["data"]["log"]


def test_install_declined_is_still_not_reported_as_installed(monkeypatch):
    """A user-owned no-op (None verdict: already installed / declined) must not claim
    a fresh install either — `installed` tracks the actual outcome."""
    _stub_do_install(monkeypatch, None, ["Warning: 'x' is already installed at /skills/x"])

    out = _install({"action": "install", "query": "x"})

    assert "error" in out
    assert out["error"]["code"] == 5031
    assert out["error"]["data"]["installed"] is False
