"""The chat PTY's bootstrap names the dashboard login on its ticket request (#144140).

``gateway_bootstrap`` mints the TUI's session-ticket over the private control
socket. When the dashboard hands its authenticated login down with the chat PTY
(``HERMES_TUI_SESSION_SUBJECT``), the request names it as the ticket subject so
the resume admits as that login: an issuer-less login can then adopt the
``desktop`` rows serve stamped with its ``<provider>:<user>`` label (that label
drops the issuer, so a self-hosted OIDC login's stay refused). The mint's
verified-peer relabel is #138915's
(tests/gateway/test_control_socket_forwarded_subject.py), and without the
variable the request keeps its anonymous shape
(tests/tui_gateway/test_gateway_bootstrap_control_home.py). The helper takes the
variable out of its own environment first, so an owner it starts never inherits
one login's identity.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

from hermes_cli import gateway_runtime
import hermes_constants


_SCRIPT = Path(__file__).resolve().parents[2] / "ui-tui" / "scripts" / "gateway_bootstrap.py"

_FORWARDED = 'auth:v1:["basic","","david"]'


def _load_bootstrap():
    spec = importlib.util.spec_from_file_location("ink_gateway_bootstrap_subject_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_discovery(monkeypatch, module, endpoint, seen):
    receipt = SimpleNamespace(state="ready", endpoint=endpoint, reason_code=None)

    def ensure(home, timeout=30, idle_exit=False):
        # What an owner started here inherits: spawn_unmanaged_gateway copies os.environ.
        seen["started_with"] = os.environ.get("HERMES_TUI_SESSION_SUBJECT")
        return receipt

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: Path("/tmp/any-home"))
    monkeypatch.setattr(gateway_runtime, "discover_gateway_endpoint",
                        lambda home, timeout=5: receipt)
    monkeypatch.setattr(gateway_runtime, "ensure_gateway_runtime", ensure)
    grant = json.dumps({
        "ok": True, "id": 1,
        "result": {"instance_id": endpoint.instance_id,
                   "profile_id": endpoint.profile_id, "ticket": "grant-1"},
    }).encode()
    if module.os.name == "nt":
        import gateway.runtime_bootstrap_windows as windows_bootstrap

        def query_runtime_control(_control, request, _timeout):
            seen["request"] = json.loads(request)
            return grant

        monkeypatch.setattr(windows_bootstrap, "query_runtime_control", query_runtime_control)
        return

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def readline(self, _limit):
            return grant

    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def sendall(self, request):
            seen["request"] = json.loads(request)

        def makefile(self, _mode):
            return FakeStream()

    import hermes_cli.gateway_runtime_discovery as discovery
    monkeypatch.setattr(discovery, "connect_private", lambda home, timeout: FakeSocket())


def _endpoint(tmp_path):
    return SimpleNamespace(
        profile_id=str(tmp_path), instance_id="inst-1",
        api_origin="http://127.0.0.1:1234", runtime_protocol=1,
        control_home=str(tmp_path))


def test_the_forwarded_login_is_named_on_the_ticket_request(monkeypatch, tmp_path):
    module = _load_bootstrap()
    endpoint = _endpoint(tmp_path)
    seen = {}
    _fake_discovery(monkeypatch, module, endpoint, seen)
    monkeypatch.setenv("HERMES_TUI_SESSION_SUBJECT", _FORWARDED)

    module.bootstrap(False)

    assert seen["request"]["params"]["subject"] == _FORWARDED


def test_an_owner_the_bootstrap_starts_never_inherits_the_login(monkeypatch, tmp_path):
    """``--start`` ensures the owner from this helper's environment: one chat's login is out of it
    by then, and still names that chat's ticket."""
    module = _load_bootstrap()
    endpoint = _endpoint(tmp_path)
    seen = {}
    _fake_discovery(monkeypatch, module, endpoint, seen)
    monkeypatch.setenv("HERMES_TUI_SESSION_SUBJECT", _FORWARDED)

    module.bootstrap(True)

    assert seen["started_with"] is None
    assert seen["request"]["params"]["subject"] == _FORWARDED
