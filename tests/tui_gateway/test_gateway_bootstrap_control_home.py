"""Ink bootstrap must acquire a served secondary's ticket from its multiplexer control home."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import gateway_runtime
import hermes_constants


_SCRIPT = Path(__file__).resolve().parents[2] / "ui-tui" / "scripts" / "gateway_bootstrap.py"


def _load_bootstrap():
    spec = importlib.util.spec_from_file_location("ink_gateway_bootstrap_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_served_secondary_uses_multiplexer_control_home(monkeypatch, tmp_path):
    module = _load_bootstrap()
    root = tmp_path.resolve()
    secondary = root / "profiles" / "cold"
    secondary.mkdir(parents=True)
    endpoint = SimpleNamespace(
        profile_id=str(secondary.resolve()),
        instance_id="mux-owner",
        api_origin="http://127.0.0.1:1234",
        runtime_protocol=1,
        control_home=str(root),
    )
    receipt = SimpleNamespace(state="ready", endpoint=endpoint, reason_code=None)

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: secondary)
    monkeypatch.setattr(gateway_runtime, "discover_gateway_endpoint", lambda home, timeout=5: receipt)

    seen = {}
    reply = json.dumps({
        "ok": True,
        "protocol": 1,
        "id": 1,
        "result": {
            "instance_id": endpoint.instance_id,
            "profile_id": endpoint.profile_id,
            "ticket": "served-grant",
            "runtime_protocol": 1,
        },
    }).encode() + b"\n"

    if module.os.name == "nt":
        import gateway.runtime_bootstrap_windows as windows_bootstrap

        def query_runtime_control(home, request, timeout):
            seen["control_home"] = Path(home).resolve()
            seen["request"] = json.loads(request)
            return reply

        monkeypatch.setattr(windows_bootstrap, "query_runtime_control", query_runtime_control)
    else:
        import hermes_cli.gateway_runtime_discovery as discovery

        stream = io.BytesIO(reply)

        class FakeSocket:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def sendall(self, request):
                seen["request"] = json.loads(request)

            def settimeout(self, _timeout):
                pass

            def recv(self, size):
                return stream.read(size)

        def connect_private(home, timeout):
            seen["control_home"] = Path(home).resolve()
            return FakeSocket()

        monkeypatch.setattr(discovery, "connect_private", connect_private)

    result = module.bootstrap(False)

    assert seen["control_home"] == root
    assert seen["request"]["params"] == {
        "profile_id": endpoint.profile_id,
        "instance_id": endpoint.instance_id,
        "purpose": "interactive",
    }
    assert result["profile_id"] == endpoint.profile_id


def _bootstrap_failure(monkeypatch, root, error):
    """The line the script's ``__main__`` prints (``str(exc)``, exit 1) when the ticket mint raises ``error``."""
    import hermes_cli.gateway_runtime_discovery as discovery

    module = _load_bootstrap()
    endpoint = SimpleNamespace(profile_id=str(root), instance_id="owner", api_origin="http://127.0.0.1:1234",
                               runtime_protocol=1, control_home=str(root))

    def connect_private(home, timeout):
        raise error

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: root)
    monkeypatch.setattr(gateway_runtime, "discover_gateway_endpoint",
                        lambda home, timeout=5: SimpleNamespace(state="ready", endpoint=endpoint, reason_code=None))
    monkeypatch.setattr(discovery, "connect_private", connect_private)
    with pytest.raises((OSError, RuntimeError, ValueError)) as failed:
        module.bootstrap(False)
    return str(failed.value)


@pytest.mark.platforms("posix")  # the mint dials connect_private; Windows uses the named pipe
def test_a_refused_mint_prints_its_repair_command(monkeypatch, tmp_path):
    """The control home's refusal reaches the TUI as the verdict `hermes gateway attach` prints: each
    line a `gateway inaccessible:` verdict Ink echoes inline, the last ending in the repair command."""
    from hermes_cli.gateway_runtime_discovery import _open_mode

    root = tmp_path.resolve()
    assert _bootstrap_failure(monkeypatch, root, _open_mode(root / "control", 0o755, 0o700)) == (
        "gateway inaccessible: ticket_unavailable\n"
        f"gateway inaccessible: {root / 'control'} has mode 0755, expected 0700; chmod 700 {root / 'control'}")


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("refusal", [None, "invalid_control_response"])
def test_a_mint_socket_error_still_prints_only_its_class(monkeypatch, tmp_path, refusal):
    """Raw socket text (paths, peer data) stays out, and a refusal with no repair hint (the owner
    answered ``ok: false``) still names its code under the ``ticket_unavailable`` head."""
    from hermes_cli.gateway_runtime_discovery import DiscoveryError

    error = DiscoveryError(refusal) if refusal else ConnectionResetError("peer /secret/path")
    assert _bootstrap_failure(monkeypatch, tmp_path.resolve(), error) == (
        f"gateway inaccessible: ticket_unavailable\ngateway inaccessible: {refusal or 'ConnectionResetError'}")
