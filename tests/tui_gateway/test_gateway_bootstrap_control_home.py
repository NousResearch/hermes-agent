"""Ink bootstrap must acquire a served secondary's ticket from its multiplexer control home."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import gateway.runtime as gateway_runtime
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

    if module.os.name == "nt":
        import gateway.runtime_bootstrap_windows as windows_bootstrap

        def query_runtime_control(home, request, timeout):
            seen["control_home"] = Path(home).resolve()
            seen["request"] = json.loads(request)
            return json.dumps({
                "ok": True,
                "id": 1,
                "result": {
                    "instance_id": endpoint.instance_id,
                    "profile_id": endpoint.profile_id,
                    "ticket": "served-grant",
                },
            }).encode()

        monkeypatch.setattr(windows_bootstrap, "query_runtime_control", query_runtime_control)
    else:
        import gateway.runtime_discovery as discovery

        class FakeStream:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def readline(self, _limit):
                return json.dumps({
                    "ok": True,
                    "id": 1,
                    "result": {
                        "instance_id": endpoint.instance_id,
                        "profile_id": endpoint.profile_id,
                        "ticket": "served-grant",
                    },
                }).encode()

        class FakeSocket:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def settimeout(self, timeout):
                seen["timeout"] = timeout

            def connect(self, target):
                seen["socket"] = target

            def sendall(self, request):
                seen["request"] = json.loads(request)

            def makefile(self, _mode):
                return FakeStream()

        def socket_path(home):
            seen["control_home"] = Path(home).resolve()
            return Path("/fake/gateway.sock")

        monkeypatch.setattr(discovery, "_socket_path", socket_path)
        monkeypatch.setattr(module.socket, "socket", lambda *_a, **_k: FakeSocket())

    result = module.bootstrap(False)

    assert seen["control_home"] == root
    assert seen["request"]["params"] == {
        "profile_id": endpoint.profile_id,
        "instance_id": endpoint.instance_id,
        "purpose": "interactive",
    }
    assert result["profile_id"] == endpoint.profile_id
