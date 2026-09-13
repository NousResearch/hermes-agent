"""Ink's private pipe bootstrap; the child is a client, never an agent owner."""
import contextlib
import json
import os
from pathlib import Path
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def bootstrap(start: bool) -> dict:
    from hermes_constants import get_hermes_home
    from hermes_cli.gateway_runtime import discover_gateway_endpoint, ensure_gateway_runtime
    from hermes_cli.gateway_runtime_discovery import _socket_path

    home = get_hermes_home().resolve()
    # Launch policy travels in session.create, not into daemon-wide defaults.
    for key in ("HERMES_MODEL", "HERMES_INFERENCE_MODEL", "HERMES_TUI_PROVIDER",
                "HERMES_INFERENCE_PROVIDER", "HERMES_TUI_TOOLSETS", "HERMES_TUI_SKILLS",
                "HERMES_CWD", "TERMINAL_CWD", "HERMES_YOLO", "HERMES_ACCEPT_HOOKS"):
        os.environ.pop(key, None)
    receipt = (ensure_gateway_runtime(home, timeout=30) if start
               else discover_gateway_endpoint(home, timeout=5))
    if receipt.state != "ready" or receipt.endpoint is None:
        raise RuntimeError(f"gateway {receipt.state}: {receipt.reason_code or 'not ready'}")
    endpoint = receipt.endpoint
    request = json.dumps({"protocol": 1, "id": 1, "verb": "session-ticket", "params": {
        "profile_id": endpoint.profile_id, "instance_id": endpoint.instance_id,
        "purpose": "interactive"}}).encode() + b"\n"
    if os.name == "nt":
        from gateway.runtime_bootstrap_windows import query_runtime_control
        raw = query_runtime_control(home, request, 5)
    else:
        with socket.socket(socket.AF_UNIX) as peer:
            peer.settimeout(5)
            peer.connect(str(_socket_path(home)))
            peer.sendall(request)
            with peer.makefile("rb") as stream:
                raw = stream.readline(65537)
    result = json.loads(raw)
    grant = result.get("result", {})
    if (result.get("ok") is not True or result.get("id") != 1
            or grant.get("instance_id") != endpoint.instance_id
            or grant.get("profile_id") != endpoint.profile_id
            or not isinstance(grant.get("ticket"), str)):
        raise RuntimeError("gateway private bootstrap rejected")
    return {"url": endpoint.api_origin.replace("http", "ws", 1) + "/api/ws",
            "protocols": ["hermes-gateway-v1", "hermes-gateway-ticket." + grant["ticket"]],
            "profile_id": endpoint.profile_id, "instance_id": endpoint.instance_id}


if __name__ == "__main__":
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = bootstrap("--start" in sys.argv)
        print(json.dumps(result))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
