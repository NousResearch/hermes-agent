"""Ink's private pipe bootstrap; the child is a client, never an agent owner.

``--start`` ensures the owner (first launch). A reconnect discovers only, except ``--recover``
(a bounded slot, ``gatewayClient.ts`` ``OWNER_RECOVERY_LIMIT`` per window) re-ensures whenever the
owner is not ready: it crashed, ended by its idle exit, was stopped with ``hermes gateway stop``
while this TUI was open, or an older Hermes's gateway answers (ensure replaces that one). Main's TUI
ran its own sidecar and never depended on an external daemon, so an open TUI comes back on its own.
"""
import contextlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))



def failure_message(receipt) -> str:
    """What the TUI shows when there is no owner to attach: the bounded verdict, then its detail
    (a refusal's remedy, or a start that died's own redacted output), each line still a verdict."""
    head = f"gateway {receipt.state}: {receipt.reason_code or 'not ready'}"
    detail = getattr(receipt, "detail", None)
    if not detail:
        return head
    if receipt.reason_code == "runtime_exited":
        return f"{head}\n{detail}"
    lines = [line.strip() for line in str(detail).splitlines() if line.strip()]
    return "\n".join([head, *(f"gateway {receipt.state}: {line}" for line in lines)])


def bootstrap(start: bool, recover: bool = False) -> dict:
    from hermes_constants import get_hermes_home
    from hermes_cli.gateway_runtime import control_home_for, discover_gateway_endpoint, ensure_gateway_runtime
    from hermes_cli.gateway_runtime_discovery import connect_private

    home = get_hermes_home().resolve()
    # Launch policy travels in session.create, not into daemon-wide defaults. That includes the
    # Kanban board `hermes --tui` pins at launch (_pin_kanban_board_env): a daemon started from
    # here would otherwise resolve every later chat's, Desktop window's and cron job's board to
    # this terminal's launch-time board instead of the profile's `kanban/current`.
    for key in ("HERMES_MODEL", "HERMES_INFERENCE_MODEL", "HERMES_TUI_PROVIDER",
                "HERMES_INFERENCE_PROVIDER", "HERMES_TUI_TOOLSETS", "HERMES_TUI_SKILLS",
                "HERMES_CWD", "TERMINAL_CWD", "HERMES_YOLO", "HERMES_ACCEPT_HOOKS",
                "HERMES_TUI_CHECKPOINTS", "HERMES_TUI_PASS_SESSION_ID", "HERMES_KANBAN_BOARD"):
        os.environ.pop(key, None)
    receipt = (ensure_gateway_runtime(home, timeout=30, idle_exit=True) if start
               else discover_gateway_endpoint(home, timeout=5))
    if not start and recover and receipt.state != "ready":
        receipt = ensure_gateway_runtime(home, timeout=30, idle_exit=True)
    if receipt.state != "ready" or receipt.endpoint is None:
        raise RuntimeError(failure_message(receipt))
    endpoint = receipt.endpoint
    # A profile served by the default multiplexer has no socket of its own; the host mints its ticket.
    control = control_home_for(home, endpoint)
    request = json.dumps({"protocol": 1, "id": 1, "verb": "session-ticket", "params": {
        "profile_id": endpoint.profile_id, "instance_id": endpoint.instance_id,
        "purpose": "interactive"}}).encode() + b"\n"
    if os.name == "nt":
        from gateway.runtime_bootstrap_windows import query_runtime_control
        raw = query_runtime_control(control, request, 5)
    else:
        with connect_private(control, 5) as peer:
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
    from hermes_cli.gateway_client import gateway_ws_target
    url, protocols = gateway_ws_target(endpoint, grant["ticket"])
    return {"url": url, "protocols": protocols,
            "profile_id": endpoint.profile_id, "instance_id": endpoint.instance_id}


if __name__ == "__main__":
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = bootstrap("--start" in sys.argv, recover="--recover" in sys.argv)
        print(json.dumps(result))
    # Discovery/ticket failures: missing modules, socket/pipe I/O (incl. timeouts), DiscoveryError and
    # bad JSON (ValueError), rejected grants (RuntimeError), malformed reply shapes. Anything else
    # still exits 1, with a traceback.
    except (ImportError, OSError, RuntimeError, ValueError, LookupError, AttributeError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
