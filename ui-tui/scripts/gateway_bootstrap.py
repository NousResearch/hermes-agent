"""Ink's private pipe bootstrap; the child is a client, never an agent owner.

``--start`` ensures the owner (first launch). A reconnect discovers only, except ``--recover``
re-ensures an owner that is gone without anyone asking: it CRASHED (its retained runtime record
still claims it live) or it ended ITSELF by the auto-started idle exit (``gateway/run_idle_exit.py``,
"the next client starts a fresh gateway" — this reconnecting client is that next client). An
explicitly stopped gateway (``hermes gateway stop`` / a planned stop: ``stopped`` without the idle
exit's reason, or a durable ``desired_state: stopped``) is never resurrected by a reconnect.
"""
import contextlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ``gateway_state`` values only a live owner publishes; ``hermes gateway stop`` and planned stops
# persist ``stopped``, so a dead process still claiming one of these did not stop cleanly.
_LIVE_CLAIMS = frozenset({"running", "starting", "draining", "degraded"})


def owner_crashed(home: Path) -> bool:
    """After discovery answered ``absent``: did the owner serving *home* end without anyone asking
    (a crash, or its own idle exit while this client was away — suspended, asleep, disconnected)?"""
    from gateway.run_idle_exit import IDLE_EXIT_STOP_PREFIX
    from gateway.status import read_runtime_status
    from hermes_cli.gateway_runtime_multiplex import implied_host_root, multiplexer_serves_home
    owner = multiplexer_serves_home(home) or implied_host_root(home) or home
    record = read_runtime_status(owner / "gateway_state.json")
    if not isinstance(record, dict) or record.get("desired_state") == "stopped":
        return False
    # The owner's own idle exit (gateway/run_idle_exit.py) persists ``stopped`` with this reason;
    # no operator asked for it, and "the next client starts a fresh gateway" is this client.
    idle_exited = (record.get("gateway_state") == "stopped"
                   and str(record.get("exit_reason") or "").startswith(IDLE_EXIT_STOP_PREFIX))
    return idle_exited or record.get("gateway_state") in _LIVE_CLAIMS


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
    if not start and recover and receipt.state == "absent" and owner_crashed(home):
        receipt = ensure_gateway_runtime(home, timeout=30, idle_exit=True)
    if receipt.state != "ready" or receipt.endpoint is None:
        detail = f"\n{receipt.detail}" if receipt.reason_code == "runtime_exited" and receipt.detail else ""
        raise RuntimeError(f"gateway {receipt.state}: {receipt.reason_code or 'not ready'}{detail}")
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
