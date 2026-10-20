"""Ink's gateway bootstrap: a thin wrapper over ``hermes_cli.gateway_runtime_attach.attach_gateway``,
the attach ``hermes gateway attach`` runs. The child is a client, never an agent owner.

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

# The Python root, not this file's location: the TUI product ships this helper in its own
# dist/scripts/ (a separate Nix store path), so parents[2] is only right in a source tree.
# gatewayClient.ts passes the root it put on PYTHONPATH.
def python_root() -> Path:
    return Path(os.environ.get("HERMES_PYTHON_SRC_ROOT") or Path(__file__).resolve().parents[2])


sys.path.insert(0, str(python_root()))

# The terminal verdict's next step: the client stops retrying on it, so the user needs one. A
# verdict line like the rest, so the TUI echoes it inline (userMessages.ts STDERR_PROBLEM_RE).
BOOTSTRAP_REMEDY = "gateway unavailable: run `hermes pm repair`, then restart the TUI"


def bootstrap(start: bool, recover: bool = False) -> dict:
    from pm.environments import NoCommittedEnvironment, activate_dependencies

    # Ink runs this on HERMES_PYTHON with PYTHONPATH=<root> only. On a source install that is PM's
    # store Python, which owns no packages, and the no-owner discovery fallback already imports one
    # (gateway.status -> utils -> hermes_yaml -> ruamel): boot the committed generation first, as
    # hermes_bootstrap does for every other entry point.
    try:
        activate_dependencies(python_root())
    except NoCommittedEnvironment as exc:
        # Only a missing generation is terminal. An I/O error reading the install state (OSError)
        # or a damaged record (RuntimeError) reaches the ordinary failure path below, which the
        # client retries: a one-off read error must not end the TUI's gateway for its lifetime.
        raise ImportError(f"dependency environment unavailable: {exc}") from exc
    from hermes_constants import get_hermes_home
    from hermes_cli.gateway_client import gateway_ws_target
    from hermes_cli.gateway_runtime_attach import attach_gateway, failure_message, scrub_client_launch_env

    scrub_client_launch_env()
    # The same attach `hermes gateway attach` runs; Ink keeps its own 30 s start budget.
    attached = attach_gateway(get_hermes_home().resolve(), start=start, recover=recover, timeout=30)
    if attached.ticket is None:
        raise RuntimeError(failure_message(attached.receipt))
    endpoint = attached.receipt.endpoint
    url, protocols = gateway_ws_target(endpoint, attached.ticket)
    return {"url": url, "protocols": protocols,
            "profile_id": endpoint.profile_id, "instance_id": endpoint.instance_id}


if __name__ == "__main__":
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = bootstrap("--start" in sys.argv, recover="--recover" in sys.argv)
        print(json.dumps(result))
    # No owner to attach (RuntimeError, the verdict), missing modules, socket/pipe I/O (incl.
    # timeouts), DiscoveryError, bad JSON and rejected grants (ValueError), malformed reply shapes.
    # Anything else still exits 1, with a traceback.
    except ImportError as exc:
        # This interpreter cannot load Hermes at all: no reconnect fixes that, so name it
        # (gatewayClient.ts stops retrying on this verdict).
        print(f"gateway unavailable: bootstrap_dependencies\n{exc}\n{BOOTSTRAP_REMEDY}", file=sys.stderr)
        raise SystemExit(1)
    except (OSError, RuntimeError, ValueError, LookupError, AttributeError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
