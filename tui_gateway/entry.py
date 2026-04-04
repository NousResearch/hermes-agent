import json
import signal
import sys

from tui_gateway.server import handle_request, resolve_skin

signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def _write(obj: dict):
    try:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()
    except BrokenPipeError:
        sys.exit(0)


def main():
    _write({
        "jsonrpc": "2.0",
        "method": "event",
        "params": {"type": "gateway.ready", "payload": {"skin": resolve_skin()}},
    })

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            _write({"jsonrpc": "2.0", "error": {"code": -32700, "message": "parse error"}, "id": None})
            continue

        resp = handle_request(req)
        if resp is not None:
            _write(resp)


if __name__ == "__main__":
    main()
