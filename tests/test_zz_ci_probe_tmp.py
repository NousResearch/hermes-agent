"""TEMPORARY CI probe — reveals what shard runners import (delete before merge).

Four shard-5 tests fail on CI with evidence contradicting the checked-out
tree while passing locally on the identical commit. This always-passing
probe prints the disk hash and the imported-object reality so the next CI
run tells us which artifact diverges.
"""
import ast
import hashlib
import inspect
import sys
import textwrap
from pathlib import Path


def test_zz_ci_probe():
    root = Path(__file__).resolve().parents[1]
    for rel in ("gateway/run.py", "tui_gateway/server.py"):
        digest = hashlib.sha256((root / rel).read_bytes()).hexdigest()[:12]
        print(f"PROBE disk {rel} sha={digest}", file=sys.stderr)

    from gateway.run import GatewayRunner
    src_file = inspect.getsourcefile(GatewayRunner._send_voice_reply)
    print(f"PROBE import gateway.run -> {src_file}", file=sys.stderr)
    src = textwrap.dedent(inspect.getsource(GatewayRunner._send_voice_reply))
    func = ast.parse(src).body[0]
    has = any(
        isinstance(node, ast.Try) and node.finalbody
        and ("unlink" in ast.dump(node.finalbody[0]) or "remove" in ast.dump(node.finalbody[0]))
        for node in ast.walk(func)
    )
    print(f"PROBE _send_voice_reply finally-unlink={has} len={len(src)}", file=sys.stderr)

    from tui_gateway import server
    print(f"PROBE import tui_gateway.server -> {inspect.getsourcefile(server)}", file=sys.stderr)
    row = server._session_live_item("probe-sid", {"session_key": "k"}, "")
    print(f"PROBE live_item keys={sorted(row.keys())}", file=sys.stderr)
    assert True
