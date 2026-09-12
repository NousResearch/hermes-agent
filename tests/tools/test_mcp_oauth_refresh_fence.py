"""Acceptance tests for the cross-process refresh fence (#94846).

These are deliberately NOT unit tests of the lock primitive. The defect class
they close is an interleaving between two OS processes that each run the real
SDK auth flow against a provider issuing SINGLE-USE refresh tokens:

    A: get_tokens() -> R1       (narrow token-store lock taken and RELEASED)
    B: get_tokens() -> R1       (narrow token-store lock taken and RELEASED)
    A: POST R1                  -> 200, receives R2
    B: POST R1                  -> 400, credential already burned
    B: clear_tokens()           -> user logged out of a working session

Every step respects ``_token_store_lock``, so only a fence spanning
read -> POST -> persist can prevent it. A single-process test cannot observe
this: ``threading.RLock`` alone would make it pass.

The authorization server here records every refresh_token it is asked to
redeem and rejects a second redemption of the same one, exactly like the real
providers (Notion, Supabase, Linear) that motivated the fix.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tools.mcp_oauth import (
    _REFRESH_FENCE_TIMEOUT_SECONDS,
    RefreshFenceTimeout,
    _refresh_fence,
    _token_store_lock,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# A single-use-refresh-token authorization server, in-process, stdlib only.
# ---------------------------------------------------------------------------

_AS_SRC = '''
import json, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs

# Every refresh_token ever presented, in order. A provider with single-use
# refresh tokens rejects the second presentation of the same value.
redeemed = []
lock = threading.Lock()
generation = [1]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
        form = parse_qs(body)
        presented = (form.get("refresh_token") or [""])[0]

        with lock:
            already_used = presented in redeemed
            redeemed.append(presented)
            if already_used:
                payload = json.dumps({"error": "invalid_grant"}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            generation[0] += 1
            n = generation[0]

        # Deliberately slow: widens the window between read and persist so an
        # unfenced peer would reliably POST the same token.
        import time as _t
        _t.sleep(0.35)

        payload = json.dumps({
            "access_token": f"at-{n}",
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": f"rt-{n}",
            "scope": "read",
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        # /__audit returns what was presented, so the test can assert that R1
        # was redeemed exactly once.
        payload = json.dumps(redeemed).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


srv = HTTPServer(("127.0.0.1", 0), Handler)
print(srv.server_port, flush=True)
srv.serve_forever()
'''


# A refresher process: runs the real provider refresh path end to end.
_REFRESHER_SRC = '''
import asyncio, json, os, sys
sys.path.insert(0, sys.argv[1])
os.environ["HERMES_HOME"] = sys.argv[2]
os.environ.setdefault("HERMES_NONINTERACTIVE", "0")
token_endpoint = sys.argv[3]
label = sys.argv[4]

import httpx
from types import SimpleNamespace
from tools.mcp_oauth import HermesTokenStorage
from tools.mcp_oauth_provider import HermesProviderMixin


class Ctx:
    """Minimal stand-in for the SDK's auth context.

    Only the members the fenced refresh path touches: the fence must work
    without dragging the whole SDK provider into a subprocess.
    """

    def __init__(self, storage):
        self.storage = storage
        self.current_tokens = None
        self.client_info = SimpleNamespace(client_id="cid", client_secret=None)
        self.oauth_metadata = SimpleNamespace(token_endpoint=token_endpoint)
        self.cleared = False

    def clear_tokens(self):
        self.cleared = True
        self.current_tokens = None

    def update_token_expiry(self, t):
        pass


class Provider(HermesProviderMixin):
    """Drives the real fence + real _handle_refresh_response over real HTTP."""

    def __init__(self, storage):
        self.context = Ctx(storage)
        self._initialized = True

    def _coerce_client_secret_post(self):
        pass

    def _prepare_token_request(self, req):
        return req

    async def _refresh_token_request(self):
        # What the SDK would build: POST the refresh token we believe we own.
        tok = self.context.current_tokens
        return httpx.Request(
            "POST",
            token_endpoint,
            data={
                "grant_type": "refresh_token",
                "refresh_token": tok.refresh_token,
                "client_id": "cid",
            },
        )


async def main():
    storage = HermesTokenStorage("srv")
    p = Provider(storage)
    p.context.current_tokens = await storage.get_tokens()

    # Mirror the SDK flow: fence + adopt, build request, POST, handle response.
    try:
        p._hermes_acquire_refresh_fence()
    except Exception as exc:
        print(json.dumps({"label": label, "outcome": "fence_denied", "error": type(exc).__name__}), flush=True)
        return

    try:
        await p._hermes_adopt_tokens_from_disk()
        presented = p.context.current_tokens.refresh_token
        req = await p._refresh_token_request()
        async with httpx.AsyncClient() as client:
            resp = await client.send(req)
        ok = await p._hermes_handle_refresh_response(resp)
    finally:
        p._hermes_release_refresh_fence()

    final = await storage.get_tokens()
    print(json.dumps({
        "label": label,
        "outcome": "ok" if ok else "failed",
        "presented": presented,
        "status": resp.status_code,
        "cleared": p.context.cleared,
        "final_refresh": getattr(final, "refresh_token", None),
    }), flush=True)


asyncio.run(main())
'''


def _start_auth_server(tmp_path):
    src = tmp_path / "as.py"
    src.write_text(_AS_SRC, encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(src)], stdout=subprocess.PIPE, text=True
    )
    port = proc.stdout.readline().strip()
    if not port.isdigit():  # pragma: no cover
        proc.kill()
        pytest.skip(f"authorization server did not start (got {port!r})")
    return proc, int(port)


def _seed_tokens(home: Path, refresh_token: str) -> Path:
    d = home / "mcp-tokens"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "srv.json"
    path.write_text(
        json.dumps(
            {
                "access_token": "at-1",
                "token_type": "Bearer",
                "expires_in": 3600,
                "refresh_token": refresh_token,
                "scope": "read",
            }
        ),
        encoding="utf-8",
    )
    return path


def _audit(port: int):
    import urllib.request

    with urllib.request.urlopen(f"http://127.0.0.1:{port}/__audit", timeout=10) as r:
        return json.loads(r.read().decode())


# ---------------------------------------------------------------------------
# The acceptance test the review asked for.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not sys.executable, reason="needs a real interpreter")
def test_two_processes_redeem_r1_exactly_once_and_converge_on_r2(tmp_path):
    """THE acceptance criterion from the #94846 review.

    Two OS processes refresh concurrently against a single-use-refresh-token
    AS. Required outcome:

      * ``R1`` is presented to the token endpoint EXACTLY ONCE
      * both processes end up authenticated
      * both converge on the same replacement token
      * neither clears the session

    Without the fence the loser POSTs ``R1`` a second time, gets 400, and --
    before the recovery half existed -- wiped a live session.
    """
    as_proc, port = _start_auth_server(tmp_path)
    token_endpoint = f"http://127.0.0.1:{port}/token"
    try:
        _seed_tokens(tmp_path, "rt-1")
        script = tmp_path / "refresher.py"
        script.write_text(_REFRESHER_SRC, encoding="utf-8")

        procs = [
            subprocess.Popen(
                [
                    sys.executable, str(script), str(REPO_ROOT), str(tmp_path),
                    token_endpoint, label,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for label in ("A", "B")
        ]
        outs = []
        for p in procs:
            stdout, stderr = p.communicate(timeout=120)
            if p.returncode != 0:  # pragma: no cover
                pytest.fail(f"refresher crashed ({p.returncode}):\n{stderr}")
            line = (stdout or "").strip().splitlines()
            if not line:  # pragma: no cover
                pytest.fail(f"refresher produced no result:\n{stderr}")
            outs.append(json.loads(line[-1]))

        presented = _audit(port)

        # 1. R1 redeemed exactly once -- the core invariant.
        assert presented.count("rt-1") == 1, (
            f"R1 was presented {presented.count('rt-1')}x to the token endpoint; "
            f"the fence did not serialize the consuming POST (audit={presented})"
        )

        # 2. Nobody logged the user out.
        assert [o["cleared"] for o in outs] == [False, False], outs

        # 3. Both processes succeeded...
        assert {o["outcome"] for o in outs} == {"ok"}, outs

        # 4. NO token was redeemed twice -- not just R1. Each process
        #    presented a distinct, live credential.
        assert len(set(presented)) == len(presented), (
            f"a refresh token was redeemed more than once: {presented}"
        )
        assert sorted(o["presented"] for o in outs) != ["rt-1", "rt-1"], outs

        # 5. Disk converges on the newest issued token, so the next refresh
        #    starts from a live credential rather than a burned one.
        #    (This harness refreshes unconditionally, so the process that
        #    adopted a peer's token legitimately rotates it once more; the
        #    invariant is the final state, not a mid-flight snapshot.)
        final_on_disk = json.loads(
            (tmp_path / "mcp-tokens" / "srv.json").read_text(encoding="utf-8")
        )["refresh_token"]
        newest = f"rt-{len(presented) + 1}"
        assert final_on_disk == newest, (
            f"disk holds {final_on_disk!r}, newest issued is {newest!r}: a "
            f"process overwrote a peer's fresher token (audit={presented})"
        )
        assert final_on_disk != "rt-1", "token never rotated; test proved nothing"
    finally:
        as_proc.kill()
        for p in procs:
            if p.poll() is None:  # pragma: no cover
                p.kill()


# ---------------------------------------------------------------------------
# Fence properties the acceptance test relies on.
# ---------------------------------------------------------------------------


def test_fence_fails_closed_when_a_peer_holds_it(tmp_path):
    """A busy fence must RAISE, never fall through to an unowned POST.

    ``_token_store_lock`` degrades to in-process locking on timeout -- correct
    there, fatal here: proceeding without ownership is the race itself.
    """
    target = _seed_tokens(tmp_path, "rt-1")

    holder_src = '''
import os, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
os.environ["HERMES_HOME"] = sys.argv[2]
from tools.mcp_oauth import _refresh_fence
with _refresh_fence(Path(sys.argv[3])):
    print("HELD", flush=True)
    time.sleep(float(sys.argv[4]))
'''
    holder = tmp_path / "holder.py"
    holder.write_text(holder_src, encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(holder), str(REPO_ROOT), str(tmp_path), str(target), "6"],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        if proc.stdout.readline().strip() != "HELD":  # pragma: no cover
            pytest.skip("peer could not take the fence")

        start = time.monotonic()
        with pytest.raises(RefreshFenceTimeout):
            with _refresh_fence(target, timeout=1.0):  # pragma: no cover
                pytest.fail("fence was granted while a peer held it")
        waited = time.monotonic() - start
    finally:
        proc.kill()
        proc.wait(timeout=15)

    # It waited for its bound instead of failing instantly or hanging forever.
    assert 0.8 < waited < 5.0, waited


def test_fence_is_independent_of_the_token_store_lock(tmp_path):
    """Holding the fence must not block get_tokens/set_tokens.

    flock/msvcrt locks are per-file-descriptor, so reusing one path for both
    scopes self-deadlocks on Windows and silently no-ops on POSIX. The fence
    therefore uses a ``.refresh.lock`` sibling -- this test pins that apart.
    """
    target = _seed_tokens(tmp_path, "rt-1")

    with _refresh_fence(target):
        start = time.monotonic()
        with _token_store_lock(target):
            pass
        assert time.monotonic() - start < 1.0, "fence blocked the token-store lock"

    assert (target.parent / "srv.json.refresh.lock").exists()
    assert target.with_suffix(".json.lock").name != "srv.json.refresh.lock"


def test_fence_releases_on_exception(tmp_path):
    """An aborted refresh must not strand ownership for the process lifetime."""
    target = _seed_tokens(tmp_path, "rt-1")

    with pytest.raises(ValueError):
        with _refresh_fence(target):
            raise ValueError("boom")

    start = time.monotonic()
    with _refresh_fence(target, timeout=2.0):
        pass
    assert time.monotonic() - start < 1.0, "fence leaked after an exception"


def test_fence_default_timeout_outlasts_a_network_round_trip():
    """The fence spans a POST, so its bound must exceed the narrow lock's 10s.

    Pins the intent: a healthy-but-slow token endpoint must not trip the
    fail-closed path and force a spurious reauth.
    """
    assert _REFRESH_FENCE_TIMEOUT_SECONDS >= 30.0
