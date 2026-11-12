"""REAL-BROWSER regressions for the ``browser:`` real-profile saved-password resolver.

These tests drive a REAL Chrome (``--password-store=basic``) with seeded saved logins over CDP and
run the production resolver (``hermes_agent.tools.browser_real_profile_logins.resolve_login_password``) unmodified
against it — Chrome's own PasswordAutofillAgent fills the field, exactly as in production. They
reproduce bugs an independent Chrome-151 review found that the CDP-substring fake in
``tests/agent/test_vault_backend_browser.py`` cannot: that fake returns canned values keyed on an
expression substring and never runs the real selector / readiness JS, so it is green on the buggy
code. Each test here was confirmed red on the PR-base resolver and green on the fix.

Chrome absence (CI without a browser) skips the whole module cleanly; the host that owns this fix
runs it live. No network: a tiny loopback HTTP server serves each sign-in page.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import threading
import urllib.request
from pathlib import Path

import pytest

from tests.waiting import wait_until

pytestmark = pytest.mark.platforms("linux")

_CHROME = next((p for p in ("/usr/bin/google-chrome", "/usr/bin/google-chrome-stable",
                            shutil.which("google-chrome"), shutil.which("google-chrome-stable"),
                            shutil.which("chromium"), shutil.which("chromium-browser"))
                if p and Path(p).exists()), None)

_requires_chrome = pytest.mark.skipif(
    _CHROME is None, reason="real Chrome/Chromium with --password-store=basic is not installed")

_PBKDF_KEY = hashlib.pbkdf2_hmac("sha1", b"peanuts", b"saltysalt", 1, dklen=16)


def _encrypt_v10(plaintext: str) -> bytes:
    """Encrypt a password exactly as Chrome's Linux ``basic`` password store does: AES-128-CBC with a
    PBKDF2(sha1, 'peanuts', 'saltysalt', 1) key and a 16-space IV, prefixed ``v10``. The seeded blob
    is what the real browser decrypts and autofills — no decryption happens in product code."""
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    data = plaintext.encode("utf-8")
    pad = 16 - (len(data) % 16)
    data += bytes([pad]) * pad
    enc = Cipher(algorithms.AES(_PBKDF_KEY), modes.CBC(b" " * 16)).encryptor()
    return b"v10" + enc.update(data) + enc.finalize()


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _launch_chrome(profile_root: Path) -> tuple[subprocess.Popen, str]:
    """Launch headless Chrome on ``profile_root`` with a CDP port; return (proc, browser ws URL)."""
    port = _free_port()
    proc = subprocess.Popen(
        [_CHROME, f"--user-data-dir={profile_root}", "--password-store=basic", "--headless=new",
         "--no-first-run", "--no-default-browser-check", f"--remote-debugging-port={port}",
         "--disable-gpu", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _devtools_ws():
        """The CDP endpoint once it is up; None until then. A dead process is a hard failure, not a
        thing to keep waiting on."""
        if proc.poll() is not None:
            raise RuntimeError("chrome exited before opening its CDP port")
        try:
            info = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version"))
            return info["webSocketDebuggerUrl"]
        except Exception:
            return None

    try:
        ws_url = wait_until(_devtools_ws, timeout=20, interval=0.1,
                            what="chrome CDP endpoint")
    except Exception:
        # The caller registers proc for cleanup only once we return, so a startup that never opens
        # the CDP port would otherwise leak a live Chrome across failed tests and retries.
        _kill(proc)
        raise
    return proc, ws_url


def _kill(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _login_data_schema(scratch: Path) -> bytes:
    """Launch Chrome once so it writes a genuine ``Login Data`` schema, then return its bytes to seed
    from — the real schema (dozens of columns) rather than a hand-rolled one Chrome would reject."""
    root = scratch / "schema-gen"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    proc, _ = _launch_chrome(root)
    ld = root / "Default" / "Login Data"
    probe = root / "schema-probe.db"
    try:
        # Wait for Chrome to finish initialising the schema: the file existing is not enough (it is
        # created before the ``logins`` table is), and the live DB is write-locked by the running
        # browser, so poll a COPY until its ``logins`` table is queryable — then return that copy's
        # bytes. This replaces a fixed 1 s sleep that raced schema creation on a loaded runner.
        def _schema_bytes():
            if not ld.is_file():
                return None
            try:
                shutil.copy(ld, probe)
                con = sqlite3.connect(probe)
                try:
                    con.execute("SELECT origin_url, username_value, password_value, "
                                "blacklisted_by_user FROM logins LIMIT 0")
                finally:
                    con.close()
                return probe.read_bytes()
            except (OSError, sqlite3.Error):
                return None

        return wait_until(_schema_bytes, timeout=30, interval=0.1,
                          what="chrome Login Data logins table")
    finally:
        _kill(proc)


def _seed_profile(scratch: Path, name: str, schema: bytes, origin: str,
                  creds: list[tuple[str, str, int]]) -> Path:
    """Build a profile dir seeded with ``creds`` = [(username, password, times_used), …] for one
    origin. ``times_used`` lets a 'preferred' account win Chrome's autofill ordering."""
    root = scratch / f"profile-{name}"
    if root.exists():
        shutil.rmtree(root)
    (root / "Default").mkdir(parents=True)
    ld = root / "Default" / "Login Data"
    ld.write_bytes(schema)
    con = sqlite3.connect(ld)
    cols = {r[1] for r in con.execute("PRAGMA table_info(logins)").fetchall()}
    for username, password, used in creds:
        row = {
            "origin_url": origin + "/login", "action_url": origin + "/login",
            "username_element": "username", "username_value": username,
            "password_element": "password", "password_value": _encrypt_v10(password),
            "signon_realm": origin + "/", "date_created": 13300000000000000,
            "blacklisted_by_user": 0, "scheme": 0, "password_type": 0, "times_used": used,
            "date_last_used": 13300000000000000 + used, "date_password_modified": 13300000000000000,
        }
        row = {k: v for k, v in row.items() if k in cols}
        con.execute(f"INSERT INTO logins ({','.join(row)}) VALUES ({','.join('?' * len(row))})",
                    list(row.values()))
    con.commit()
    con.close()
    return root


class _PageServer:
    """A loopback HTTP server that serves one HTML body (the sign-in page) on a fixed port."""

    def __init__(self, port: int, body: bytes):
        outer = self

        class _H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(outer._body)

            def log_message(self, *a):
                pass

        self._body = body
        self._httpd = http.server.HTTPServer(("127.0.0.1", port), _H)

    def __enter__(self):
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *a):
        self._httpd.shutdown()
        return False


# A standard sign-in form with autocomplete hints.
_FORM_STD = (b"<!doctype html><html><body><form action='/login' method='post'>"
             b"<input type='text' name='username' autocomplete='username'>"
             b"<input type='password' name='password' autocomplete='current-password'>"
             b"<button type='submit'>Sign in</button></form></body></html>")
# A plain username field (name=login, NO autocomplete/email/user token) the old name heuristic missed.
_FORM_NAME_LOGIN = (b"<!doctype html><html><body><form action='/login' method='post'>"
                    b"<input type='text' name='login'>"
                    b"<input type='password' name='password'>"
                    b"<button type='submit'>Sign in</button></form></body></html>")
# A form that mounts ~1.8s after load, past the old fixed 1s deadline.
_FORM_LATE = (b"<!doctype html><html><body><div id=app></div><script>setTimeout(function(){"
              b"document.getElementById('app').innerHTML="
              b"\"<form action='/login' method='post'>"
              b"<input type='text' name='username' autocomplete='username'>"
              b"<input type='password' name='password' autocomplete='current-password'>"
              b"<button type='submit'>Sign in</button></form>\";}, 1800);</script>"
              b"</body></html>")
# An autocomplete=username field FOLLOWED by a plain trailing text input (e.g. a 2FA/search box).
# The round-2 resolver chose the LAST texty input before the password, which is the trailing box,
# read its empty value and refused the valid login. The fix prefers the explicit username hint.
_FORM_HINT_THEN_TRAILING = (
    b"<!doctype html><html><body><form action='/login' method='post'>"
    b"<input type='text' name='user' autocomplete='username'>"
    b"<input type='text' name='coupon' autocomplete='off'>"
    b"<input type='password' name='password' autocomplete='current-password'>"
    b"<button type='submit'>Sign in</button></form></body></html>")
# A CSS-hidden honeypot text input placed AFTER the real username field. The round-2 resolver, which
# only skipped type=hidden and picked the last texty input before the password, chose the honeypot
# and read the wrong (empty) username. The fix skips non-visible inputs.
_FORM_HONEYPOT = (
    b"<!doctype html><html><body><form action='/login' method='post'>"
    b"<input type='text' name='username' autocomplete='username'>"
    b"<input type='text' name='email_confirm' style='display:none'>"
    b"<input type='password' name='password' autocomplete='current-password'>"
    b"<button type='submit'>Sign in</button></form></body></html>")


@pytest.fixture(scope="module")
def _schema(tmp_path_factory):
    if _CHROME is None:
        pytest.skip("no chrome")
    return _login_data_schema(tmp_path_factory.mktemp("rpl-schema"))


@pytest.fixture
def live(tmp_path, _schema):
    """Per-test harness: seed a profile, serve a page, launch Chrome, run the real resolver.

    Yields ``run(creds, html, handle_user, *, name)`` returning the resolver's verdict."""
    procs: list[subprocess.Popen] = []

    def run(creds, html, handle_user, *, name):
        from hermes_agent.tools.browser_real_profile_logins import resolve_login_password

        port = _free_port()
        origin = f"http://127.0.0.1:{port}"
        root = _seed_profile(tmp_path, name, _schema, origin, creds)
        with _PageServer(port, html):
            proc, ws_url = _launch_chrome(root)
            procs.append(proc)
            return resolve_login_password(ws_url, origin + "/login", origin, handle_user)

    try:
        yield run
    finally:
        for p in procs:
            _kill(p)


# ── finding 1 (security): blank-username login must not return a named account's password ─────────

@_requires_chrome
def test_blank_username_does_not_leak_a_named_accounts_password(live):
    """The origin has a blank-username cred AND a preferred 'alice' cred. Resolving the BLANK login,
    Chrome autofills ALICE's password (confirmed: username='alice@…'). The PR base only matched the
    account when the handle username was truthy, so it returned ALICE's password for the blank entry.
    The fix treats an empty saved username as a real identity, so the named autofill is refused."""
    got = live([("", "BLANK-PW", 0), ("alice@example.com", "ALICE-PW", 50)],
               _FORM_STD, "", name="blank-leak")
    assert got is None, f"blank-username resolve leaked a named account's password: {got!r}"


@_requires_chrome
def test_blank_username_resolves_when_it_is_the_only_account(live):
    """Control for finding 1: a blank-username login that is the ONLY account on its origin still
    resolves — the fix refuses a MISMATCHED identity, it does not refuse blank handles wholesale."""
    got = live([("", "SOLO-BLANK-PW", 10)], _FORM_STD, "", name="blank-solo")
    assert got == "SOLO-BLANK-PW"


# ── finding 2: a plain <input name=login> username field (no name heuristic match) resolves ───────

@_requires_chrome
def test_name_login_text_field_resolves(live):
    """A form whose username field is a plain ``<input type=text name=login>`` — matching none of the
    old autocomplete=username / type=email / name*=user|email selectors. The base read the username
    back empty and refused; the fix discovers the username tied to the autofilled password field."""
    got = live([("jane@example.com", "NAME-LOGIN-PW", 10)], _FORM_NAME_LOGIN,
               "jane@example.com", name="name-login")
    assert got == "NAME-LOGIN-PW"


# ── round-3 finding (1): the explicit username field is chosen over a trailing/honeypot control ───

@_requires_chrome
def test_username_hint_preferred_over_trailing_text_field(live):
    """A form with an explicit ``autocomplete=username`` field FOLLOWED by a plain trailing text
    input. The round-2 resolver picked the LAST texty input before the password (the trailing box),
    read its empty value and refused the valid login. The fix prefers the explicit username hint, so
    the real username is read and the saved password resolves."""
    got = live([("jane@example.com", "HINT-PW", 10)], _FORM_HINT_THEN_TRAILING,
               "jane@example.com", name="hint-trailing")
    assert got == "HINT-PW"


@_requires_chrome
def test_css_hidden_honeypot_text_field_is_skipped(live):
    """A CSS-hidden honeypot text input sits AFTER the real username field. The round-2 resolver only
    skipped ``type=hidden`` and chose the honeypot (last texty input before the password), reading
    the wrong (empty) username and refusing. The fix skips non-visible inputs, so the real username
    resolves."""
    got = live([("jane@example.com", "HONEYPOT-PW", 10)], _FORM_HONEYPOT,
               "jane@example.com", name="honeypot")
    assert got == "HONEYPOT-PW"


# ── finding 3: a form that renders after the old 1s deadline still resolves ───────────────────────

@_requires_chrome
def test_late_rendered_form_resolves(live):
    """The sign-in form mounts ~1.8s after navigation. The base slept a fixed 1s, inspected once and
    returned None; the fix polls for a usable form AND autofill completion within a bounded wait."""
    got = live([("jane@example.com", "LATE-PW", 10)], _FORM_LATE,
               "jane@example.com", name="late-form")
    assert got == "LATE-PW"


# ── finding 6: a saved username with surrounding whitespace matches (no silent strip) ─────────────

@_requires_chrome
def test_whitespace_username_resolves_through_the_backend(tmp_path, _schema, monkeypatch):
    """A saved username ``' bob '`` (leading/trailing spaces), driven through the WHOLE backend path
    (``list_items`` → handle → ``resolve_password``) against real Chrome. The base ``list_items``
    stripped the username into the handle while ``resolve`` matched Chrome's raw autofill (``' bob '``)
    exactly, so the handle's ``'bob'`` never matched and resolve returned None. The fix carries the
    raw saved username into the handle, so the account match succeeds. Only the backend path exercises
    the strip — calling the resolver directly would not reproduce it."""
    from hermes_agent.agent.vault_backends import enabled_backends

    port = _free_port()
    origin = f"http://127.0.0.1:{port}"
    root = _seed_profile(tmp_path, "ws-user", _schema, origin, [(" bob ", "BOB-PW", 10)])

    # Consent on; the snapshot copy dir IS the seeded Chrome profile (Default/Login Data present).
    monkeypatch.setattr("hermes_agent.agent.vault_backends.base._browser_profile_enabled", lambda: True)
    monkeypatch.setattr("hermes_cli.browser_connect.detect_default_chromium", lambda: "chrome")
    monkeypatch.setattr("hermes_cli.browser_connect.real_profile_copy_dir", lambda browser: str(root))

    with _PageServer(port, _FORM_STD):
        proc, ws_url = _launch_chrome(root)
        try:
            # The lifecycle call that would (re)build the snapshot is a no-op here: the seeded profile
            # already has Login Data, and the live browser is the one we launched.
            monkeypatch.setattr("hermes_agent.tools.browser_session_real_profile.real_profile_cdp",
                                lambda: (ws_url, None))
            backend = next(b for b in enabled_backends() if b.name == "browser")
            item = next(i for i in backend.list_items() if i.origin == origin)
            # The identifier is what the Desktop key button writes into the page's username field, so
            # it must carry the RAW saved username (' bob '), not the stripped 'bob' (a different
            # account). The label stays stripped for display.
            assert item.identifier == " bob ", f"identifier lost its spaces: {item.identifier!r}"
            got = backend.resolve_password(item.id)
        finally:
            _kill(proc)
    assert got == "BOB-PW"


# ── round-1 guard still holds: a redirect off the saved origin must not leak another site's pw ────

@_requires_chrome
def test_redirect_off_origin_still_refused_live(live, tmp_path, _schema):
    """Round-1 origin guard, live: the saved URL 302s to a DIFFERENT origin that has its own saved
    login; Chrome autofills THAT site's password. The resolver must refuse because the final page's
    origin is not the handle's saved origin."""
    from hermes_agent.tools.browser_real_profile_logins import resolve_login_password

    victim_port = _free_port()
    victim_origin = f"http://127.0.0.1:{victim_port}"
    saved_port = _free_port()
    saved_origin = f"http://127.0.0.1:{saved_port}"

    # Seed the VICTIM origin with a saved login so Chrome autofills it after the redirect lands there.
    victim_root = _seed_profile(tmp_path, "redir-victim", _schema, victim_origin,
                                [("victim@example.com", "VICTIM-PW", 10)])

    victim_body = _FORM_STD

    class _RedirH(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", victim_origin + "/login")
            self.end_headers()

        def log_message(self, *a):
            pass

    redir = http.server.HTTPServer(("127.0.0.1", saved_port), _RedirH)
    threading.Thread(target=redir.serve_forever, daemon=True).start()
    proc = None
    try:
        with _PageServer(victim_port, victim_body):
            proc, ws_url = _launch_chrome(victim_root)
            # Handle names the SAVED origin, but navigating it redirects to the victim origin.
            got = resolve_login_password(ws_url, saved_origin + "/login", saved_origin,
                                         "victim@example.com")
        assert got is None, f"redirect off saved origin leaked another site's password: {got!r}"
    finally:
        if proc is not None:
            _kill(proc)
        redir.shutdown()
