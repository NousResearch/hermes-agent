"""Region B unit tests — egress interposer + parent-side plumbing.

Fresh-interpreter harness: the child runs ``[sys.executable, -c, body]`` with
the guard dir prepended to PYTHONPATH (native C:/ paths on Windows — F5.3)
and the per-spawn env flags, so ``sitecustomize`` installs the interposer
before any model code. Markers are read from the child's captured stderr.
"""

import json
import os
import subprocess
import sys

import pytest

from tools.browser_exec_egress_guard import (
    _egress_guard_dir,
    _install_egress_guard,
    _parse_guard_markers,
    _policy_snapshot,
    _strip_guard_markers,
    _verify_or_regenerate,
)

NONCE = "testnonce123"


@pytest.fixture
def guard_env(tmp_path, monkeypatch):
    """HERMES_HOME sandbox + generated guard dir + child env dict."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.delenv("BU_CDP_WS", raising=False)
    monkeypatch.delenv("BU_CDP_URL", raising=False)
    gd = _egress_guard_dir()
    _verify_or_regenerate(gd)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(gd) + os.pathsep + env.get("PYTHONPATH", "")
    env["HERMES_BROWSER_EXEC_EGRESS_GUARD"] = "1"
    env["HERMES_BROWSER_EXEC_EGRESS_GUARD_NONCE"] = NONCE
    env["HERMES_BROWSER_EXEC_EGRESS_POLICY"] = json.dumps(_policy_snapshot(nonce=NONCE))
    return gd, env


def run_child(body, env, timeout=90):
    """Run model-code body in a fresh interpreter under the guard env."""
    p = subprocess.run(
        [sys.executable, "-c", body],
        capture_output=True, text=True, timeout=timeout, env=env,
    )
    return p.returncode, p.stdout, p.stderr


class TestSitecustomizeInstall:
    def test_sitecustomize_auto_installs(self, guard_env):
        _, env = guard_env
        body = (
            "import socket\n"
            "print(socket.socket.__name__)\n"
        )
        rc, out, err = run_child(body, env)
        assert rc == 0
        assert out.strip() == "_GuardedSocket"
        assert f"__HERMES_EGRESS_GUARD__:installed:{NONCE}" in err

    def test_construction_and_http_client_work(self, guard_env):
        """F1 regression: wrapper machinery survives; http.client round-trips."""
        _, env = guard_env
        body = (
            "import socket, http.server, threading\n"
            "class H(http.server.BaseHTTPRequestHandler):\n"
            "    def do_GET(self):\n"
            "        self.send_response(200); self.send_header('Content-Length','2'); self.end_headers()\n"
            "        self.wfile.write(b'ok')\n"
            "    def log_message(self, *a): pass\n"
            "srv = http.server.HTTPServer(('127.0.0.1', 0), H)\n"
            "t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()\n"
            "port = srv.server_address[1]\n"
            "import http.client\n"
            "c = http.client.HTTPConnection('127.0.0.1', port, timeout=5)\n"
            "c.request('GET', '/x')\n"
            "r = c.getresponse()\n"
            "print('HTTP', r.status, r.read().decode())\n"
            "s = socket.socket()\n"
            "print('makefile', hasattr(s, 'makefile'), 'accept', hasattr(s, 'accept'),\n"
            "      'dup', hasattr(s, 'dup'), 'sendfile', hasattr(s, 'sendfile'))\n"
            "s.close()\n"
            "srv.shutdown()\n"
        )
        rc, out, err = run_child(body, env)
        assert rc == 0, err
        assert "HTTP 200 ok" in out
        assert "makefile True accept True dup True sendfile True" in out

    def test_bases0_recovery_is_guarded(self, guard_env):
        """F2 regression: __bases__[0] yields the in-place-patched wrapper."""
        _, env = guard_env
        body = (
            "import socket\n"
            "b0 = socket.socket.__bases__[0]\n"
            "print('guarded', b0.connect.__name__)\n"
            "s = b0()\n"
            "try:\n"
            "    s.connect(('169.254.169.254', 80))\n"
            "    print('NOT BLOCKED')\n"
            "except OSError as e:\n"
            "    print('BLOCKED', type(e).__name__)\n"
        )
        rc, out, err = run_child(body, env)
        assert rc == 0
        assert "guarded _g_connect" in out
        assert "BLOCKED EgressBlocked" in out

    def test_named_socket_paths_guarded(self, guard_env):
        _, env = guard_env
        body = (
            "import socket, _socket, sys\n"
            "print(socket.socket is _socket.socket)\n"
            "print(socket._socket.socket is socket.socket)\n"
            "print(sys.modules['_socket'].socket is socket.socket)\n"
            "print(socket.create_connection.__name__)\n"
        )
        rc, out, err = run_child(body, env)
        assert rc == 0
        assert "True" in out and out.count("True") == 3
        assert "_g_create_connection" in out

    def test_no_recursion_on_construction(self, guard_env):
        """F0/F0b regression: all families construct and close cleanly."""
        _, env = guard_env
        body = (
            "import socket\n"
            "s1 = socket.socket(); s1.close()\n"
            "s2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s2.close()\n"
            "s3 = socket.socket(socket.AF_INET6, socket.SOCK_STREAM) if hasattr(socket, 'AF_INET6') else None\n"
            "if s3: s3.close()\n"
            "print('CONSTRUCT-OK')\n"
        )
        rc, out, err = run_child(body, env)
        assert rc == 0, err
        assert "CONSTRUCT-OK" in out


class TestBlockPrivateExternal:
    def test_all_private_external_vectors_blocked(self, guard_env):
        _, env = guard_env
        ips = [
            "10.0.0.1", "172.16.0.1", "192.168.1.1", "100.64.0.1",
            "169.254.169.254", "169.254.170.2", "100.100.100.200",
            "0.0.0.0", "::ffff:169.254.169.254", "::ffff:10.0.0.1",
        ]
        body = (
            "import socket\n"
            "ips = " + repr(ips) + "\n"
            "blocked = []\n"
            "for ip in ips:\n"
            "    try:\n"
            "        socket.create_connection((ip, 53), timeout=2)\n"
            "    except OSError as e:\n"
            "        if 'egress guard blocked' in str(e):\n"
            "            blocked.append(ip)\n"
            "print('BLOCKED', sorted(blocked))\n"
        )
        rc, out, err = run_child(body, env)
        assert rc == 0
        import ast
        assert sorted(ips) == sorted(ast.literal_eval(out.split("BLOCKED ")[1]))

    def test_ipv6_vectors(self, guard_env):
        _, env = guard_env
        ips = ["fd00:ec2::254", "fe80::1", "fc00::1", "::"]
        body = (
            "import socket\n"
            "ips = " + repr(ips) + "\n"
            "blocked = []\n"
            "for ip in ips:\n"
            "    try:\n"
            "        socket.create_connection((ip, 53), timeout=2)\n"
            "    except OSError as e:\n"
            "        if 'egress guard blocked' in str(e):\n"
            "            blocked.append(ip)\n"
            "    except Exception:\n"
            "        pass\n"
            "print('BLOCKED', sorted(blocked))\n"
        )
        rc, out, err = run_child(body, env)
        assert rc == 0
        import ast
        assert sorted(ips) == sorted(ast.literal_eval(out.split("BLOCKED ")[1]))

    def test_connect_ex_returns_eacces_and_markers(self, guard_env):
        _, env = guard_env
        body = (
            "import socket\n"
            "s = socket.socket()\n"
            "rc = s.connect_ex(('192.168.1.1', 22))\n"
            "print('RC', rc)\n"
        )
        rc, out, err = run_child(body, env)
        assert rc == 0
        assert "RC 13" in out
        assert "__HERMES_EGRESS_GUARD__:block:192.168.1.1:22:connect_ex" in err

    def test_sendto_and_sendmsg_blocked(self, guard_env):
        _, env = guard_env
        body = (
            "import socket\n"
            "s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n"
            "try:\n"
            "    s.sendto(b'x', ('10.0.0.1', 53))\n"
            "    print('SENDTO-NOT-BLOCKED')\n"
            "except OSError as e:\n"
            "    print('SENDTO-BLOCKED', 'egress guard blocked' in str(e))\n"
            "try:\n"
            "    s.sendmsg([b'x'], (), 0, ('10.0.0.1', 53))\n"
            "    print('SENDMSG-NOT-BLOCKED')\n"
            "except (OSError, AttributeError) as e:\n"
            "    # Windows CPython has no sendmsg at all (nothing to bypass);\n"
            "    # POSIX builds must block it with the guard message.\n"
            "    print('SENDMSG-BLOCKED', 'egress guard blocked' in str(e))\n"
            "    print('SENDMSG-ABSENT', not hasattr(socket.socket, 'sendmsg'))\n"
        )
        rc, out, err = run_child(body, env)
        assert rc == 0
        assert "SENDTO-BLOCKED True" in out
        if "SENDMSG-BLOCKED True" not in out:
            assert "SENDMSG-ABSENT True" in out  # platform without sendmsg

    def test_fail_closed_on_dns_error(self, guard_env):
        _, env = guard_env
        body = (
            "import socket\n"
            "try:\n"
            "    socket.create_connection(('nonexistent-host-zzz.invalid', 80), timeout=3)\n"
            "    print('NOT-BLOCKED')\n"
            "except OSError as e:\n"
            "    print('BLOCKED', 'egress guard blocked' in str(e) or 'DNS failure' in str(e))\n"
        )
        rc, out, err = run_child(body, env)
        assert rc == 0
        assert "BLOCKED True" in out

    def test_metadata_hostnames_no_dns(self, guard_env):
        _, env = guard_env
        body = (
            "import socket\n"
            "hosts = ['metadata.google.internal', 'metadata.goog', 'foo.internal', 'bar.local', 'baz.lan']\n"
            "blocked = []\n"
            "for h in hosts:\n"
            "    try:\n"
            "        socket.getaddrinfo(h, None, socket.AF_UNSPEC, socket.SOCK_STREAM)\n"
            "    except OSError as e:\n"
            "        if 'egress guard blocked' in str(e):\n"
            "            blocked.append(h)\n"
            "print('BLOCKED', sorted(blocked))\n"
        )
        rc, out, err = run_child(body, env)
        assert rc == 0
        import ast

        expected = ["metadata.google.internal", "metadata.goog", "foo.internal",
                    "bar.local", "baz.lan"]
        assert sorted(expected) == sorted(ast.literal_eval(out.split("BLOCKED ")[1]))


class TestLoopbackAllowed:
    def test_loopback_connect_allowed(self, guard_env):
        _, env = guard_env
        body = (
            "import socket, threading\n"
            "srv = socket.socket(); srv.bind(('127.0.0.1', 0)); srv.listen(1)\n"
            "port = srv.getsockname()[1]\n"
            "def accept():\n"
            "    c, _ = srv.accept(); c.close()\n"
            "t = threading.Thread(target=accept, daemon=True); t.start()\n"
            "c = socket.create_connection(('127.0.0.1', port), timeout=5)\n"
            "print('LOOPBACK-OK')\n"
            "c.close()\n"
            "c2 = socket.create_connection(('localhost', port), timeout=5)\n"
            "print('LOCALHOST-OK')\n"
            "c2.close()\n"
            "srv.close()\n"
        )
        rc, out, err = run_child(body, env)
        assert rc == 0, err
        assert "LOOPBACK-OK" in out and "LOCALHOST-OK" in out


class TestAllowlist:
    def test_cdp_and_operator_allowlist(self, guard_env):
        gd, env = guard_env
        env["BU_CDP_WS"] = "ws://127.0.0.1:9222/devtools/browser/x"
        env["HERMES_BROWSER_EXEC_EGRESS_ALLOW"] = "10.0.0.9:8443"
        # The policy JSON is frozen at spawn; the allowlist lives in env and
        # is parsed by the child at install — rebuild the env so the child
        # picks it up (the parent merges operator allow into the policy).
        from tools.browser_exec_egress_guard import _policy_snapshot

        env["HERMES_BROWSER_EXEC_EGRESS_POLICY"] = json.dumps(_policy_snapshot(nonce=NONCE))
        body = (
            "import socket, threading\n"
            "# 10.0.0.9:8443 is allowlisted — the OS would refuse/route it,\n"
            "# but the guard must NOT raise EgressBlocked for the exact pair.\n"
            "try:\n"
            "    socket.create_connection(('10.0.0.9', 8443), timeout=1)\n"
            "    print('ALLOWED-EXACT (OS result)')\n"
            "except OSError as e:\n"
            "    print('ALLOWED-EXACT', 'egress guard blocked' not in str(e))\n"
            "try:\n"
            "    socket.create_connection(('10.0.0.9', 8444), timeout=1)\n"
            "    print('OTHER-PORT-NOT-BLOCKED')\n"
            "except OSError as e:\n"
            "    print('OTHER-PORT-BLOCKED', 'egress guard blocked' in str(e))\n"
        )
        rc, out, err = run_child(body, env)
        assert rc == 0
        assert "ALLOWED-EXACT True" in out
        assert "OTHER-PORT-BLOCKED True" in out


class TestMarkerFdSurvivesRedirect:
    def test_marker_fd_survives_stderr_rebind(self, guard_env):
        """F3 regression: sys.stderr rebinding + dup2(NUL, 2) cannot swallow markers."""
        _, env = guard_env
        body = (
            "import os, sys, socket\n"
            "sys.stderr = open('NUL', 'w')\n"
            "os.dup2(os.open('NUL', os.O_WRONLY), 2)\n"
            "try:\n"
            "    socket.create_connection(('10.1.1.1', 80), timeout=2)\n"
            "except OSError:\n"
            "    pass\n"
            "print('DONE')\n"
        )
        rc, out, err = run_child(body, env)
        assert rc == 0
        assert "DONE" in out
        assert "__HERMES_EGRESS_GUARD__:block:10.1.1.1" in err


class TestTamperTripwire:
    def test_left_in_place_tamper_detected(self, guard_env):
        _, env = guard_env
        body = (
            "import socket\n"
            "# 2-step MRO walk back to the C type — left-in-place tamper.\n"
            "socket.socket = socket.socket.__bases__[0].__bases__[0]\n"
            "print('TAMPERED')\n"
        )
        rc, out, err = run_child(body, env)
        assert rc == 0
        assert "TAMPERED" in out
        assert "__HERMES_EGRESS_GUARD__:tamper:binding" in err


class TestGuardDisableHatch:
    def test_disable_hatch_leaves_sockets_untouched(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
        gd = _egress_guard_dir()
        _verify_or_regenerate(gd)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(gd) + os.pathsep + env.get("PYTHONPATH", "")
        env["HERMES_BROWSER_EXEC_EGRESS_GUARD"] = "0"
        body = (
            "import socket\n"
            "print(socket.socket.__name__)\n"
        )
        rc, out, err = run_child(body, env)
        assert rc == 0
        assert out.strip() == "socket"  # untouched
        assert "__HERMES_EGRESS_GUARD__:" not in err


class TestGuardFileRegeneration:
    def test_forged_stale_files_regenerated_and_nonce_checked(self, tmp_path, monkeypatch):
        """F4 regression: sha256 verify + regenerate; forged marker → withhold."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
        gd = _egress_guard_dir()
        _verify_or_regenerate(gd)
        # Tamper: rewrite sitecustomize.py as a no-op with a forged marker.
        (gd / "sitecustomize.py").write_text(
            "import os\nos.write(2, b'__HERMES_EGRESS_GUARD__:installed:stale-nonce\\n')\n",
            encoding="utf-8",
        )
        env = {}
        assert _install_egress_guard(env) is True
        fresh_nonce = env["HERMES_BROWSER_EXEC_EGRESS_GUARD_NONCE"]
        # The file was regenerated from the checked-in source.
        from tools.browser_exec_egress_guard import _egress_guard_source

        source = _egress_guard_source()
        assert (gd / "sitecustomize.py").read_text(encoding="utf-8") == source["sitecustomize.py"]
        # A forged marker cannot carry the fresh nonce → parent withholds.
        forged_stderr = "__HERMES_EGRESS_GUARD__:installed:stale-nonce\n"
        reason = _parse_guard_markers(forged_stderr, fresh_nonce)
        assert reason is not None
        assert "nonce" in reason
        # Clean installed marker with the fresh nonce passes.
        assert _parse_guard_markers(
            f"__HERMES_EGRESS_GUARD__:installed:{fresh_nonce}\n", fresh_nonce
        ) is None

    def test_block_marker_parsed_as_withhold(self):
        stderr = (
            "__HERMES_EGRESS_GUARD__:installed:nn\n"
            "__HERMES_EGRESS_GUARD__:block:10.0.0.1:80:create_connection\n"
        )
        reason = _parse_guard_markers(stderr, "nn")
        assert reason is not None
        assert "10.0.0.1" in reason

    def test_strip_guard_markers(self):
        stderr = "hello\n__HERMES_EGRESS_GUARD__:installed:nn\nworld\n"
        stripped = _strip_guard_markers(stderr)
        assert "__HERMES_EGRESS_GUARD__" not in stripped
        assert "hello" in stripped and "world" in stripped


class TestEgressProxyPin:
    def test_proxy_pin_sets_vars_and_filters(self, tmp_path, monkeypatch):
        """L2: env proxy vars pinned + a blocked target gets 403 through it."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
        from tools.browser_exec_egress_guard import _pin_egress_proxy

        env = {}
        _pin_egress_proxy(env)
        assert env.get("HTTP_PROXY", "").startswith("http://127.0.0.1:")
        assert "127.0.0.1" in env.get("NO_PROXY", "")
        proxy = env["HTTP_PROXY"]

        import urllib.request

        handler = urllib.request.ProxyHandler({
            "http": proxy, "https": proxy,
        })
        opener = urllib.request.build_opener(handler)
        req = urllib.request.Request("http://169.254.169.254/latest/meta-data/")
        try:
            opener.open(req, timeout=10)
            raise AssertionError("blocked target must not be reachable through the proxy")
        except urllib.error.HTTPError as e:
            assert e.code == 403

    def test_proxy_pin_failure_fails_closed(self, tmp_path, monkeypatch):
        """Defect-6 regression: armed + proxy unpinnable → raise (fail-closed)."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
        import tools.browser_exec_egress_guard as egress

        # Force a fresh pin attempt that cannot start the proxy.
        monkeypatch.setattr(egress, "_PROXY_INSTANCE", None)
        monkeypatch.setattr(egress._EgressFilterProxy, "start", lambda self: None)
        env = {}
        with pytest.raises(RuntimeError):
            egress._pin_egress_proxy(env)
        # The full install path propagates the failure (caller withholds).
        with pytest.raises(RuntimeError):
            egress._install_egress_guard(env)


# ── Defect 4: captured C primitives hidden after install() ────────────────

class TestCapturedPrimitiveHygiene:
    def test_captured_primitives_hidden_after_install(self, guard_env):
        """Defect-4 regression: no raw primitive is a module attribute.

        After ``install()`` the interposer module must not expose
        ``_C_CONNECT`` / ``_C_GETADDR`` / ``_C_SENDTO`` / ``_C_SENDMSG`` (or
        the other captured callables), so ``import browser_exec_egress_guard
        as g; g._C_CONNECT(...)`` is not a one-line bypass — while the guard
        itself keeps working.
        """
        _, env = guard_env
        body = (
            "import browser_exec_egress_guard as g, socket\n"
            "hidden = []\n"
            "for name in ('_C_CONNECT', '_C_CONNECT_EX', '_C_GETADDR',\n"
            "             '_C_SENDTO', '_C_SENDMSG', '_C_CREATE', '_C_INIT', '_C_TYPE'):\n"
            "    if not hasattr(g, name):\n"
            "        hidden.append(name)\n"
            "print('HIDDEN', sorted(hidden))\n"
            "print('SOCK', socket.socket.__name__)\n"
            "try:\n"
            "    socket.create_connection(('169.254.169.254', 80), timeout=2)\n"
            "    print('NOT-BLOCKED')\n"
            "except OSError as e:\n"
            "    print('BLOCKED', 'egress guard blocked' in str(e))\n"
            "try:\n"
            "    g._C_CONNECT\n"
            "    print('PRIMITIVE-EXPOSED')\n"
            "except AttributeError:\n"
            "    print('PRIMITIVE-HIDDEN')\n"
        )
        rc, out, err = run_child(body, env)
        assert rc == 0, err
        import ast

        hidden_line = next(ln for ln in out.splitlines() if ln.startswith("HIDDEN "))
        expected = ["_C_CONNECT", "_C_CONNECT_EX", "_C_CREATE", "_C_GETADDR",
                    "_C_INIT", "_C_SENDMSG", "_C_SENDTO", "_C_TYPE"]
        assert sorted(expected) == sorted(ast.literal_eval(hidden_line.split("HIDDEN ")[1]))
        assert "SOCK _GuardedSocket" in out
        assert "BLOCKED True" in out
        assert "PRIMITIVE-HIDDEN" in out
        assert f"__HERMES_EGRESS_GUARD__:installed:{NONCE}" in err


# ── Reviewer item 1: closure-introspection bypass ──────────────────────────

class TestClosureIntrospectionBypass:
    def test_no_raw_primitive_reachable_via_closure_of_exported_callables(self, guard_env):
        """Adversarial-reviewer regression: ``__closure__`` walks find nothing.

        The pre-fix design captured the raw C primitives
        (``_socket.socket.connect`` & co.) in the closure of the exported
        factory functions (``install``/``_check_and_resolve``/
        ``_g_create_connection``/``_g_getaddrinfo``), so a ~6-line
        introspection snippet walked ``f.__closure__`` and recovered
        ``<method 'connect' of '_socket.socket' objects>`` to dial a private
        IP with zero enforcement and no marker. The module now keeps the
        primitives in a hidden indirection (``_PRIMS``) referenced only
        through module globals: every exported callable is a plain
        module-level function whose ``__closure__`` is None. The deliberate
        MRO/ctypes walk (``socket.socket.__bases__[0].__bases__[0]``) remains
        the accepted residual — this test intentionally does NOT walk it
        (it only uses it to identify raw primitives if any were leaked).
        """
        _, env = guard_env
        body = (
            "import browser_exec_egress_guard as g, _socket, socket, types\n"
            "raw_c_type = socket.socket.__bases__[0].__bases__[0]\n"
            "RAW_NAMES = {'connect', 'connect_ex', 'sendto', 'sendmsg', 'create'}\n"
            "def _is_raw(obj):\n"
            "    if getattr(obj, '__objclass__', None) is raw_c_type:\n"
            "        return getattr(obj, '__name__', '') in RAW_NAMES\n"
            "    return (isinstance(obj, types.FunctionType)\n"
            "            and getattr(obj, '__module__', '') == 'socket'\n"
            "            and getattr(obj, '__name__', '') in ('create_connection', 'getaddrinfo'))\n"
            "found = []\n"
            "seen = set()\n"
            "def _walk(obj, path):\n"
            "    if id(obj) in seen:\n"
            "        return\n"
            "    seen.add(id(obj))\n"
            "    for cell in (getattr(obj, '__closure__', None) or ()):\n"
            "        try:\n"
            "            value = cell.cell_contents\n"
            "        except ValueError:\n"
            "            continue\n"
            "        if _is_raw(value):\n"
            "            found.append(path)\n"
            "        elif callable(value) and not isinstance(value, type):\n"
            "            _walk(value, path + '.<closure>')\n"
            "for name in g.__all__:\n"
            "    obj = getattr(g, name)\n"
            "    if callable(obj):\n"
            "        _walk(obj, name)\n"
            "    elif isinstance(obj, type):\n"
            "        for mname in dir(obj):\n"
            "            mobj = getattr(obj, mname, None)\n"
            "            if callable(mobj) and not isinstance(mobj, type):\n"
            "                _walk(mobj, name + '.' + mname)\n"
            "print('CLOSURE-RAW', sorted(found))\n"
            "print('SOCK', socket.socket.__name__)\n"
            "try:\n"
            "    g._PRIMS\n"
            "    print('PRIMS-VISIBLE')\n"
            "except AttributeError:\n"
            "    print('PRIMS-HIDDEN')\n"
        )
        rc, out, err = run_child(body, env)
        assert rc == 0, err
        import ast

        raw_line = next(ln for ln in out.splitlines() if ln.startswith("CLOSURE-RAW "))
        assert ast.literal_eval(raw_line.split("CLOSURE-RAW ")[1]) == []
        assert "SOCK _GuardedSocket" in out
        assert "PRIMS-HIDDEN" in out
        assert f"__HERMES_EGRESS_GUARD__:installed:{NONCE}" in err
