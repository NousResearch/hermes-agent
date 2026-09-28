"""Windows essentials and native Winsock/legacy-encoding controls.

Actual production UTF-8 env and Unicode RPC are exercised in
`test_code_execution_modes.py`; never reproduce the production scrubber here.
"""

import os
import subprocess
import sys
import textwrap
import time

import pytest

from tools.code_execution_env import _scrub_child_env
from tools import code_execution_env


def _no_passthrough(_):
    return False


@pytest.mark.parametrize("windows", [False, True])
@pytest.mark.parametrize("passthrough", [False, True])
def test_native_essentials_and_passthrough_priority(windows, passthrough):
    # Platform is explicit input to this pure policy helper, not a fake host.
    essentials = {
        "SYSTEMROOT": r"C:\Windows", "SystemRoot": r"C:\Windows",
        "SystemDrive": "C:", "WINDIR": r"C:\Windows",
        "ComSpec": r"C:\Windows\System32\cmd.exe", "comspec": r"C:\Windows\System32\cmd.exe",
        "APPDATA": r"C:\Users\alice\AppData\Roaming",
        "LOCALAPPDATA": r"C:\Users\alice\AppData\Local",
    }
    safe = {"PATH": r"C:\Windows\System32;C:\Python", "HOME": r"C:\Users\alice",
            "PATHEXT": ".COM;.EXE;.BAT;.CMD;.PY",
            "USERPROFILE": r"C:\Users\alice", "TEMP": r"C:\Users\alice\Temp"}
    secret = {"OPENAI_API_KEY": "fake-provider", "GITHUB_TOKEN": "fake-github",
              "MY_PASSWORD": "fake-password", "TENOR_API_KEY": "fake-third-party",
              "RANDOM_UNKNOWN_VAR": "unknown"}
    result = _scrub_child_env({**essentials, **safe, **secret}, is_windows=windows,
                             is_passthrough=lambda k: passthrough and k == "TENOR_API_KEY")
    assert result == {**safe, **(essentials if windows else {}),
                      **({"TENOR_API_KEY": "fake-third-party"} if passthrough else {})}


# ``platforms("windows")`` rather than ``skipif(sys.platform != "win32")``: the
# dedicated Windows CI job selects its files by grepping for the marker, so a
# bare skipif is invisible to it — the file is never imported there and these
# tests run on no host at all.
@pytest.mark.platforms("windows")
class TestWindowsSocketSmokeTest:
    """Integration-ish smoke test: spawn a child Python with a scrubbed
    env and confirm it can create an AF_INET socket.  This is the
    regression that motivated the fix — without SYSTEMROOT the child
    hits WinError 10106 before any RPC is attempted."""

    def test_child_can_create_socket_with_scrubbed_env(self):
        scrubbed = _scrub_child_env(os.environ, is_passthrough=_no_passthrough)

        # Build a tiny child script that simply opens an AF_INET socket.
        script = textwrap.dedent("""
            import socket, sys
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.close()
                print("OK")
                sys.exit(0)
            except OSError as exc:
                print(f"FAIL: {exc}")
                sys.exit(1)
        """).strip()

        result = subprocess.run(
            [sys.executable, "-c", script],
            env=scrubbed,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, (
            f"Child failed to create socket with scrubbed env:\n"
            f"  stdout={result.stdout!r}\n"
            f"  stderr={result.stderr!r}\n"
            f"  scrubbed keys={sorted(scrubbed.keys())}"
        )
        assert "OK" in result.stdout


class TestNativeLegacyEncodingControls:
    @pytest.mark.platforms("windows")
    def test_windows_default_encoding_would_have_failed(self):
        """Negative control: prove that on Windows, writing the stub
        *without* ``encoding="utf-8"`` would corrupt the file.  If this
        test ever starts failing (i.e. default write succeeds), it means
        Python's default encoding has changed and the explicit UTF-8
        requirement may be obsolete — reconsider the fix."""
        from tools.code_execution_tool import generate_hermes_tools_module
        import tempfile

        stub = generate_hermes_tools_module(["terminal"], transport="uds")
        # Find a non-ASCII character we can use to prove the corruption.
        non_ascii = [c for c in stub if ord(c) > 127]
        if not non_ascii:
            pytest.skip("stub has no non-ASCII chars — nothing to corrupt")

        # Write with default encoding (simulating the old buggy code).
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            try:
                f.write(stub)
                tmp_path = f.name
                wrote_successfully = True
            except UnicodeEncodeError:
                # Default encoding can't even encode it — that's the bug
                # in a different form.  Still proves the point.
                tmp_path = f.name
                wrote_successfully = False

        try:
            if not wrote_successfully:
                # Default-encoding write raised outright.  The bug is real.
                return

            # Read back as UTF-8 (what Python does on import).
            with open(tmp_path, encoding="utf-8") as fh:
                try:
                    fh.read()
                    # If this succeeds on Windows, the platform default is
                    # already UTF-8 (e.g. Python 3.15 with UTF-8 mode on).
                    # In that case the explicit encoding= is belt-and-
                    # suspenders but no longer strictly required.  Skip.
                    pytest.skip(
                        "Default text-file encoding is UTF-8-compatible on "
                        "this Windows build — explicit encoding= is no "
                        "longer load-bearing, but keep it for belt-and-"
                        "suspenders."
                    )
                except UnicodeDecodeError:
                    # Exactly the failure mode that motivated the fix.
                    pass
        finally:
            os.unlink(tmp_path)

    @pytest.mark.platforms("windows")
    def test_windows_child_without_utf8_env_would_fail(self):
        """Negative control: spawn a Python child *without* our env
        overrides and prove that on Windows, printing non-ASCII fails.
        If this ever starts passing, Python has changed its default
        stdio encoding on Windows and the fix may be obsolete — but
        keep the env vars anyway for belt-and-suspenders."""
        script = textwrap.dedent("""
            import sys
            print("em-dash \\u2014 arrow \\u2192")
            sys.exit(0)
        """).strip()

        # Scrubbed env WITHOUT the PYTHONIOENCODING / PYTHONUTF8 overrides.
        # Also scrub PYTHONUTF8 and PYTHONIOENCODING from the inherited
        # env so we reproduce the buggy state even if the parent test
        # runner has them set.
        scrubbed = _scrub_child_env(os.environ, is_passthrough=_no_passthrough)
        for k in ("PYTHONIOENCODING", "PYTHONUTF8", "PYTHONLEGACYWINDOWSSTDIO"):
            scrubbed.pop(k, None)

        result = subprocess.run(
            [sys.executable, "-c", script],
            env=scrubbed,
            capture_output=True,
            text=False,
            timeout=15,
        )
        # Either the child crashed (expected), or modern Python handled
        # it anyway — in which case the fix is still defensive but no
        # longer strictly required.  Skip with a note if so.
        if result.returncode == 0 and b"\xe2\x80\x94" in result.stdout:
            pytest.skip(
                "This Python/Windows build handles non-ASCII stdout even "
                "without PYTHONIOENCODING/PYTHONUTF8 — fix is defensive "
                "but no longer strictly load-bearing.  Keep the env vars "
                "for older Python builds and C.ASCII-locale containers."
            )
        # Otherwise: crash OR garbled output — both count as proving the
        # bug is real on this system.


def _configured_timezone_child_env():
    return code_execution_env._build_child_env(
        rpc_endpoint="socket",
        rpc_token="token",
        tmpdir="/tmp/hermes-code-execution-test",
        child_python=sys.executable,
    )




@pytest.mark.platforms("windows")
def test_windows_live_child_offset_matches_os_zone_when_timezone_is_configured(monkeypatch):
    """The user-visible contract of #112233: with ``timezone:`` configured, a real Windows child
    must report the OS zone's UTC offset — an IANA name in ``TZ`` made the MSVC runtime derive
    ``time.timezone == 0`` (+01:00 instead of -07:00) while ``time.tzname`` still read correctly."""
    import json

    monkeypatch.setattr("hermes_time.get_timezone_name", lambda: "America/Los_Angeles")
    child_env = _configured_timezone_child_env()
    assert "TZ" not in child_env

    # The runner starts Python with TZ=UTC; its cached timezone is not the OS zone.
    # Query a fresh control process with TZ removed, independently of the builder.
    control_env = os.environ.copy()
    control_env.pop("TZ", None)
    timestamp = str(time.time())  # Both children observe the same instant across DST changes.
    script = (
        "import datetime, json, sys, time; "
        "instant = datetime.datetime.fromtimestamp(float(sys.argv[1]), datetime.timezone.utc); "
        "print(json.dumps([time.timezone, instant.astimezone().utcoffset().total_seconds()]))"
    )

    def read_timezone(env):
        result = subprocess.run(
            [sys.executable, "-c", script, timestamp],
            env=env, stdin=subprocess.DEVNULL, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=30,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    assert read_timezone(child_env) == read_timezone(control_env)
