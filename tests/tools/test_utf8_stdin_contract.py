"""The Windows UTF-8 stdin contract is structural, not inherited.

Regression class (peer-dm, 55eb05ba8e): a parent that pipes raw non-ASCII
into a child which reads ``sys.stdin`` as text depends on a UTF-8 pin
reaching the child's env. ``hermes_bootstrap`` setdefaults the pin into
``os.environ`` at entry points, so a hermes parent normally transfers it by
inheritance — but a parent spawned outside that path (cron, CI, service
manager) exports nothing, and the child then decodes with the LOCALE codec
(cp936 on a GBK-locale Windows box): mojibake. These tests build the child
env through the REAL factories and run a REAL child interpreter, so they
can only pass when the factories themselves carry the contract.
"""

import os
import subprocess
import sys
import textwrap

import pytest

from tools.environments.local import build_subprocess_env


@pytest.fixture(autouse=True)
def _sandbox(tmp_path, monkeypatch):
    # The factories' PATH work resolves the hermes install bin dir; on a
    # default-install checkout (repo inside the real Hermes home) that
    # probes real-home state tests/home_io_guard.py rightly refuses — same
    # tripwire clean_slate's _isolate fixes, same triple.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "pm-runtime"))
    monkeypatch.setattr(
        "hermes_constants.get_hermes_home", lambda: tmp_path / "hermes-home")
    monkeypatch.setattr(
        "tools.environments.local._resolve_hermes_bin_dir", lambda: None)


def test_factory_pins_the_utf8_stdin_contract_from_a_scrubbed_base():
    """A cron/CI parent whose environ carries no pins must still export them:
    the contract lives in the factory, not in what the parent inherited."""
    base = {"PATH": os.defpath, "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
    env = build_subprocess_env(base)
    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"


def test_served_profile_child_env_carries_the_contract():
    """The delivery path's factory (``delivery_env`` builds on it) must pin
    too — this is the exact env the peer-dm stdin pipe travels through."""
    from tools.environments.local import served_profile_child_env

    env = served_profile_child_env(base={})
    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"


def test_explicit_user_opt_out_wins_over_the_contract():
    """setdefault, not assignment: PYTHONUTF8=0 / a chosen PYTHONIOENCODING
    are authoritative user settings the factory must not override."""
    env = build_subprocess_env({"PYTHONUTF8": "0", "PYTHONIOENCODING": "latin-1"})
    assert env["PYTHONUTF8"] == "0"
    assert env["PYTHONIOENCODING"] == "latin-1"


def test_gbk_simulated_child_reads_utf8_stdin_verbatim(tmp_path):
    """The peer-dm bug class, end to end: strip every pin from the parent
    environ (as a service-manager launch would), build the child env through
    the REAL factory, and pipe raw non-ASCII into a child that reads
    ``sys.stdin`` as TEXT. On a GBK-locale Windows box an unpinned child
    decodes cp936 and this round trip arrives as mojibake — only the
    factory's pin can save it."""
    child = tmp_path / "reader.py"
    child.write_text(
        textwrap.dedent(
            """\
            import sys

            # The product shape: decode stdin with the locale codec (text
            # read), re-encode as UTF-8 — the codecs differ, so corruption
            # sticks. An unpinned child on a GBK-locale box decodes cp936 and
            # this round trip arrives as mojibake; only a UTF-8 pin in the
            # child env can save it.
            sys.stdout.buffer.write(sys.stdin.read().encode("utf-8"))
            """
        ),
        encoding="utf-8",
    )
    payload = "secret λ 世界 ✅"

    env = dict(os.environ)
    for name in ("PYTHONUTF8", "PYTHONIOENCODING", "PYTHONLEGACYWINDOWSSTDIO"):
        env.pop(name, None)
    env = build_subprocess_env(env)

    result = subprocess.run(
        [sys.executable, str(child)], input=payload.encode("utf-8"),
        env=env, capture_output=True, timeout=30)

    assert result.returncode == 0, result.stderr
    assert result.stdout.decode("utf-8") == payload
