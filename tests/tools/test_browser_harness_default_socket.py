"""browser_exec outside a served profile: the harness daemon socket must bind under a deep HOME.

browser-harness binds ``<home>/.config/browser-harness/runtime/bu-<BU_NAME>.sock`` unless
``BH_RUNTIME_DIR`` says otherwise. A container's profile home (``$HERMES_HOME/home``) or a long
account home overflowed sun_path and every call died with ``AF_UNIX path too long``.
"""
import json
import os
import socket
import stat
from pathlib import Path

import pytest

import tools.browser_use_cli as bu_cli
from tools import browser_harness_dirs as harness_dirs
from tools.browser_harness_dirs import claim_private_dir as _REAL_CLAIM  # conftest stubs it per test
from tools import browser_session as bs


@pytest.fixture
def harness_env(tmp_path, monkeypatch):
    """A fake harness CLI that echoes the env it was launched with; returns a driver."""
    for key in ("BU_NAME", "BH_HOME", "BROWSER_HARNESS_HOME", "XDG_CONFIG_HOME", "BH_RUNTIME_DIR",
                "BH_TMP_DIR", "BROWSER_USE_API_KEY", "BROWSER_CDP_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(bs, "get_session_info", lambda key: {
        "cdp_url": "ws://127.0.0.1:47000/devtools/browser/x", "features": {"local": True}})
    from tools import browser_supervisor

    class _Registry:
        def get_or_start(self, *a, **kw):
            return None

        def get(self, task_id):
            return None

    monkeypatch.setattr(browser_supervisor, "SUPERVISOR_REGISTRY", _Registry())
    monkeypatch.setattr("hermes_cli.config_effective.load_user_config_effective", lambda: {"browser": {}})
    monkeypatch.setattr("tools.browser_session_cdp.resolve_cdp_override", lambda url: url)
    cli = tmp_path / "browser-harness"
    cli.write_text('#!/bin/sh\ncat > /dev/null\necho "{\\"home\\": \\"$HOME\\", \\"runtime\\": \\"$BH_RUNTIME_DIR\\", '
                   '\\"shared\\": \\"$BH_RUNTIME_DIR_SHARED\\", \\"name\\": \\"${BU_NAME:-default}\\"}"\n',
                   encoding="utf-8")
    cli.chmod(cli.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(bu_cli, "_find_cli", lambda: [str(cli)])
    claimed = []  # conftest keeps unit tests out of the shared temp root; record the claim instead
    monkeypatch.setattr(harness_dirs, "claim_private_dir", lambda path: claimed.append(path) or True)

    def drive(home: str, session: str = "") -> tuple[dict, list]:
        monkeypatch.setenv("HOME", home)
        result = json.loads(bu_cli.browser_exec("print(1)", session=session, task_id="t-sock"))
        assert result["success"] is True, result
        return json.loads(result["output"]), claimed

    return drive


def _bound_socket(seen: dict) -> str:
    """The socket browser_harness/_ipc.py binds for that env: ``bu-<NAME>.sock`` under the resolved
    BH_RUNTIME_DIR (shared stems), else under ``<home>/.config/browser-harness/runtime``."""
    runtime = seen["runtime"] or f"{seen['home']}/.config/browser-harness/runtime"
    return f"{Path(runtime).resolve()}/bu-{seen['name']}.sock"


def _kernel_accepts(path: str) -> bool:
    """An unbound dir answers ENOENT; an overlong path answers "AF_UNIX path too long" first."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.bind(path)
    except OSError as e:
        return "too long" not in str(e)
    else:
        os.unlink(path)
        return True
    finally:
        s.close()


@pytest.mark.platforms("linux")
def test_a_deep_home_gets_a_daemon_socket_the_kernel_accepts(harness_env, tmp_path):
    deep_home = str(tmp_path / "profiles" / ("research-assistant-" * 3) / "home")
    seen, claimed = harness_env(deep_home)
    assert seen["shared"] == "1" and claimed == [seen["runtime"]]  # keeps bu-<name> stems, private dir
    assert _kernel_accepts(_bound_socket(seen)), _bound_socket(seen)


@pytest.mark.platforms("linux")
def test_a_socket_that_already_binds_keeps_its_path_and_a_long_name_is_judged_by_its_own_length(harness_env):
    """The default name under an ordinary home binds today: its daemon (and live socket) stay put.
    A 64-character session under the same home overflows, so that call moves."""
    seen, claimed = harness_env("/home/alexander.smith")
    assert seen["runtime"] == "" and claimed == []
    seen, claimed = harness_env("/home/alexander.smith", session="n" * 64)
    assert seen["runtime"] and _kernel_accepts(_bound_socket(seen)), _bound_socket(seen)


@pytest.mark.platforms("posix")
def test_runtime_root_is_the_users_private_xdg_dir_else_none(tmp_path, monkeypatch):
    monkeypatch.setattr(harness_dirs, "claim_private_dir", _REAL_CLAIM)
    private, open_dir, link = tmp_path / "rt-private", tmp_path / "rt-open", tmp_path / "rt-link"
    private.mkdir(mode=0o700)
    open_dir.mkdir()
    open_dir.chmod(0o755)
    link.symlink_to(private)
    for raw, expected in ((private, str(private)), (open_dir, None), (link, None),
                          ("relative/rt", None), (tmp_path / "missing", None)):
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(raw))
        assert harness_dirs.private_user_runtime_dir() == expected, raw
    # The short dir lands under the private runtime root when one exists.
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(private))
    short = harness_dirs._claim_short_dir("deep-home\0tag", None)
    assert short and os.path.dirname(short) == str(private)
    assert stat.S_IMODE(os.lstat(short).st_mode) == 0o700


@pytest.mark.platforms("posix")
def test_an_existing_short_dir_is_claimed_only_when_private_and_real(tmp_path):
    claim = _REAL_CLAIM
    ok, loose, target = tmp_path / "hbh-ok", tmp_path / "hbh-loose", tmp_path / "elsewhere"
    ok.mkdir(mode=0o700)
    loose.mkdir()
    loose.chmod(0o755)
    target.mkdir(mode=0o700)
    (tmp_path / "hbh-link").symlink_to(target)
    assert claim(str(tmp_path / "hbh-new"))
    assert claim(str(ok))
    assert not claim(str(loose))
    assert not claim(str(tmp_path / "hbh-link"))
