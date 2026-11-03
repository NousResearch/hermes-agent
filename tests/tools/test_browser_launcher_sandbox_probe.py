"""The AppArmor userns arm of the Chromium sandbox bypass is a property of the binary, not the host.

Under ``kernel.apparmor_restrict_unprivileged_userns=1`` only a Chromium without a ``userns`` grant (the
PM/Playwright build) dies with ``No usable sandbox!``; a ``browser.executable_path`` pin that has one (the
chromium snap, Google Chrome under Ubuntu's ``/etc/apparmor.d/chrome``) must keep its sandbox. A fake
``chromium`` stands in for both kinds: run with ``--dump-dom`` (the launcher's sandbox probe) it either
prints the DOM or Chromium's sandbox FATAL; otherwise it writes ``DevToolsActivePort`` like a real browser.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

import pytest

from tools import browser_launcher as launcher

_FAKE = textwrap.dedent('''\
    #!{python}
    import json, os, sys, time
    udd = next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--user-data-dir="))
    if "--dump-dom" in sys.argv:
        with open({probes!r}, "a") as f:
            f.write("probe\\n")
        if not {sandbox_ok}:
            sys.stderr.write("[1:1:FATAL:zygote_host_impl_linux.cc(128)] No usable sandbox! ...\\n")
            sys.exit(134)
        print("<html><head></head><body></body></html>")
        sys.exit(0)
    os.makedirs(udd, exist_ok=True)
    with open(os.path.join(udd, "argv.json"), "w") as f:
        json.dump(sys.argv, f)
    with open(os.path.join(udd, "DevToolsActivePort"), "w") as f:
        f.write("45678\\n/devtools/browser/fake-id\\n")
    time.sleep(600)
''')


def _fake(tmp_path: Path, *, sandbox_ok: bool) -> tuple[Path, Path]:
    name = "chromium-with-userns-grant" if sandbox_ok else "chromium-without-userns-grant"
    script, probes = tmp_path / name, tmp_path / f"{name}.probes"
    script.write_text(_FAKE.format(python=sys.executable, probes=str(probes), sandbox_ok=sandbox_ok))
    script.chmod(0o755)
    return script, probes


def _probe_count(probes: Path) -> int:
    return len(probes.read_text().split()) if probes.exists() else 0


@pytest.fixture
def restricted_host(monkeypatch):
    """A non-root, non-container host where AppArmor restricts unprivileged user namespaces."""
    monkeypatch.setattr(launcher.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(launcher, "running_in_docker", lambda: False)
    monkeypatch.setattr(launcher, "apparmor_restricts_unprivileged_userns", lambda: True)
    monkeypatch.setattr(launcher, "_sandbox_verdicts", {}, raising=False)  # per-process probe cache
    monkeypatch.setattr("tools.browser_session_cloud.chromium_args", list)
    monkeypatch.setattr("tools.browser_session_cloud.is_headed_mode", lambda: False)
    yield
    launcher.stop_all_browsers()


def _launch_argv(monkeypatch, exe: Path, key: str) -> list[str]:
    """The argv the agent's default launch path starts ``exe`` with (a ``browser.executable_path`` pin
    resolves inside the packaged branch)."""
    monkeypatch.setattr(launcher, "_resolve_executable", lambda: str(exe))
    browser, err = launcher.launch_chromium(key, env=dict(os.environ))
    assert browser is not None, err
    return json.loads(Path(browser.user_data_dir, "argv.json").read_text())


@pytest.mark.platforms("linux")
def test_a_pinned_browser_that_can_sandbox_keeps_its_sandbox(restricted_host, tmp_path, monkeypatch):
    exe, _ = _fake(tmp_path, sandbox_ok=True)
    argv = _launch_argv(monkeypatch, exe, "pinned")
    assert not set(launcher.CHROMIUM_SANDBOX_BYPASS_ARGS) & set(argv)


@pytest.mark.platforms("linux")
def test_a_build_the_restriction_blocks_still_gets_the_bypass_after_one_probe(restricted_host, tmp_path,
                                                                             monkeypatch):
    exe, probes = _fake(tmp_path, sandbox_ok=False)
    for key in ("first", "second"):
        argv = _launch_argv(monkeypatch, exe, key)
        assert set(launcher.CHROMIUM_SANDBOX_BYPASS_ARGS) <= set(argv)
    assert _probe_count(probes) == 1  # one probe per binary per process, not per launch


@pytest.mark.platforms("linux")
def test_the_dock_icon_and_the_agent_start_one_binary_with_the_same_sandbox_flags(restricted_host, tmp_path,
                                                                                  monkeypatch):
    """The human's dock click and the agent's launch run the same binary on the same profile: a binary that
    sandboxes must not get --no-sandbox on the dock either, and the dock reuses the agent's probe."""
    from tools.bot_desktop import browser as bd_browser
    for sandbox_ok in (True, False):
        exe, probes = _fake(tmp_path, sandbox_ok=sandbox_ok)
        agent = set(_launch_argv(monkeypatch, exe, f"dock-{sandbox_ok}")) & set(launcher.CHROMIUM_SANDBOX_BYPASS_ARGS)
        dock = set(bd_browser.dock_argv(str(exe), "/p/dir")) & set(launcher.CHROMIUM_SANDBOX_BYPASS_ARGS)
        assert dock == agent and bool(agent) is not sandbox_ok
        assert _probe_count(probes) == 1
