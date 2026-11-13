"""The sandbox's Playwright Chromium is found on every architecture the sandbox image is built for (#153626).

Playwright installs it under ``chrome-linux64/`` on x64 and ``chrome-linux-arm64/`` on arm64; an arm64 sandbox
reported "the sandbox image has no Chromium" with the browser right there. The real probe script runs here
under bash, standing in for the sandbox exec, against a fake browsers directory.
"""

import subprocess
from types import SimpleNamespace

import pytest

from tools.bot_desktop import sandbox_host

pytestmark = pytest.mark.platforms("posix")  # the probe is a POSIX shell script run inside the Linux sandbox


@pytest.fixture
def run_probe(tmp_path, monkeypatch):
    browsers, home = tmp_path / "browsers", tmp_path / "home"
    browsers.mkdir()
    home.mkdir()

    def run_in(env, argv, *, child_env=None, user=None, timeout=30.0, stdin=None):
        return subprocess.run(argv, capture_output=True, timeout=timeout, check=False,
                              env={"PATH": "/usr/bin:/bin", "HOME": str(home), "PLAYWRIGHT_BROWSERS_PATH": str(browsers)})

    monkeypatch.setattr(sandbox_host.streams, "run_in", run_in)

    def probe(*relative, executable=True):
        for rel in relative:
            exe = browsers / rel
            exe.parent.mkdir(parents=True)
            exe.write_text("#!/bin/sh\n", encoding="utf-8")
            exe.chmod(0o755 if executable else 0o644)
        return sandbox_host.chromium_executable(SimpleNamespace(_bd_desktop_user=None))

    probe.browsers = browsers
    return probe


@pytest.mark.parametrize("layout", [
    "chromium-1248/chrome-linux-arm64/chrome",  # arm64 (the reported image)
    "chromium-1248/chrome-linux64/chrome",  # x64
    "chromium-1208/chrome-linux/chrome",  # older Playwright builds
])
def test_the_sandbox_chromium_is_found_on_every_arch(run_probe, layout):
    assert run_probe(layout) == str(run_probe.browsers / layout)


def _host_has_opt_playwright_chromium():
    import glob
    return bool(glob.glob("/opt/playwright/chromium-*/chrome-linux*/chrome"))


@pytest.mark.skipif(_host_has_opt_playwright_chromium(), reason="the probe also searches the host's /opt/playwright")
def test_no_browser_or_a_non_executable_one_is_still_none(run_probe):
    assert run_probe() is None
    assert run_probe("chromium-1248/chrome-linux-arm64/chrome", executable=False) is None


@pytest.mark.skipif(_host_has_opt_playwright_chromium(), reason="the probe also searches the host's /opt/playwright")
def test_the_headless_shell_build_is_not_taken_for_the_headed_browser(run_probe):
    assert run_probe("chromium_headless_shell-1248/chrome-linux-arm64/headless_shell") is None
