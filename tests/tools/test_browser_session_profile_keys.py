"""Served-profile isolation of local browsers: one process serving profile A, then B, then A again must hand
each profile its OWN Chromium (profile dir under that profile's HERMES_HOME), and A's browser again on the
return trip, never B's (#110032). Runs the real session layer + launcher against a fake Chromium binary."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("posix")

_FAKE_CHROMIUM = textwrap.dedent('''\
    #!{python}
    import os, sys, time
    udd = next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--user-data-dir="))
    os.makedirs(udd, exist_ok=True)
    with open(os.path.join(udd, "DevToolsActivePort"), "w") as f:
        f.write("{port}\\n/devtools/browser/{{}}\\n".format(os.getpid()))
    time.sleep(600)
''')


@pytest.fixture
def two_profiles(tmp_path, monkeypatch):
    from tools import browser_launcher as launcher
    script = tmp_path / "chromium"
    script.write_text(_FAKE_CHROMIUM.format(python=sys.executable, port=45679))
    script.chmod(0o755)
    monkeypatch.setattr(launcher, "_resolve_executable", lambda: str(script))
    monkeypatch.setattr(launcher, "chromium_ready", lambda: (True, ""))
    monkeypatch.setattr(launcher, "needs_chromium_sandbox_bypass", lambda: False)
    records = tmp_path / "records"
    records.mkdir()
    monkeypatch.setattr(launcher, "_records_dir", lambda _home: records)
    homes = {}
    for name in ("a", "b"):
        home = tmp_path / f"home-{name}"
        home.mkdir()
        (home / "config.yaml").write_text(json.dumps({"browser": {}}))
        homes[name] = home
    yield homes
    from tools.browser_session_lifecycle import cleanup_all_browsers
    cleanup_all_browsers()


def _session_in(home: Path) -> dict:
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.browser_session import get_session_info
    from tools.browser_use_cli import _backend_cache_key
    token = set_hermes_home_override(str(home))
    try:
        return get_session_info(_backend_cache_key("same-task"))
    finally:
        reset_hermes_home_override(token)


def _pid(session: dict) -> int:
    return int(session["cdp_url"].rsplit("/", 1)[1])


def test_profile_a_then_b_then_a_each_get_their_own_browser(two_profiles):
    first_a = _session_in(two_profiles["a"])
    b = _session_in(two_profiles["b"])
    again_a = _session_in(two_profiles["a"])

    assert _pid(first_a) != _pid(b)
    assert _pid(again_a) == _pid(first_a)

    from tools.browser_launcher import get_browser
    dirs = {name: Path(get_browser(s["launcher_key"]).user_data_dir)
            for name, s in (("a", first_a), ("b", b))}
    assert two_profiles["a"] in dirs["a"].parents
    assert two_profiles["b"] in dirs["b"].parents


def test_cleaning_up_one_profile_leaves_the_other_profiles_browser_running(two_profiles):
    from hermes_constants import hermes_home_key
    from tools.browser_launcher import get_browser
    from tools.browser_session_lifecycle import cleanup_profile_browsers
    a = _session_in(two_profiles["a"])
    b = _session_in(two_profiles["b"])

    cleanup_profile_browsers(hermes_home_key(two_profiles["b"]))

    assert get_browser(b["launcher_key"]) is None
    assert get_browser(a["launcher_key"]) is not None
