"""Exercise cache contention through the real warmup subprocess path."""
import json
import os
import subprocess
import sys
import time

from tools import browser_tool as browser
from tools import browser_tool_install as install


def test_shared_cache_contention_is_bounded_and_releases_after_warmup(tmp_path):
    trace, release = tmp_path / 'trace', tmp_path / 'release'
    fake = tmp_path / 'fake_npx.py'
    fake.write_text(
        'from pathlib import Path\nimport time\n'
        f'trace=Path({str(trace)!r}); release=Path({str(release)!r})\n'
        "with trace.open('a') as f: f.write('enter\\n')\n"
        'while not release.exists(): time.sleep(0.01)\n'
        "with trace.open('a') as f: f.write('exit\\n')\n",
        encoding='utf-8',
    )
    worker = '''
import json, subprocess, sys
from tools import browser_tool_install as install
real_popen = subprocess.Popen
install._resolve_npx_bin = lambda: 'fake-npx'
def spawn(cmd, **kwargs):
    assert cmd[0] == 'fake-npx'
    return real_popen([sys.executable, sys.argv[1], *cmd[1:]], **kwargs)
install.subprocess.Popen = spawn
print(json.dumps(install.warm_agent_browser_npx_cache(float(sys.argv[2]))))
'''
    processes = []

    def start(profile, timeout):
        env = dict(os.environ, HERMES_HOME=str(tmp_path / profile),
                   NPM_CONFIG_CACHE=str(tmp_path / 'shared-cache'))
        proc = subprocess.Popen([sys.executable, '-c', worker, str(fake), str(timeout)],
                                env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        processes.append(proc)
        return proc

    first = start('profile-a', 15)
    try:
        deadline = time.monotonic() + 10
        while not trace.exists() and first.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert trace.exists(), first.communicate(timeout=2)
        # The first real subprocess holds the cache lock. Another profile must
        # exhaust its own budget without launching a competing npm writer.
        second = start('profile-b', 0.25)
        out, err = second.communicate(timeout=10)
        assert second.returncode == 0, err
        assert json.loads(out) is False
        assert trace.read_text().splitlines() == ['enter']
        release.touch()
        out, err = first.communicate(timeout=10)
        assert first.returncode == 0, err
        assert json.loads(out) is True
        third = start('profile-b', 5)
        out, err = third.communicate(timeout=10)
        assert third.returncode == 0, err
        assert json.loads(out) is True
        assert trace.read_text().splitlines() == ['enter', 'exit', 'enter', 'exit']
        assert (tmp_path / 'shared-cache' / '.hermes-agent-browser-warmup.lock').exists()
    finally:
        release.touch()
        for proc in processes:
            if proc.poll() is None:
                proc.kill()
            proc.communicate(timeout=10)


def test_validated_fallback_warms_before_publishing_cached_sentinel(monkeypatch):
    monkeypatch.setattr(browser, '_agent_browser_resolved', False)
    monkeypatch.setattr(browser, '_cached_agent_browser', None)
    monkeypatch.setattr(install, '_agent_browser_candidates', lambda path: ())
    monkeypatch.setattr(install, '_resolve_npx_bin', lambda: 'fake-npx')
    calls = []

    def warm():
        assert not browser._agent_browser_resolved
        assert browser._cached_agent_browser is None
        calls.append('warm')
        return False  # Best-effort failure preserves the existing fallback.

    monkeypatch.setattr(install, 'warm_agent_browser_npx_cache', warm)
    assert install._find_agent_browser(validate=False) == browser.NPX_AGENT_BROWSER_SENTINEL
    assert calls == []
    assert install._find_agent_browser() == browser.NPX_AGENT_BROWSER_SENTINEL
    assert calls == ['warm']
    assert browser._agent_browser_resolved
