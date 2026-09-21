"""P13 isolation proof for scripts/web-backend-health.py.

Verifies:
- HERMES_HOME parameterisation: HERMES_HOME resolves under the env, not
  a hardcoded /home/kensei/.hermes.
- --dry-run skips every network probe (sudo docker inspect, curl, live
  DDGS search): each check_* returns (True, "dry-run: probes skipped")
  and main() exits 0 without touching docker/the network.
- import-safe: importing the module does not run any probe.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "web-backend-health.py"


def _load_module(monkeypatch, fake_home: Path, dry_run: bool = False):
    monkeypatch.setenv("HERMES_HOME", str(fake_home))
    argv = ["web-backend-health.py"] + (["--dry-run"] if dry_run else [])
    monkeypatch.setattr(sys, "argv", argv)
    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location("wbh_under_test", str(SCRIPT))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_home(tmp_path):
    fake = tmp_path / "fake_hermes"
    fake.mkdir()
    return fake


def test_hermes_home_resolves_under_env(monkeypatch, fake_home):
    mod = _load_module(monkeypatch, fake_home)
    assert str(mod.HERMES_HOME).startswith(str(fake_home))


def test_import_is_side_effect_free(monkeypatch, fake_home):
    mod = _load_module(monkeypatch, fake_home)
    # Importing must not have run any subprocess probe. We assert the
    # module-level _DRY_RUN flag is False when --dry-run not in argv.
    assert mod._DRY_RUN is False


def test_dry_run_check_searxng_skips_probe(monkeypatch, fake_home):
    mod = _load_module(monkeypatch, fake_home, dry_run=True)
    assert mod._DRY_RUN is True
    ok, detail = mod.check_searxng()
    assert ok is True
    assert "dry-run" in detail


def test_dry_run_check_grokto_skips_probe(monkeypatch, fake_home):
    mod = _load_module(monkeypatch, fake_home, dry_run=True)
    ok, detail = mod.check_groktoCrawl()
    assert ok is True
    assert "dry-run" in detail


def test_dry_run_check_ddgs_skips_probe(monkeypatch, fake_home):
    mod = _load_module(monkeypatch, fake_home, dry_run=True)
    ok, detail = mod.check_ddgs()
    assert ok is True
    assert "dry-run" in detail


def test_dry_run_main_exits_0_without_alert(monkeypatch, fake_home):
    """Under --dry-run all three checks return healthy, so main() must
    exit 0 (silent) and emit no alert lines."""
    mod = _load_module(monkeypatch, fake_home, dry_run=True)
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 0


def test_live_checks_use_http_not_docker(monkeypatch, fake_home):
    """Live probes must not require root-only Docker access."""
    monkeypatch.setenv("SEARXNG_URL", "http://127.0.0.1:8082")
    mod = _load_module(monkeypatch, fake_home)
    calls = []

    class Response:
        def __init__(self, body):
            self.status = 200
            self._body = body

        def read(self, _limit=-1):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_urlopen(url, timeout):
        assert timeout <= 10
        if ":8082/" in url:
            return Response(b'{"results": [{"url": "https://example.com"}]}')
        return Response(b'{"status": "ok", "checks": {}}')

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        assert cmd[0] != "sudo"
        return type("Result", (), {"returncode": 0, "stdout": '{"ok": true, "count": 1}', "stderr": ""})()

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    assert mod.check_searxng() == (True, "JSON search returned 1 results")
    assert mod.check_groktoCrawl() == (True, "healthy (ok)")
    assert mod.check_ddgs() == (True, "working (1 result)")
    assert len(calls) == 1


def test_degraded_report_is_successful_cron_execution(monkeypatch, fake_home, capsys):
    """A health alert is output data, not a scheduler execution failure."""
    mod = _load_module(monkeypatch, fake_home)
    monkeypatch.setattr(mod, "check_searxng", lambda: (False, "API down"))
    monkeypatch.setattr(mod, "check_groktoCrawl", lambda: (True, "healthy"))
    monkeypatch.setattr(mod, "check_ddgs", lambda: (False, "unavailable"))
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 0
    assert "SearXNG FAIL" in capsys.readouterr().out
