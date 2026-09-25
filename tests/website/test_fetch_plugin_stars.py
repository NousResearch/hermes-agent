"""fetch-plugin-stars.py: plugin-catalog star counts, GitHub consulted only from the scheduled run.

The contract under test is rate-limit discipline, not the numbers: a deploy (no ``--probe``)
must never reach GitHub, the scheduled probe must be ONE request for every repo, and a failed
probe must keep the previous counts rather than zeroing them.
"""

from __future__ import annotations

import importlib.util
import json
import os
import urllib.error
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "website" / "scripts" / "fetch-plugin-stars.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("fetch_plugin_stars", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _catalog(tmp_path: Path, *repos: str) -> Path:
    import hermes_yaml as yaml

    cat = tmp_path / "plugin-catalog"
    cat.mkdir()
    for i, repo in enumerate(repos):
        (cat / f"p{i}.yaml").write_text(yaml.safe_dump({
            "name": f"p{i}", "repo": repo, "sha": "38fe0fb53eff98d477f807432e965429e665ca33",
            "description": "d", "maintainer": "m"}), encoding="utf-8")
    return cat


def test_deploy_reuses_the_cache_without_any_github_call(mod, tmp_path, monkeypatch):
    cat = _catalog(tmp_path, "https://github.com/a/one")
    out = tmp_path / "plugin-stars.json"
    out.write_text(json.dumps({"fetched_at": "2026-01-01T00:00:00+00:00", "stars": {"a/one": 7}}), encoding="utf-8")

    def boom(*a, **k):
        raise AssertionError("GitHub must not be called without --probe")
    monkeypatch.setattr(mod, "_graphql", boom)
    monkeypatch.setattr(mod, "_http_json", boom)

    assert mod.main(catalog_dir=cat, output=out, probe=False, live_url=None) == 0
    assert json.loads(out.read_text())["stars"] == {"a/one": 7}


def test_probe_is_one_graphql_request_and_a_failure_keeps_previous_counts(mod, tmp_path, monkeypatch):
    cat = _catalog(tmp_path, "https://github.com/a/one", "https://github.com/b/two", "https://gitlab.com/c/three")
    out = tmp_path / "plugin-stars.json"
    out.write_text(json.dumps({"fetched_at": "2026-01-01T00:00:00+00:00", "stars": {"a/one": 7, "b/two": 9}}),
                   encoding="utf-8")
    calls: list[str] = []

    def one_request(query, token):
        calls.append(query)
        # b/two errored (renamed repo): its node is null, previous count must survive.
        return {"data": {"r0": {"stargazerCount": 42}, "r1": None},
                "errors": [{"message": "Could not resolve to a Repository"}]}
    monkeypatch.setattr(mod, "_graphql", one_request)

    assert mod.main(catalog_dir=cat, output=out, probe=True, live_url=None, token="t") == 0
    data = json.loads(out.read_text())
    assert data["stars"] == {"a/one": 42, "b/two": 9}
    assert len(calls) == 1 and "gitlab" not in calls[0] and 'owner: "a"' in calls[0] and 'owner: "b"' in calls[0]
    assert data["fetched_at"] > "2026-01-01"

    # A rate-limited / failed probe keeps everything as it was.
    def limited(query, token):
        raise urllib.error.HTTPError("u", 403, "rate limited", hdrs=None, fp=None)
    monkeypatch.setattr(mod, "_graphql", limited)
    assert mod.main(catalog_dir=cat, output=out, probe=True, live_url=None, token="t") == 0
    assert json.loads(out.read_text())["stars"] == {"a/one": 42, "b/two": 9}


def test_failed_probe_keeps_the_previous_timestamp_and_warns(mod, tmp_path, monkeypatch, capsys):
    """An expired-token 401 must not restamp ``fetched_at``: the catalog footer reads it as
    "ranking as of <date>" and showed today's date over five-day-old counts (#118113)."""
    cat = _catalog(tmp_path, "https://github.com/a/one", "https://github.com/b/two")
    out = tmp_path / "plugin-stars.json"
    out.write_text(json.dumps({"fetched_at": "2026-09-16T18:40:16+00:00", "stars": {"a/one": 7}}), encoding="utf-8")

    def unauthorized(query, token):
        raise urllib.error.HTTPError("u", 401, "Unauthorized", hdrs=None, fp=None)
    monkeypatch.setattr(mod, "_graphql", unauthorized)

    assert mod.main(catalog_dir=cat, output=out, probe=True, live_url=None, token="expired") == 0
    data = json.loads(out.read_text())
    assert data == {"fetched_at": "2026-09-16T18:40:16+00:00", "stars": {"a/one": 7}}
    assert "::warning::" in capsys.readouterr().out


def test_script_imports_hermes_yaml_under_the_ci_invocation():
    """skills-index.yml runs ``python website/scripts/fetch-plugin-stars.py --probe`` with no
    PYTHONPATH (unlike deploy-site, whose setup-pm exports PYTHONPATH=<repo>). b4a294f9 added
    ``import hermes_yaml`` without the sys.path.insert the sibling scripts have, so every
    scheduled run died with ModuleNotFoundError and the stale-index watchdog reopened (#122609).
    Regression: the script must import cleanly as a plain file, the way CI invokes it."""
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run([sys.executable, str(SCRIPT)], env=env,
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "Reused plugin stars" in proc.stdout
