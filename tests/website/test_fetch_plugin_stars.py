"""fetch-plugin-stars.py: the daily GitHub-stars cache behind catalog ranking.

The contract under test is rate-limit discipline, not the numbers: a fresh cache must never
reach GitHub, and a rate-limited probe must keep the previous counts rather than zeroing them.
"""

from __future__ import annotations

import importlib.util
import json
import urllib.error
from datetime import datetime, timedelta, timezone
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
    import yaml

    cat = tmp_path / "plugin-catalog"
    cat.mkdir()
    for i, repo in enumerate(repos):
        (cat / f"p{i}.yaml").write_text(yaml.safe_dump({
            "name": f"p{i}", "repo": repo, "sha": "38fe0fb53eff98d477f807432e965429e665ca33",
            "description": "d", "maintainer": "m"}), encoding="utf-8")
    return cat


def test_fresh_cache_is_reused_without_any_github_call(mod, tmp_path, monkeypatch):
    cat = _catalog(tmp_path, "https://github.com/a/one")
    out = tmp_path / "plugin-stars.json"
    recent = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    out.write_text(json.dumps({"fetched_at": recent, "stars": {"a/one": 7}}), encoding="utf-8")

    def boom(*a, **k):
        raise AssertionError("GitHub must not be called while the cache is fresh")
    monkeypatch.setattr(mod, "_http_json", boom)

    assert mod.main(catalog_dir=cat, output=out, max_age_hours=24, live_url=None) == 0
    assert json.loads(out.read_text())["stars"] == {"a/one": 7}


def test_stale_cache_probes_and_rate_limit_keeps_previous_counts(mod, tmp_path, monkeypatch):
    cat = _catalog(tmp_path, "https://github.com/a/one", "https://github.com/b/two", "https://gitlab.com/c/three")
    out = tmp_path / "plugin-stars.json"
    old = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    out.write_text(json.dumps({"fetched_at": old, "stars": {"a/one": 7, "b/two": 9}}), encoding="utf-8")

    calls: list[str] = []

    def fake(url, headers, timeout=15.0):
        calls.append(url)
        if url.endswith("/repos/a/one"):
            return {"stargazers_count": 42}
        raise urllib.error.HTTPError(url, 403, "rate limited", hdrs=None, fp=None)
    monkeypatch.setattr(mod, "_http_json", fake)

    assert mod.main(catalog_dir=cat, output=out, max_age_hours=24, live_url=None) == 0
    data = json.loads(out.read_text())
    # a/one refreshed; b/two kept its old count instead of dropping to 0; gitlab never probed.
    assert data["stars"] == {"a/one": 42, "b/two": 9}
    assert calls == ["https://api.github.com/repos/a/one", "https://api.github.com/repos/b/two"]
    assert data["fetched_at"] > old
