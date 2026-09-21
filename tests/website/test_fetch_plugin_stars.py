"""fetch-plugin-stars.py: plugin-catalog star counts, GitHub consulted only from the scheduled run.

The contract under test is rate-limit discipline and cache freshness: a deploy (no ``--probe``)
must never reach GitHub, the scheduled probe starts with one GraphQL request, and unresolved
repos use REST without making stale counts look fresh.
"""

from __future__ import annotations

import importlib.util
import json
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
    import yaml

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


def test_probe_starts_with_one_graphql_request_and_rest_fills_missing_counts(mod, tmp_path, monkeypatch):
    cat = _catalog(tmp_path, "https://github.com/a/one", "https://github.com/b/two", "https://gitlab.com/c/three")
    out = tmp_path / "plugin-stars.json"
    out.write_text(json.dumps({"fetched_at": "2026-01-01T00:00:00+00:00", "stars": {"a/one": 7, "b/two": 9}}),
                   encoding="utf-8")
    calls: list[str] = []

    def one_request(query, token):
        calls.append(f"graphql:{query}")
        return {"data": {"r0": {"stargazerCount": 42}, "r1": None},
                "errors": [{"message": "Could not resolve to a Repository"}]}
    monkeypatch.setattr(mod, "_graphql", one_request)

    def rest(url, headers):
        calls.append(f"rest:{url}")
        return {"stargazers_count": 21}
    monkeypatch.setattr(mod, "_http_json", rest)

    assert mod.main(catalog_dir=cat, output=out, probe=True, live_url=None, token="t") == 0
    data = json.loads(out.read_text())
    assert data["stars"] == {"a/one": 42, "b/two": 21}
    assert len(calls) == 2
    assert "gitlab" not in calls[0] and 'owner: "a"' in calls[0] and 'owner: "b"' in calls[0]
    assert calls[1].endswith("/repos/b/two")
    assert data["fetched_at"] > "2026-01-01"


@pytest.mark.parametrize(
    "failure_mode", ["exception", "no_data", "malformed_errors", "malformed_payload"],
)
def test_failed_graphql_probe_stops_fallback_and_keeps_cache(
    mod, tmp_path, monkeypatch, capsys, failure_mode,
):
    cat = _catalog(tmp_path, "https://github.com/a/one", "https://github.com/b/two")
    out = tmp_path / "plugin-stars.json"
    timestamp = "2026-01-01T00:00:00+00:00"
    out.write_text(json.dumps({
        "fetched_at": timestamp,
        "stars": {"a/one": 7, "retired/old": 99},
    }), encoding="utf-8")

    def limited(query, token):
        if failure_mode == "exception":
            raise urllib.error.HTTPError("u", 401, "unauthorized", hdrs=None, fp=None)
        if failure_mode == "no_data":
            return {"data": None, "errors": [None]}
        if failure_mode == "malformed_errors":
            return {"data": None, "errors": 1}
        return None
    monkeypatch.setattr(mod, "_graphql", limited)

    def unexpected_rest(*args, **kwargs):
        raise AssertionError("a wholesale GraphQL failure must not fan out into REST requests")
    monkeypatch.setattr(mod, "_http_json", unexpected_rest)

    assert mod.main(catalog_dir=cat, output=out, probe=True, live_url=None, token="t") == 0
    data = json.loads(out.read_text())
    assert data == {"fetched_at": timestamp, "stars": {"a/one": 7}}
    assert "::warning::Plugin star probe incomplete; missing cached counts for: b/two" in capsys.readouterr().err


def test_rest_fallback_stops_after_rate_limit_and_ignores_malformed_payload(
    mod, tmp_path, monkeypatch,
):
    cat = _catalog(
        tmp_path,
        "https://github.com/a/one",
        "https://github.com/b/two",
        "https://github.com/c/three",
    )
    out = tmp_path / "plugin-stars.json"
    timestamp = "2026-01-01T00:00:00+00:00"
    out.write_text(json.dumps({
        "fetched_at": timestamp,
        "stars": {"a/one": 1, "b/two": 2, "c/three": 3},
    }), encoding="utf-8")
    monkeypatch.setattr(mod, "_graphql", lambda query, token: {
        "data": {"r0": None, "r1": None, "r2": None},
    })
    calls: list[str] = []

    def rest(url, headers):
        calls.append(url)
        if len(calls) == 1:
            return []
        raise urllib.error.HTTPError(url, 429, "rate limited", hdrs=None, fp=None)
    monkeypatch.setattr(mod, "_http_json", rest)

    assert mod.main(catalog_dir=cat, output=out, probe=True, live_url=None, token="t") == 0
    assert len(calls) == 2
    assert json.loads(out.read_text()) == {
        "fetched_at": timestamp,
        "stars": {"a/one": 1, "b/two": 2, "c/three": 3},
    }
