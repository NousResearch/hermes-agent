"""Tests for tools/skill_recommender.py."""

import pytest

from tools.skill_recommender import (
    _describe_index,
    _score_query,
    _tokenise,
    dismiss,
    recommend_for_recent_activity,
    recommend_skills,
)


def _catalog():
    return [
        {"name": "github-ops", "description": "GitHub end-to-end: auth, repos, issues, PRs.",
         "category": "software-development"},
        {"name": "github-ops-fix", "description": "Triage and fix open GitHub issues.",
         "category": "software-development"},
        {"name": "pdf", "description": "Create, read, merge, fill, and secure PDF files.",
         "category": "productivity"},
        {"name": "docker-management", "description": "Manage Docker containers, images, volumes.",
         "category": "devops"},
        {"name": "homelab", "description": "Home network and remote access configuration.",
         "category": "networking"},
    ]


def _empty_usage():
    return {}


def test_tokenise_lowercases_and_strips_punctuation():
    toks = _tokenise("GitHub, end-to-end: auth (repos) - issues!")
    assert "github" in toks
    assert "end-to-end" in toks
    assert "auth" in toks
    assert "repos" in toks
    assert "issues" in toks
    # Stopwords gone
    assert "to" not in toks
    assert "and" not in toks


def test_tokenise_drops_short_tokens():
    assert "ok" not in _tokenise("ok ok ok")
    # 'ai' is short and not in stopwords, but len<3 -> dropped


def test_recommend_skills_returns_matches_for_query():
    out = recommend_skills(
        "github repos auth",
        top_k=3,
        catalog_loader=_catalog,
        suppressed_loader=lambda: set(),
    )
    names = [r["name"] for r in out]
    assert names
    # github-ops must beat homelab (and any non-github skill)
    assert "github-ops" in names
    assert names[0] == "github-ops"


def test_recommend_skills_returns_empty_on_blank_query():
    assert recommend_skills("", catalog_loader=_catalog) == []
    assert recommend_skills("   ", catalog_loader=_catalog) == []


def test_recommend_skills_excludes_suppressed():
    out = recommend_skills(
        "github pr review",
        top_k=3,
        catalog_loader=_catalog,
        suppressed_loader=lambda: {"github-ops"},
    )
    names = [r["name"] for r in out]
    assert "github-ops" not in names
    # but github-ops-fix is still matchable
    assert "github-ops-fix" in names


def test_recommend_skills_respects_top_k():
    out = recommend_skills("github", top_k=1, catalog_loader=_catalog)
    assert len(out) == 1


def test_recommend_skills_includes_score_and_why():
    out = recommend_skills("github", top_k=1, catalog_loader=_catalog)
    assert "score" in out[0]
    assert "why" in out[0]
    assert isinstance(out[0]["score"], float)
    assert out[0]["why"]  # non-empty rationale


def test_recommend_skills_handles_missing_catalog_loader():
    # If the catalog loader returns [], we get [] back, never an exception
    out = recommend_skills("github", catalog_loader=lambda: [], suppressed_loader=lambda: set())
    assert out == []


def test_recommend_skills_handles_suppressed_loader_failure():
    def bad():
        raise RuntimeError("boom")
    out = recommend_skills("github", catalog_loader=_catalog, suppressed_loader=bad)
    # Caught by _safe_suppressed_loader's import path - but bad() isn't called there.
    # We should still get results because the default loader is used when None.
    # Calling bad() directly in the function should still be handled.
    names = [r["name"] for r in out]
    assert names  # default loader returns empty -> not what we want here
    # Instead test that the function NEVER raises, regardless:
    try:
        recommend_skills("github", catalog_loader=_catalog, suppressed_loader=bad)
    except Exception as e:
        pytest.fail(f"recommend_skills raised: {e!r}")


def test_score_query_higher_use_count_increases_rank():
    idx = _describe_index(_catalog())
    # Manually inflate docker-management use count
    idx["docker-management"]["use_count"] = 50
    idx["docker-management"]["days_since_used"] = 1
    # Query about docker
    scored = _score_query(_tokenise("docker"), idx, set(), history_weight=0.5)
    assert scored
    assert scored[0][0] == "docker-management"


def test_score_query_skips_suppressed():
    idx = _describe_index(_catalog())
    scored = _score_query(_tokenise("github"), idx, {"github-ops"}, history_weight=0.0)
    names = [row[0] for row in scored]
    assert "github-ops" not in names


def test_score_query_returns_empty_on_no_overlap():
    idx = _describe_index(_catalog())
    scored = _score_query(_tokenise("cooking recipes"), idx, set(), history_weight=0.0)
    assert scored == []


def test_recommend_for_recent_activity_uses_seeds():
    def catalog_loader():
        return _catalog()

    def usage_loader():
        return {
            "github-ops": {
                "use_count": 5,
                "view_count": 1,
                "last_used_at": "2026-08-30T00:00:00+00:00",
                "last_viewed_at": "2026-08-30T00:00:00+00:00",
            },
        }

    out = recommend_for_recent_activity(
        top_k=3,
        lookback_days=30,
        catalog_loader=catalog_loader,
        suppressed_loader=lambda: set(),
        usage_loader=usage_loader,
    )
    # Seeding from github-ops should pull in github-ops-fix
    names = [r["name"] for r in out]
    assert "github-ops-fix" in names
    # And never the seed itself
    assert "github-ops" not in names


def test_recommend_for_recent_activity_empty_when_no_recent_use():
    out = recommend_for_recent_activity(
        catalog_loader=_catalog,
        suppressed_loader=lambda: set(),
        usage_loader=lambda: {},
    )
    assert out == []


def test_recommend_for_recent_activity_respects_suppressed():
    def usage_loader():
        return {
            "github-ops": {
                "use_count": 5,
                "last_used_at": "2026-08-30T00:00:00+00:00",
            },
        }

    out = recommend_for_recent_activity(
        top_k=5,
        catalog_loader=_catalog,
        suppressed_loader=lambda: {"github-ops-fix"},
        usage_loader=usage_loader,
    )
    names = [r["name"] for r in out]
    assert "github-ops-fix" not in names


def test_dismiss_rejects_empty():
    ok, reason = dismiss("")
    assert ok is False
    assert "empty" in reason.lower()


def test_dismiss_calls_suppression_registry(monkeypatch):
    """dismiss() must delegate to skill_usage.add_suppressed_name."""
    calls: list[str] = []

    def fake_add(name):
        calls.append(name)

    monkeypatch.setattr(
        "tools.skill_usage.add_suppressed_name", fake_add, raising=False
    )
    ok, reason = dismiss("pdf")
    assert ok is True
    assert reason == ""
    assert calls == ["pdf"]


def test_dismiss_returns_error_on_failure(monkeypatch):
    def bad(name):
        raise RuntimeError("disk full")

    monkeypatch.setattr(
        "tools.skill_usage.add_suppressed_name", bad, raising=False
    )
    ok, reason = dismiss("pdf")
    assert ok is False
    assert "disk full" in reason
