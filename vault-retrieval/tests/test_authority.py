"""Tests for authority ranking and freshness resolution.

Covers acceptance tests #10 (higher authority beats newer lower; same-rank
conflict -> data_hold with both sources) and #11 (expired review_by ->
stale-source data_hold; last_updated beats mtime when both exist).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from textwrap import dedent

import pytest


from vault_retrieval.authority import (
    AuthorityRank,
    Freshness,
    SourceMetadata,
    rank_source,
    rank_candidates,
    freshness_value,
    is_expired,
    better_authority,
    detect_conflict,
    parse_frontmatter,
)


def _write_with_frontmatter(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


class TestParseFrontmatter:
    def test_parses_minimal_frontmatter(self, tmp_path):
        p = tmp_path / "note.md"
        _write_with_frontmatter(p, dedent("""\
            ---
            title: Sample
            owner: tech-cto
            last_updated: 2026-10-01T10:00:00+07:00
            review_by: 2026-12-01
            ---
            body
        """))
        fm = parse_frontmatter(p)
        assert fm["owner"] == "tech-cto"
        assert fm["last_updated"] == "2026-10-01T10:00:00+07:00"
        assert fm["review_by"] == "2026-12-01"

    def test_missing_frontmatter_returns_empty(self, tmp_path):
        p = tmp_path / "note.md"
        p.write_text("just body\n", encoding="utf-8")
        fm = parse_frontmatter(p)
        assert fm == {}

    def test_no_separator_returns_empty(self, tmp_path):
        p = tmp_path / "note.md"
        p.write_text("title: x\n---\nbody\n", encoding="utf-8")
        # No closing --- at start means we treat as no frontmatter
        fm = parse_frontmatter(p)
        assert fm == {}


class TestFreshness:
    def test_expired_review_by(self):
        # review_by in the past -> expired
        assert is_expired("2020-01-01") is True

    def test_future_review_by_not_expired(self):
        assert is_expired("2099-01-01") is False

    def test_last_updated_beats_mtime(self, tmp_path):
        p = tmp_path / "note.md"
        p.write_text("hi\n", encoding="utf-8")
        fm = parse_frontmatter(p)
        # last_updated absent -> fall back to mtime
        f1 = freshness_value(p, fm, now=datetime(2026, 10, 1, tzinfo=timezone.utc))
        # last_updated present (mtime is the same physical file) -> prefer last_updated
        p2 = tmp_path / "note2.md"
        p2.write_text("hi\n", encoding="utf-8")
        # set the file mtime backward to make mtime older than last_updated
        import os
        old = (datetime(2020, 1, 1).timestamp())
        os.utime(p2, (old, old))
        fm2 = parse_frontmatter(p2)
        fm2["last_updated"] = "2026-09-01T10:00:00+07:00"
        f2 = freshness_value(p2, fm2, now=datetime(2026, 10, 1, tzinfo=timezone.utc))
        # The mtime is 2020, last_updated is 2026-09; freshness_value should
        # return last_updated (newer is better).
        assert f2 > f1


class TestAuthorityRank:
    def test_owner_decision_is_highest(self):
        sm = SourceMetadata(
            path="x.md",
            range_text="lines 1-2",
            characters_read=10,
            authority_rank=AuthorityRank.OWNER_DECISION,
            freshness=Freshness(basis="mtime", value="2026-10-01T00:00:00+00:00", stale=False),
        )
        assert sm.authority_rank == AuthorityRank.OWNER_DECISION

    def test_rank_ordering(self):
        ranks = [
            AuthorityRank.HISTORY,
            AuthorityRank.NEWER_EVIDENCE,
            AuthorityRank.APPROVED_SPEC,
            AuthorityRank.CANONICAL_REGISTER,
            AuthorityRank.OWNER_DECISION,
        ]
        ranks_sorted = sorted(ranks, key=lambda r: r.value)
        assert ranks_sorted[0] == AuthorityRank.OWNER_DECISION
        assert ranks_sorted[-1] == AuthorityRank.HISTORY

    def test_better_authority_returns_higher_rank(self):
        owner = SourceMetadata(
            path="o.md", range_text="lines 1-2", characters_read=10,
            authority_rank=AuthorityRank.OWNER_DECISION,
            freshness=Freshness(basis="mtime", value="2026-10-01T00:00:00+00:00", stale=False),
        )
        spec = SourceMetadata(
            path="s.md", range_text="lines 1-2", characters_read=10,
            authority_rank=AuthorityRank.APPROVED_SPEC,
            freshness=Freshness(basis="mtime", value="2026-10-02T00:00:00+00:00", stale=False),
        )
        # Owner decision is higher authority than newer spec
        assert better_authority(owner, spec) == owner
        assert better_authority(spec, owner) == owner


class TestDetectConflict:
    def test_same_rank_same_path_no_conflict(self):
        owner = SourceMetadata(
            path="o.md", range_text="lines 1-2", characters_read=10,
            authority_rank=AuthorityRank.OWNER_DECISION,
            freshness=Freshness(basis="mtime", value="2026-10-01T00:00:00+00:00", stale=False),
        )
        owner2 = SourceMetadata(
            path="o.md", range_text="lines 3-4", characters_read=10,
            authority_rank=AuthorityRank.OWNER_DECISION,
            freshness=Freshness(basis="mtime", value="2026-10-01T00:00:00+00:00", stale=False),
        )
        # No conflict because same source — complementary ranges
        assert detect_conflict(owner, owner2) is None

    def test_same_rank_different_paths_conflict(self):
        a = SourceMetadata(
            path="a.md", range_text="lines 1-2", characters_read=10,
            authority_rank=AuthorityRank.OWNER_DECISION,
            freshness=Freshness(basis="mtime", value="2026-10-01T00:00:00+00:00", stale=False),
        )
        b = SourceMetadata(
            path="b.md", range_text="lines 1-2", characters_read=10,
            authority_rank=AuthorityRank.OWNER_DECISION,
            freshness=Freshness(basis="mtime", value="2026-10-01T00:00:00+00:00", stale=False),
        )
        conflict = detect_conflict(a, b)
        assert conflict is not None
        assert "a.md" in conflict.paths
        assert "b.md" in conflict.paths


class TestRankCandidates:
    def test_returns_ranked_unique_list(self):
        owner = SourceMetadata(
            path="o.md", range_text="lines 1-2", characters_read=10,
            authority_rank=AuthorityRank.OWNER_DECISION,
            freshness=Freshness(basis="mtime", value="2026-10-01T00:00:00+00:00", stale=False),
        )
        spec = SourceMetadata(
            path="s.md", range_text="lines 1-2", characters_read=10,
            authority_rank=AuthorityRank.APPROVED_SPEC,
            freshness=Freshness(basis="mtime", value="2026-10-02T00:00:00+00:00", stale=False),
        )
        ranked = rank_candidates([spec, owner])
        assert ranked[0].path == "o.md"
        assert ranked[1].path == "s.md"


# ---------------------------------------------------------------------------
# Acceptance #10: higher authority beats newer lower authority;
# acceptance #11: expired review_by returns stale-source data_hold;
# last_updated beats mtime when both exist.
# ---------------------------------------------------------------------------


from datetime import datetime, timezone


class TestAcceptance10HigherAuthorityBeatsNewerLower:
    def test_owner_beats_newer_spec(self):
        # Owner decision from earlier timestamp
        owner = SourceMetadata(
            path="owner.md", range_text="lines 1-5", characters_read=200,
            authority_rank=AuthorityRank.OWNER_DECISION,
            freshness=Freshness(basis="mtime", value="2026-09-01T00:00:00+00:00", stale=False),
        )
        # Newer spec from later timestamp
        spec = SourceMetadata(
            path="spec.md", range_text="lines 1-5", characters_read=200,
            authority_rank=AuthorityRank.APPROVED_SPEC,
            freshness=Freshness(basis="mtime", value="2026-10-01T00:00:00+00:00", stale=False),
        )
        # Even though spec is newer, owner wins
        assert better_authority(owner, spec) == owner
        assert better_authority(spec, owner) == owner

    def test_owner_beats_newer_evidence(self):
        owner = SourceMetadata(
            path="owner.md", range_text="lines 1-5", characters_read=200,
            authority_rank=AuthorityRank.OWNER_DECISION,
            freshness=Freshness(basis="mtime", value="2026-09-01T00:00:00+00:00", stale=False),
        )
        evidence = SourceMetadata(
            path="evidence.md", range_text="lines 1-5", characters_read=200,
            authority_rank=AuthorityRank.NEWER_EVIDENCE,
            freshness=Freshness(basis="mtime", value="2026-10-01T00:00:00+00:00", stale=False),
        )
        assert better_authority(owner, evidence) == owner


class TestAcceptance11ExpiredReviewBy:
    def test_expired_review_by_marks_stale(self):
        # review_by in the past → stale=True
        assert is_expired("2020-01-01") is True

    def test_future_review_by_not_expired(self):
        assert is_expired("2099-01-01") is False

    def test_missing_review_by_not_expired(self):
        assert is_expired(None) is False
        assert is_expired("") is False

    def test_freshness_prefers_review_by_when_not_expired(self, tmp_path):
        p = tmp_path / "fm.md"
        p.write_text(
            "---\n"
            "last_updated: 2026-10-01T10:00:00+07:00\n"
            "review_by: 2099-12-31\n"
            "---\n\nbody\n",
            encoding="utf-8",
        )
        fm = parse_frontmatter(p)
        f = freshness_value(p, fm)
        assert f.basis == "review_by"
        assert f.stale is False

    def test_freshness_falls_back_to_last_updated_when_review_by_expired(
        self, tmp_path
    ):
        p = tmp_path / "fm.md"
        p.write_text(
            "---\n"
            "last_updated: 2026-10-01T10:00:00+07:00\n"
            "review_by: 2020-01-01\n"
            "---\n\nbody\n",
            encoding="utf-8",
        )
        fm = parse_frontmatter(p)
        f = freshness_value(p, fm)
        assert f.basis == "last_updated"
        assert f.stale is True

    def test_last_updated_beats_mtime(self, tmp_path):
        # last_updated is in the future, mtime is now → last_updated wins
        p = tmp_path / "fm.md"
        p.write_text(
            "---\n"
            "last_updated: 2099-01-01T00:00:00+00:00\n"
            "---\n\nbody\n",
            encoding="utf-8",
        )
        fm = parse_frontmatter(p)
        f = freshness_value(p, fm)
        assert f.basis == "last_updated"
        # last_updated (2099) > mtime (now)
        assert f.as_datetime() > datetime.now(timezone.utc)


class TestAcceptance10UnresolvedConflict:
    def test_detect_conflict_returns_pair(self):
        a = SourceMetadata(
            path="a.md", range_text="lines 1-2", characters_read=10,
            authority_rank=AuthorityRank.CANONICAL_REGISTER,
            freshness=Freshness(basis="mtime", value="2026-10-01T00:00:00+00:00", stale=False),
        )
        b = SourceMetadata(
            path="b.md", range_text="lines 1-2", characters_read=10,
            authority_rank=AuthorityRank.CANONICAL_REGISTER,
            freshness=Freshness(basis="mtime", value="2026-10-01T00:00:00+00:00", stale=False),
        )
        c = detect_conflict(a, b)
        assert c is not None
        assert set(c.paths) == {"a.md", "b.md"}
        assert c.rank == AuthorityRank.CANONICAL_REGISTER
