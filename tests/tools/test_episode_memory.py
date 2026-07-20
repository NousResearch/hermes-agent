"""Tests for tools/episode_memory.py — evidence-gated episodes + FTS recall."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools.episode_memory import (
    EPISODE_SCHEMA,
    episode_tool,
    get_episode,
    list_episodes,
    recall_episodes,
    rebuild_episode_index,
    remember_episode,
)


@pytest.fixture
def episode_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Avoid config.yaml lookups pulling real user config mid-test.
    monkeypatch.setattr(
        "tools.episode_memory._episodes_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "tools.episode_memory._corpus_roots",
        lambda: ("episodes", "memories"),
    )
    return home


class TestEpisodeSchema:
    def test_requires_evidence_in_description(self):
        desc = EPISODE_SCHEMA["description"].lower()
        assert "evidence" in desc
        assert "not injected" in desc or "not" in desc and "system prompt" in desc
        assert "session_search" in desc
        assert "honcho" in desc  # explicit non-dependency

    def test_actions_listed(self):
        actions = EPISODE_SCHEMA["parameters"]["properties"]["action"]["enum"]
        for name in ("remember", "recall", "list", "get", "reindex"):
            assert name in actions


class TestRemember:
    def test_refuses_without_evidence(self, episode_home):
        result = remember_episode(
            content="Gateway reconnect needs backoff",
            evidence="",
        )
        assert result["success"] is False
        assert "evidence" in result["error"].lower()

    def test_refuses_empty_content(self, episode_home):
        result = remember_episode(content="  ", evidence="abc123")
        assert result["success"] is False

    def test_writes_markdown_and_indexes(self, episode_home):
        result = remember_episode(
            content="Gateway reconnect flakes when the PTY bridge races approval prompts.",
            evidence="commit deadbeef",
            source="review",
            tags=["gateway", "desktop"],
        )
        assert result["success"] is True
        path = Path(result["path"])
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert "evidence: commit deadbeef" in text
        assert "source: review" in text
        assert "Gateway reconnect flakes" in text

        hits = recall_episodes("gateway reconnect flakes", k=3)
        assert hits["success"] is True
        assert hits["count"] >= 1
        assert any("reconnect" in h["snippet"].lower() or "gateway" in h["path"].lower()
                   for h in hits["hits"])

    def test_status_shaped_warns_but_writes(self, episode_home):
        result = remember_episode(
            content="Phase 3 done and queue is at zero for the sprint.",
            evidence="/tmp/log.txt",
        )
        assert result["success"] is True
        assert result.get("warnings")

    def test_secret_shaped_refused(self, episode_home):
        result = remember_episode(
            content="token ghp_abcdefghijklmnopqrstuvwxyz0123456789 is bad",
            evidence="note",
        )
        assert result["success"] is False
        assert "secret" in result["error"].lower()


class TestRecallAndList:
    def test_recall_empty_index_message(self, episode_home):
        result = recall_episodes("nothing here yet")
        assert result["success"] is True
        assert result["hits"] == []

    def test_list_and_get(self, episode_home):
        written = remember_episode(
            content="Desktop updater must preserve local commits on macOS.",
            evidence="PR #64576",
            source="agent",
        )
        listed = list_episodes(limit=5)
        assert listed["count"] >= 1
        got = get_episode(written["path"])
        assert got["success"] is True
        assert "updater" in got["content"].lower()
        assert got["meta"].get("evidence") == "PR #64576"

    def test_get_rejects_path_outside_home(self, episode_home, tmp_path):
        outsider = tmp_path / "outside.md"
        outsider.write_text("nope", encoding="utf-8")
        got = get_episode(str(outsider))
        assert got["success"] is False

    def test_indexes_memories_root(self, episode_home):
        mem = episode_home / "memories"
        mem.mkdir(parents=True, exist_ok=True)
        (mem / "MEMORY.md").write_text(
            "Studio runs hermes gateway via launchd label ai.hermes.gateway\n",
            encoding="utf-8",
        )
        rebuild_episode_index()
        hits = recall_episodes("launchd hermes gateway", k=5)
        assert hits["success"] is True
        assert hits["count"] >= 1
        assert any(h["source"] == "memories" for h in hits["hits"])


class TestEpisodeToolDispatch:
    def test_tool_json_remember_and_recall(self, episode_home):
        raw = episode_tool(
            action="remember",
            content="Toolset panel flake is a Radix menu timing class under CI contention.",
            evidence="https://github.com/NousResearch/hermes-agent/actions/runs/29725368589",
            source="agent",
            tags="desktop,ci",
        )
        data = json.loads(raw)
        assert data["success"] is True

        raw2 = episode_tool(action="recall", query="Radix menu timing", k=3)
        data2 = json.loads(raw2)
        assert data2["success"] is True
        assert data2["count"] >= 1

    def test_unknown_action(self, episode_home):
        data = json.loads(episode_tool(action="explode"))
        assert data["success"] is False
