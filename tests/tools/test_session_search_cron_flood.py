"""Regression: cron sessions must not flood session_search discovery (#19434 follow-up).

Live symptom that motivated this fix: the model's own recall tool returned ONLY
cron sessions for everyday queries. Three independent causes, three fixes:

1. Discovery searched cron rows by default (they were merely DEMOTED below
   interactive hits). Cron runs are scheduled bulk output; excluded by default,
   opt back in with the exact token ``source:cron``.
2. The short-CJK LIKE fallback OR-matched ``m.tool_name`` / ``m.tool_calls``
   (paths, call JSON) in addition to content, and ANDed nothing — two-character
   queries matched half the database. LIKE fallback is now content-only with
   FTS boolean semantics, and it honours ``sort``.
3. Cron runs with ``skills=`` persisted the ENTIRE inlined skill text as the
   user message every fire, so each job out-vocabularied months of interactive
   history. The stored row is now a ``[skill-ref: ...]`` pointer stub; the agent
   still receives the full prompt.

The fixture ``fixtures/like_fallback_pre_fix_lines.txt`` captures the pre-fix
source lines from origin/main so the repro claim is anchored to real code, not
a retelling.
"""
import json
import time
from pathlib import Path

import pytest

from hermes_state import SessionDB
from hermes_state_search import _LIKE_ANY_COLUMN_SQL, _LIKE_COALESCED_COLUMN_SQL
from tools.session_search_tool import discover_query_and_excludes, session_search

FIXTURE = Path(__file__).parent / "fixtures" / "like_fallback_pre_fix_lines.txt"


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _seed_flood(db):
    """8 cron sessions stuffed with the query vocabulary + 1 older interactive hit."""
    now = int(time.time())
    db.create_session("s_human", source="cli")
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (now - 90000, "s_human"))
    db.append_message("s_human", role="user", content="venom 项目进展如何")
    db.append_message("s_human", role="assistant", content="venom 项目首里程碑已交付。")
    for i in range(8):
        sid = f"cron_{i}"
        db.create_session(sid, source="cron")
        db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (now - 1000 - i, sid))
        db.append_message(sid, role="user", content="venom 项目每日汇报")
        db.append_message(sid, role="assistant", content="venom 项目 venom 项目 venom 汇总")
    db._conn.commit()


def test_flood_repro_fixture_differs_from_current():
    """FLOOD_REPRO_OK — the pre-fix lines captured from origin/main must NOT all be
    present verbatim in the fixed source (i.e. the bug code is gone)."""
    pre_fix = FIXTURE.read_text(encoding="utf-8")
    assert "m.tool_calls LIKE" in pre_fix  # fixture really holds the old code
    current = Path(__file__).parents[2].joinpath("hermes_state_search.py").read_text(encoding="utf-8")
    assert "m.tool_calls LIKE ? ESCAPE" not in current


def test_flood_fixed_discovery_excludes_cron(db):
    """FLOOD_FIXED_OK — with cron flooding the corpus, discovery still returns the
    user's interactive session and zero cron rows."""
    _seed_flood(db)
    result = json.loads(session_search(query="venom 项目", limit=5, db=db))
    assert result["success"] is True
    sources = [r["source"] for r in result["results"]]
    assert "cron" not in sources
    assert result["results"][0]["session_id"] == "s_human"


def test_like_fallback_boolean_and_sort(db):
    """LIKE_BOOL_OK + SORT_OLDEST_OK — bare terms AND together, explicit OR works,
    tool_name/tool_calls are not matched, and sort='oldest' flips the order."""
    # Content-only constants: no tool columns anywhere in the LIKE predicates.
    assert "tool_name" not in _LIKE_ANY_COLUMN_SQL
    assert "tool_calls" not in _LIKE_ANY_COLUMN_SQL
    assert "tool_name" not in _LIKE_COALESCED_COLUMN_SQL

    # A tool-call-shaped term that lives ONLY in tool_calls/tool_name must not match.
    db.create_session("s_toolonly", source="cli")
    db.append_message("s_toolonly", role="user", content="完全无关的用户问题")
    db.append_message("s_toolonly", role="assistant", content="回答内容", tool_name="read_file")
    db._conn.commit()
    rows = db._search_messages_like_fallback("read_file", limit=10, offset=0, sort=None,
                                             include_inactive=False, source_filter=None,
                                             exclude_sources=None, role_filter=None)
    assert rows == []

    # Boolean: bare terms AND; explicit OR separates groups.
    db.create_session("s_bool", source="cli")
    db.append_message("s_bool", role="user", content="广西的桂林山水")
    db._conn.commit()
    and_rows = db._search_messages_like_fallback("广西 漓江", limit=10, offset=0, sort=None,
                                                 include_inactive=False, source_filter=None,
                                                 exclude_sources=None, role_filter=None)
    assert and_rows == []  # 漓江 absent -> AND fails
    or_rows = db._search_messages_like_fallback("广西 OR 漓江", limit=10, offset=0, sort=None,
                                                include_inactive=False, source_filter=None,
                                                exclude_sources=None, role_filter=None)
    assert len(or_rows) == 1

    # sort='oldest' flips chronological order in the LIKE path.
    db.create_session("s_sort", source="cli")
    db.append_message("s_sort", role="user", content="锚点词甲 第一次出现")
    db.append_message("s_sort", role="user", content="锚点词甲 第二次出现")
    db._conn.commit()
    newest = db._search_messages_like_fallback("锚点词甲", limit=10, offset=0, sort="newest",
                                               include_inactive=False, source_filter=None,
                                               exclude_sources=None, role_filter=None)
    oldest = db._search_messages_like_fallback("锚点词甲", limit=10, offset=0, sort="oldest",
                                               include_inactive=False, source_filter=None,
                                               exclude_sources=None, role_filter=None)
    assert [r["id"] for r in newest] == list(reversed([r["id"] for r in oldest]))


def test_cron_persist_stub_pointer():
    """SKILL_POINTER_OK — jobs with skills persist a pointer, not the skill body."""
    from cron.scheduler_prompt import _cron_persist_stub
    assembled = ('[IMPORTANT: The user has invoked the "ops" skill ... full skill content '
                 'is loaded below.]\n\n' + "X" * 4000)
    stub = _cron_persist_stub({"name": "nightly", "id": "j1", "skills": ["ops"],
                               "prompt": "run the nightly report"}, assembled)
    assert stub.startswith("[skill-ref: skills loaded for this run: ops")
    assert "The full skill content is loaded below" not in stub
    assert "[job: nightly]" in stub and "[instruction: run the nightly report]" in stub
    # Plain-prompt jobs persist unchanged.
    assert _cron_persist_stub({"name": "plain", "id": "j2"}, "hello") == "hello"


def test_discover_query_and_excludes_token_semantics():
    """The cron opt-in is an EXACT blank-separated token."""
    q, ex, named = discover_query_and_excludes("部署 进展")
    assert q == "部署 进展" and "cron" in ex and named is False
    q, ex, named = discover_query_and_excludes("source:cron 部署")
    assert q == "部署" and "cron" not in ex and named is True
    q, ex, named = discover_query_and_excludes("source:cron")
    assert q == "" and named is True
    # Embedded lookalikes are not tokens.
    q, ex, named = discover_query_and_excludes("source:cronfoo 部署")
    assert q == "source:cronfoo 部署" and "cron" in ex and named is False
