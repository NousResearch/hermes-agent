"""Deterministic low-token local repair helpers."""

import datetime as dt
import sqlite3
from zoneinfo import ZoneInfo

from tools.local_repair_tool import (
    collect_token_usage_report,
    collect_today_token_usage_report,
    format_token_usage_report,
    format_today_token_usage_report,
    repair_morning_briefing_operation_summary,
)


OLD_SCRIPT = '''import datetime as dt
import re
from pathlib import Path

OPS_DIR = Path(__file__).resolve().parent / "docs" / "operation summary"


def op_titles_for(day: dt.date) -> list[str]:
    prefix = day.strftime("%Y%m%d")
    titles: list[str] = []
    if not OPS_DIR.exists():
        return titles
    for path in sorted(OPS_DIR.glob(f"{prefix}_*.html")):
        stem = path.stem
        title = re.sub(r"^\\d{12}_", "", stem).replace("_", " ")
        titles.append(title)
    return titles[:10]


def summarize_cron(jobs):
    return [], []
'''


def _write_fixture(tmp_path):
    script = tmp_path / "morning-briefing-report.py"
    docs = tmp_path / "docs"
    ops = docs / "operation summary"
    ops.mkdir(parents=True)
    script.write_text(OLD_SCRIPT, encoding="utf-8")
    (ops / "202606041044_Antigravity反代故障诊断与开机自启设置.html").write_text(
        "<html></html>",
        encoding="utf-8",
    )
    return script, docs


def test_local_repair_dry_run_reports_fix_without_writing(tmp_path):
    script, docs = _write_fixture(tmp_path)

    result = repair_morning_briefing_operation_summary(
        script_path=script,
        docs_dir=docs,
        target_date=dt.date(2026, 6, 4),
        title_hint="Antigravity",
        dry_run=True,
    )

    assert result["success"] is True
    assert result["would_change"] is True
    assert result["changed"] is False
    assert 'OPS_DIR.glob(f"{prefix}_*.html")' in script.read_text(encoding="utf-8")


def test_local_repair_updates_script_and_verifies_wide_match(tmp_path):
    script, docs = _write_fixture(tmp_path)

    result = repair_morning_briefing_operation_summary(
        script_path=script,
        docs_dir=docs,
        target_date="2026-06-04",
        title_hint="Antigravity",
    )

    source = script.read_text(encoding="utf-8")
    assert result["success"] is True
    assert result["changed"] is True
    assert result["verify_titles"] == ["Antigravity反代故障诊断与开机自启设置"]
    assert 'OPS_DIR.glob(f"{prefix}*.html")' in source
    assert "prefix_dash" in source


def _write_state_db(tmp_path):
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT,
            user_id TEXT,
            model TEXT,
            model_config TEXT,
            system_prompt TEXT,
            parent_session_id TEXT,
            started_at REAL,
            ended_at REAL,
            end_reason TEXT,
            message_count INTEGER DEFAULT 0,
            tool_call_count INTEGER DEFAULT 0,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cache_read_tokens INTEGER DEFAULT 0,
            cache_write_tokens INTEGER DEFAULT 0,
            reasoning_tokens INTEGER DEFAULT 0,
            billing_provider TEXT,
            billing_base_url TEXT,
            billing_mode TEXT,
            estimated_cost_usd REAL,
            actual_cost_usd REAL,
            cost_status TEXT,
            cost_source TEXT,
            pricing_version TEXT,
            title TEXT,
            api_call_count INTEGER DEFAULT 0,
            handoff_state TEXT,
            handoff_platform TEXT,
            handoff_error TEXT,
            cwd TEXT,
            rewind_count INTEGER DEFAULT 0,
            archived INTEGER DEFAULT 0
        )
        """
    )
    tz = ZoneInfo("Asia/Shanghai")
    today_a = dt.datetime(2026, 6, 16, 8, 0, tzinfo=tz).timestamp()
    today_b = dt.datetime(2026, 6, 16, 9, 0, tzinfo=tz).timestamp()
    yesterday = dt.datetime(2026, 6, 15, 23, 59, tzinfo=tz).timestamp()
    old = dt.datetime(2026, 6, 14, 11, 59, tzinfo=tz).timestamp()
    conn.executemany(
        """
        INSERT INTO sessions (
            id, source, model, started_at, end_reason, message_count,
            tool_call_count, input_tokens, output_tokens, cache_read_tokens,
            estimated_cost_usd, cost_status, title, api_call_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("s-small", "feishu", "m", today_a, "done", 3, 1, 100, 10, 5, 0.01, "estimated", "Small", 2),
            ("s-large", "cron", "m", today_b, "done", 4, 2, 900, 30, 10, 0.02, "estimated", "Large", 3),
            ("s-old", "feishu", "m", yesterday, "done", 4, 2, 5000, 50, 0, 1.0, "estimated", "Old", 3),
            ("s-too-old", "feishu", "m", old, "done", 4, 2, 7000, 70, 0, 1.0, "estimated", "Too Old", 3),
        ],
    )
    conn.commit()
    conn.close()
    return db_path


def test_today_token_usage_report_sums_today_and_sorts_top_sessions(tmp_path):
    db_path = _write_state_db(tmp_path)
    now = dt.datetime(2026, 6, 16, 12, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    result = collect_today_token_usage_report(db_path=db_path, top_n=2, now=now)

    assert result["success"] is True
    assert result["session_count"] == 2
    assert result["input_tokens"] == 1000
    assert result["output_tokens"] == 40
    assert [item["session_id"] for item in result["top_sessions"]] == ["s-large", "s-small"]


def test_today_token_usage_report_can_filter_feishu_scope(tmp_path):
    db_path = _write_state_db(tmp_path)
    now = dt.datetime(2026, 6, 16, 12, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    result = collect_today_token_usage_report(
        db_path=db_path,
        scope="feishu",
        top_n=3,
        now=now,
    )
    text = format_today_token_usage_report(result)

    assert result["session_count"] == 1
    assert result["input_tokens"] == 100
    assert "飞书会话" in text
    assert "Small" in text
    assert "Old" not in text


def test_token_usage_report_can_target_yesterday_natural_day(tmp_path):
    db_path = _write_state_db(tmp_path)
    now = dt.datetime(2026, 6, 16, 12, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    result = collect_token_usage_report(
        db_path=db_path,
        target_date=dt.date(2026, 6, 15),
        top_n=5,
        now=now,
    )
    text = format_token_usage_report(result)

    assert result["session_count"] == 1
    assert result["input_tokens"] == 5000
    assert [item["session_id"] for item in result["top_sessions"]] == ["s-old"]
    assert "2026-06-15 00:00:00 至 2026-06-16 00:00:00" in text


def test_token_usage_report_accepts_string_target_date(tmp_path):
    db_path = _write_state_db(tmp_path)
    now = dt.datetime(2026, 6, 16, 12, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    result = collect_token_usage_report(
        db_path=db_path,
        target_date="2026-06-15",
        top_n=5,
        now=now,
    )

    assert result["session_count"] == 1
    assert result["input_tokens"] == 5000


def test_token_usage_report_supports_rolling_days(tmp_path):
    db_path = _write_state_db(tmp_path)
    now = dt.datetime(2026, 6, 16, 12, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    result = collect_token_usage_report(
        db_path=db_path,
        days=2,
        top_n=5,
        now=now,
        label="最近 2 天",
    )

    assert result["session_count"] == 3
    assert result["input_tokens"] == 6000
    assert [item["session_id"] for item in result["top_sessions"]] == ["s-old", "s-large", "s-small"]


def test_token_usage_report_supports_explicit_date_range(tmp_path):
    db_path = _write_state_db(tmp_path)
    now = dt.datetime(2026, 6, 16, 12, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    result = collect_token_usage_report(
        db_path=db_path,
        range_start="2026-06-15",
        range_end="2026-06-17",
        top_n=5,
        now=now,
    )

    assert result["session_count"] == 3
    assert result["input_tokens"] == 6000
    assert "s-too-old" not in [item["session_id"] for item in result["top_sessions"]]
