import json
import sqlite3

from tools.codex_context_bloat_monitor import collect_metrics


def test_collect_metrics_flags_context_bloat(tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text(
        json.dumps({"event": "usage", "last_input": 70000}) + "\n"
        + json.dumps(
            {
                "tool": "get_app_state",
                "output": "data:image/png;base64," + ("A" * 2000),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    db = tmp_path / "state_5.sqlite"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE threads (
            id TEXT,
            title TEXT,
            tokens_used INTEGER,
            rollout_path TEXT,
            archived INTEGER,
            updated_at_ms INTEGER
        )
        """
    )
    conn.execute(
        "INSERT INTO threads VALUES (?, ?, ?, ?, 0, 1)",
        ("t1", "token incident", 900000, str(rollout)),
    )
    conn.commit()
    conn.close()

    metrics = collect_metrics(tmp_path, limit=10)

    assert len(metrics) == 1
    reasons = metrics[0].incidents()
    assert "tokens_used" in reasons
    assert "last_input" in reasons
    assert "image_base64_chars" in reasons

