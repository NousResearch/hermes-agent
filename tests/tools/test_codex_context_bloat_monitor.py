import json
import sqlite3

from tools.codex_context_bloat_monitor import collect_metrics


def _write_state_db(tmp_path, rollout, *, tokens_used=900000, thread_id="t1"):
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
        (thread_id, "token incident", tokens_used, str(rollout)),
    )
    conn.commit()
    conn.close()


def test_collect_metrics_flags_context_bloat_from_real_token_count_shape(tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text(
        json.dumps(
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {
                            "input_tokens": 978420,
                            "cached_input_tokens": 800000,
                            "output_tokens": 20,
                            "total_tokens": 990000,
                        },
                        "last_token_usage": {
                            "input_tokens": 70000,
                            "cached_input_tokens": 65000,
                            "output_tokens": 10,
                            "total_tokens": 70010,
                        },
                    },
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "tool": "get_app_state",
                "output": "data:image/png;base64," + ("A" * 2000),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _write_state_db(tmp_path, rollout)

    metrics = collect_metrics(tmp_path, limit=10)

    assert len(metrics) == 1
    assert metrics[0].last_input_max == 70000
    assert metrics[0].total_input_tokens == 978420
    assert metrics[0].total_cached_input_tokens == 800000
    assert metrics[0].total_tokens_used == 990000
    reasons = metrics[0].incidents()
    assert "tokens_used" in reasons
    assert "last_input" in reasons
    assert "image_base64_chars" in reasons


def test_collect_metrics_does_not_count_thread_reads_as_computer_use(tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    large_read_thread_output = "get_app_state mentioned in old thread preview " + ("x" * 9000)
    rollout.write_text(
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "call_id": "call-read",
                    "name": "read_thread",
                    "arguments": "{}",
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call-read",
                    "output": large_read_thread_output,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_state_db(tmp_path, rollout, tokens_used=1)

    [metric] = collect_metrics(tmp_path, limit=10)

    assert metric.computer_use_output_max_chars == 0
    assert metric.thread_read_output_max_chars == len(large_read_thread_output)
    assert "computer_use_inline_chars" not in metric.incidents()


def test_collect_metrics_classifies_tool_outputs_by_call_id(tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    lines = [
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "call_id": "call-shell",
                "name": "exec_command",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call-shell",
                "output": "s" * 4000,
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "call_id": "call-ui",
                "name": "get_app_state",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call-ui",
                "output": "u" * 9000,
            },
        },
    ]
    rollout.write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n",
        encoding="utf-8",
    )
    _write_state_db(tmp_path, rollout, tokens_used=1)

    [metric] = collect_metrics(tmp_path, limit=10)

    assert metric.tool_call_count == 2
    assert metric.tool_output_count == 2
    assert metric.shell_output_max_chars == 4000
    assert metric.computer_use_output_max_chars == 9000
    assert metric.tool_output_max_chars == 9000
    assert "computer_use_inline_chars" in metric.incidents()


def test_collect_metrics_can_analyze_single_rollout_path(tmp_path):
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text(
        json.dumps(
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {
                            "input_tokens": 100,
                            "cached_input_tokens": 20,
                            "total_tokens": 110,
                        },
                        "last_token_usage": {"input_tokens": 80, "total_tokens": 85},
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    [metric] = collect_metrics(tmp_path, rollout_path=rollout, thread_id="direct")

    assert metric.thread_id == "direct"
    assert metric.last_input_max == 80
    assert metric.total_tokens_used == 110
