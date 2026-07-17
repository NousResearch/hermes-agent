"""Tests for the TrainingDataBridge."""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from agent.training_data import TrainingDataBridge


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def test_bridge_init(temp_dir):
    bridge = TrainingDataBridge(data_dir=temp_dir)
    stats = bridge.get_stats()
    assert stats["total_records"] == 0


def test_export_empty(temp_dir):
    bridge = TrainingDataBridge(data_dir=temp_dir)
    result = bridge.export()
    assert "error" in result


def test_export_sharegpt(temp_dir):
    bridge = TrainingDataBridge(data_dir=temp_dir)
    record = {
        "conversations": [{"from": "human", "value": "test"}, {"from": "gpt", "value": "ok"}],
        "completed": True,
        "metadata": {"score": 0.85, "task_domain": "software-dev"},
    }
    test_file = temp_dir / "test.jsonl"
    test_file.write_text(json.dumps(record))
    result = bridge.export(fmt="sharegpt")
    assert "error" not in result
    assert result["records_exported"] == 1
    assert Path(result["output_path"]).exists()


def test_deduplicate(temp_dir):
    bridge = TrainingDataBridge(data_dir=temp_dir)
    record = {
        "conversations": [{"from": "human", "value": "test"}, {"from": "gpt", "value": "ok"}],
        "completed": True,
    }
    test_file = temp_dir / "dup.jsonl"
    test_file.write_text(json.dumps(record) + "\n" + json.dumps(record))
    result = bridge.export()
    assert result["records_exported"] == 1  # Deduplicated


def test_format_alpaca(temp_dir):
    bridge = TrainingDataBridge(data_dir=temp_dir)
    record = {
        "conversations": [{"from": "human", "value": "hello"}, {"from": "gpt", "value": "hi"}],
        "completed": True,
    }
    (temp_dir / "test.jsonl").write_text(json.dumps(record))
    result = bridge.export(fmt="alpaca")
    assert "error" not in result
    content = Path(result["output_path"]).read_text()
    assert "hello" in content
    assert "hi" in content


def test_min_score_filter(temp_dir):
    bridge = TrainingDataBridge(data_dir=temp_dir)
    high = {"conversations": [{"from":"human","value":"a"},{"from":"gpt","value":"b"}],
            "completed": True, "metadata": {"score": 0.9}}
    low = {"conversations": [{"from":"human","value":"c"},{"from":"gpt","value":"d"}],
           "completed": True, "metadata": {"score": 0.2}}
    (temp_dir / "test.jsonl").write_text(json.dumps(high) + "\n" + json.dumps(low))
    result = bridge.export(min_score=0.5)
    assert result["records_exported"] == 1


def test_domain_filter(temp_dir):
    bridge = TrainingDataBridge(data_dir=temp_dir)
    r1 = {"conversations": [{"from":"human","value":"x"},{"from":"gpt","value":"y"}],
          "completed": True, "metadata": {"task_domain": "security"}}
    r2 = {"conversations": [{"from":"human","value":"a"},{"from":"gpt","value":"b"}],
          "completed": True, "metadata": {"task_domain": "devops"}}
    (temp_dir / "test.jsonl").write_text(json.dumps(r1) + "\n" + json.dumps(r2))
    result = bridge.export(domain="security")
    assert result["records_exported"] == 1


def test_input_paths(temp_dir):
    ext = Path(temp_dir) / "external"
    ext.mkdir()
    record = {"conversations": [{"from":"human","value":"e"},{"from":"gpt","value":"f"}],
              "completed": True}
    (ext / "ext.jsonl").write_text(json.dumps(record))
    bridge = TrainingDataBridge(data_dir=temp_dir, input_paths=[ext])
    result = bridge.export()
    assert result["records_exported"] == 1
