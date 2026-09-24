"""Cron tool responses keep user-authored Unicode readable (#26128)."""

import json

import pytest

import tools.cronjob_tools as cron_tools
from cron.jobs import create_job, get_job
from tools.registry import registry


@pytest.mark.parametrize("action", ["list", "update", "pause", "resume", "remove"])
def test_unicode_job_survives_store_and_tool_dispatch(tmp_path, monkeypatch, action):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(cron_tools, "_notify_provider_jobs_changed_safe", lambda: None)
    name = "每日简报 café 🌅"
    prompt = '总结今日天气；保留 "引号" 和路径 C:\\reports\n第二行。'
    job = create_job(prompt=prompt, schedule="every 1h", name=name, deliver="local")
    args = {"action": action, "job_id": job["id"]}
    if action == "update":
        args["prompt"] = prompt + "更新。"
    raw = registry.dispatch("cronjob_manage", args)
    payload = json.loads(raw)
    assert payload["success"] is True
    assert name in raw
    if action in {"list", "update"}:
        displayed = payload["jobs"][0] if action == "list" else payload["job"]
        assert displayed["prompt_preview"] == args.get("prompt", prompt)
        assert "总结今日天气" in raw
    if action != "remove":
        assert get_job(job["id"])["prompt"] == args.get("prompt", prompt)


@pytest.mark.parametrize("case", ["empty", "missing", "raised"])
def test_unicode_errors_and_empty_results_keep_json_contract(tmp_path, monkeypatch, case):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    name = "不存在的任务 🌅"
    args = {"action": "list"}
    if case == "missing":
        args = {"action": "update", "job_id": name, "prompt": "天气"}
    elif case == "raised":
        def fail(**kwargs):
            raise OSError(name)
        monkeypatch.setattr(cron_tools, "list_jobs", fail)
    raw = registry.dispatch("cronjob_manage", args)
    payload = json.loads(raw)
    if case == "empty":
        assert payload["success"] is True
        assert payload["jobs"] == []
    else:
        assert payload["success"] is False
        assert name in payload["error"]
        assert name in raw
