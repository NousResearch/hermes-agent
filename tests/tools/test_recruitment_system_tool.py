from __future__ import annotations

import json

import tools.recruitment_system_tool as tool


def test_recruitment_query_uses_http_api(monkeypatch):
    monkeypatch.setenv("RECRUITMENT_API_BASE_URL", "http://recruitment.local")
    monkeypatch.setenv("RECRUITMENT_API_TENANT_ID", "1001")

    calls = []

    def fake_api_request(config, method, path, **kwargs):
        calls.append({"method": method, "path": path, **kwargs})
        return {
            "total": 1,
            "records": [
                {
                    "id": 1,
                    "jobCode": "JD-GO-001",
                    "jobName": "高级golang开发工程师",
                    "department": "研发中心",
                    "employmentType": "FULL_TIME",
                    "workLocation": "上海",
                    "headcount": 1,
                    "ownerUserId": "hr_mgr_1001",
                    "ownerUserName": "HR经理",
                    "status": "ONLINE",
                }
            ],
        }

    monkeypatch.setattr(tool, "_api_request", fake_api_request)

    result = json.loads(tool._query_handler({"user_question": "当前正在招聘的岗位有哪些？"}))

    assert result["success"] is True
    assert "高级golang开发工程师" in result["answer"]
    assert calls[0]["method"] == "GET"
    assert calls[0]["path"] == "/api/v1/jobs"
    assert calls[0]["tenant_id"] == "1001"
    assert calls[0]["params"]["status"] == "ONLINE"


def test_recruitment_create_job_uses_http_api_sequence(monkeypatch):
    monkeypatch.setenv("RECRUITMENT_API_BASE_URL", "http://recruitment.local")
    monkeypatch.setenv("RECRUITMENT_API_TENANT_ID", "1001")

    calls = []

    def fake_api_request(config, method, path, **kwargs):
        calls.append({"method": method, "path": path, **kwargs})
        if method == "GET" and path == "/api/v1/jobs":
            return {"total": 0, "records": []}
        if method == "POST" and path == "/api/v1/jobs":
            assert kwargs["json_body"]["jobName"] == "高级golang开发工程师"
            return 13004
        if method == "PUT" and path == "/api/v1/jobs/13004/requirements":
            assert kwargs["json_body"]["educationRequirement"] == "UNLIMITED"
            return None
        if method == "POST" and path == "/api/v1/jobs/13004/online":
            return None
        if method == "GET" and path == "/api/v1/jobs/13004":
            return {
                "id": 13004,
                "jobCode": "JOB20260524150000000",
                "jobName": "高级golang开发工程师",
                "department": "研发中心",
                "employmentType": "FULL_TIME",
                "workLocation": "上海",
                "headcount": 1,
                "ownerUserId": "hr_mgr_1001",
                "ownerUserName": "HR经理",
                "status": "ONLINE",
            }
        raise AssertionError(f"unexpected API call: {method} {path}")

    monkeypatch.setattr(tool, "_api_request", fake_api_request)

    result = json.loads(tool._create_job_handler({"job_name": "高级golang开发工程师"}))

    assert result["success"] is True
    assert result["created"] is True
    assert result["data"][0]["job_name"] == "高级golang开发工程师"
    assert [(call["method"], call["path"]) for call in calls] == [
        ("GET", "/api/v1/jobs"),
        ("POST", "/api/v1/jobs"),
        ("PUT", "/api/v1/jobs/13004/requirements"),
        ("POST", "/api/v1/jobs/13004/online"),
        ("GET", "/api/v1/jobs/13004"),
    ]
