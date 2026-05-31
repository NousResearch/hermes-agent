from tools import calendar_tool


def test_metadata_normalizes_backend_values_to_strings():
    metadata = calendar_tool._metadata(
        {
            "tenant_id": "tenant-1",
            "labels": ["vip", "first-visit"],
            "priority": 2,
            "active": True,
            "empty": None,
            "details": {"company": "SitioUno"},
        }
    )

    assert metadata == {
        "tenant_id": "tenant-1",
        "labels": '["vip", "first-visit"]',
        "priority": "2",
        "active": "True",
        "details": '{"company": "SitioUno"}',
    }


def test_calendar_list_events_falls_back_to_user_scoped_endpoint(monkeypatch):
    calls = []

    def fake_request(method, path, body=None, query=None):
        calls.append((method, path, body, query))
        if path == "/calendar/cal-1/events":
            return {"success": False, "status": 401, "error": "unauthorized"}
        return {"success": True, "status": 200, "data": {"events": []}}

    monkeypatch.setattr(calendar_tool, "_request", fake_request)

    result = calendar_tool.calendar_list_events("cal-1", 1000, 2000)

    assert result["success"] is True
    assert calls == [
        ("GET", "/calendar/cal-1/events", None, {"startTs": 1000, "endTs": 2000}),
        ("GET", "/user/calendar/cal-1/events", None, {"startTs": 1000, "endTs": 2000}),
    ]


def test_calendar_list_events_uses_user_scoped_endpoint_when_requested(monkeypatch):
    calls = []

    def fake_request(method, path, body=None, query=None):
        calls.append((method, path, body, query))
        return {"success": True, "status": 200, "data": {"events": []}}

    monkeypatch.setattr(calendar_tool, "_request", fake_request)

    result = calendar_tool.calendar_list_events("cal-1", 1000, 2000, actor_scope=True)

    assert result["success"] is True
    assert calls == [
        ("GET", "/user/calendar/cal-1/events", None, {"startTs": 1000, "endTs": 2000}),
    ]
