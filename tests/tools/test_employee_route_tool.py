"""Tests for tools/employee_route_tool.py."""

import json
import os
from types import SimpleNamespace
from unittest.mock import patch

from gateway.config import GatewayConfig, Platform


def _make_config(*, static_routes=None):
    qq_cfg = SimpleNamespace(
        enabled=True,
        token=None,
        api_key=None,
        extra={"employee_routes": static_routes or []},
    )
    weixin_cfg = SimpleNamespace(
        enabled=True,
        token=None,
        api_key=None,
        extra={},
    )
    return SimpleNamespace(
        platforms={
            Platform.QQ_NAPCAT: qq_cfg,
            Platform.WEIXIN: weixin_cfg,
        }
    )


def test_list_routes_uses_session_platform_and_returns_effective_view(monkeypatch, tmp_path):
    monkeypatch.setattr("gateway.employee_route_store.get_hermes_home", lambda: tmp_path)

    from gateway.employee_route_store import set_employee_route
    from tools.employee_route_tool import employee_route_tool

    set_employee_route(
        Platform.QQ_NAPCAT,
        worker_name="铁柱",
        aliases=["老铁"],
        preloaded_skills=["frontend-design-pro"],
        match_modes=["explicit", "heuristic"],
        action_terms=["打磨"],
        updated_by="seed",
    )

    with patch(
        "gateway.config.load_gateway_config",
        return_value=_make_config(
            static_routes=[
                {
                    "worker_name": "阿旺",
                    "preloaded_skills": ["ops-skill"],
                }
            ]
        ),
    ), patch.dict(
        os.environ,
        {
            "HERMES_SESSION_PLATFORM": "qq_napcat",
            "HERMES_SESSION_CHAT_TYPE": "dm",
            "HERMES_SESSION_CHAT_ID": "179033731",
            "HERMES_SESSION_USER_ID": "179033731",
        },
        clear=False,
    ):
        result = json.loads(employee_route_tool({"action": "list_routes"}))

    assert result["success"] is True
    assert result["platform"] == "qq_napcat"
    assert [item["worker_name"] for item in result["routes"]] == ["铁柱"]
    assert [item["worker_name"] for item in result["effective_routes"]] == ["阿旺", "铁柱"]


def test_set_route_requires_admin_session(monkeypatch, tmp_path):
    monkeypatch.setattr("gateway.employee_route_store.get_hermes_home", lambda: tmp_path)

    from tools.employee_route_tool import employee_route_tool

    with patch(
        "gateway.config.load_gateway_config",
        return_value=_make_config(),
    ), patch.dict(
        os.environ,
        {
            "HERMES_SESSION_PLATFORM": "qq_napcat",
            "HERMES_SESSION_CHAT_TYPE": "dm",
            "HERMES_SESSION_CHAT_ID": "179033731",
            "HERMES_SESSION_USER_ID": "179033731",
            "HERMES_SESSION_IS_ADMIN": "false",
        },
        clear=False,
    ):
        result = json.loads(
            employee_route_tool(
                {
                    "action": "set_route",
                    "worker_name": "铁柱",
                    "preloaded_skills": ["frontend-design-pro"],
                }
            )
        )

    assert result["success"] is False
    assert "董事长" in result["error"]


def test_set_route_persists_route_and_overrides_static_effective_view(monkeypatch, tmp_path):
    monkeypatch.setattr("gateway.employee_route_store.get_hermes_home", lambda: tmp_path)

    from tools.employee_route_tool import employee_route_tool

    config = GatewayConfig.from_dict(
        {
            "platforms": {
                "qq_napcat": {
                    "enabled": True,
                    "extra": {
                        "employee_routes": [
                            {
                                "worker_name": "铁柱",
                                "aliases": ["旧铁柱"],
                                "preloaded_skills": ["old-skill"],
                            }
                        ]
                    },
                }
            }
        }
    )

    with patch("gateway.config.load_gateway_config", return_value=config), patch.dict(
        os.environ,
        {
            "HERMES_SESSION_PLATFORM": "qq_napcat",
            "HERMES_SESSION_CHAT_TYPE": "dm",
            "HERMES_SESSION_CHAT_ID": "179033731",
            "HERMES_SESSION_USER_ID": "179033731",
            "HERMES_SESSION_USER_NAME": "發發發",
            "HERMES_SESSION_IS_ADMIN": "true",
        },
        clear=False,
    ):
        result = json.loads(
            employee_route_tool(
                {
                    "action": "set_route",
                    "worker_name": "铁柱",
                    "aliases": ["老铁"],
                    "preloaded_skills": ["frontend-design-pro"],
                    "match_modes": ["explicit", "heuristic"],
                    "action_terms": ["打磨"],
                    "subject_terms": ["页面"],
                    "pain_terms": ["粗糙"],
                }
            )
        )

    assert result["success"] is True
    assert result["platform"] == "qq_napcat"
    assert result["route"]["worker_name"] == "铁柱"
    assert result["route"]["match_modes"] == ["explicit", "heuristic"]
    assert result["route"]["updated_by"] == "發發發(179033731)"
    assert result["effective_routes"] == [
        {
            "worker_name": "铁柱",
            "aliases": ["老铁"],
            "preloaded_skills": ["frontend-design-pro"],
            "match_modes": ["explicit", "heuristic"],
            "action_terms": ["打磨"],
            "subject_terms": ["页面"],
            "pain_terms": ["粗糙"],
        }
    ]


def test_clear_route_removes_dynamic_route(monkeypatch, tmp_path):
    monkeypatch.setattr("gateway.employee_route_store.get_hermes_home", lambda: tmp_path)

    from gateway.employee_route_store import set_employee_route
    from tools.employee_route_tool import employee_route_tool

    set_employee_route(
        Platform.WEIXIN,
        worker_name="阿旺",
        aliases=["旺财"],
        preloaded_skills=["ops-skill"],
        updated_by="seed",
    )

    with patch("gateway.config.load_gateway_config", return_value=_make_config()), patch.dict(
        os.environ,
        {
            "HERMES_SESSION_PLATFORM": "weixin",
            "HERMES_SESSION_CHAT_TYPE": "dm",
            "HERMES_SESSION_CHAT_ID": "wx-user",
            "HERMES_SESSION_USER_ID": "wx-user",
            "HERMES_SESSION_IS_ADMIN": "true",
        },
        clear=False,
    ):
        result = json.loads(
            employee_route_tool(
                {
                    "action": "clear_route",
                    "worker_name": "阿旺",
                }
            )
        )

    assert result["success"] is True
    assert result["platform"] == "weixin"
    assert result["cleared_route"]["worker_name"] == "阿旺"
    assert result["routes"] == []
    assert result["effective_routes"] == []
