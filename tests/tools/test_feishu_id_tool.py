import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from gateway.config import Platform
from gateway.session import SessionEntry, SessionSource
from tools.feishu_id_tool import feishu_id_tool


class _FakeResponse:
    def __init__(self, *, success: bool, data=None, code=0, msg="ok"):
        self._success = success
        self.data = data
        self.code = code
        self.msg = msg

    def success(self):
        return self._success


def _make_fake_client(
    *,
    user=None,
    chat=None,
    members=None,
    batch_users=None,
    search_chats=None,
):
    def _user_get(_request):
        if user is None:
            return _FakeResponse(success=False, code=404, msg="user not found")
        return _FakeResponse(success=True, data=SimpleNamespace(user=user))

    def _user_batch(_request):
        if batch_users is None:
            return _FakeResponse(success=False, code=404, msg="batch not found")
        return _FakeResponse(success=True, data=SimpleNamespace(items=batch_users))

    def _chat_get(_request):
        if chat is None:
            return _FakeResponse(success=False, code=404, msg="chat not found")
        return _FakeResponse(success=True, data=chat)

    def _chat_search(_request):
        if search_chats is None:
            return _FakeResponse(success=False, code=404, msg="search not found")
        return _FakeResponse(success=True, data=SimpleNamespace(items=search_chats))

    def _chat_members_get(_request):
        if members is None:
            return _FakeResponse(success=False, code=404, msg="members not found")
        return _FakeResponse(
            success=True,
            data=SimpleNamespace(
                items=members,
                page_token=None,
                has_more=False,
                member_total=str(len(members)),
            ),
        )

    return SimpleNamespace(
        contact=SimpleNamespace(
            v3=SimpleNamespace(
                user=SimpleNamespace(
                    get=_user_get,
                    batch=_user_batch,
                )
            )
        ),
        im=SimpleNamespace(
            v1=SimpleNamespace(
                chat=SimpleNamespace(
                    get=_chat_get,
                    search=_chat_search,
                ),
                chat_members=SimpleNamespace(get=_chat_members_get),
            )
        ),
    )


def _write_sessions(home: Path, payloads: list[dict]) -> None:
    sessions_dir = home / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().isoformat()
    data = {}
    for item in payloads:
        source = SessionSource(**item["source"])
        entry = SessionEntry(
            session_key=item["session_key"],
            session_id=item["session_id"],
            created_at=datetime.fromisoformat(item.get("created_at", now)),
            updated_at=datetime.fromisoformat(item.get("updated_at", now)),
            origin=source,
            display_name=source.chat_name,
            platform=source.platform,
            chat_type=source.chat_type,
        )
        data[item["session_key"]] = entry.to_dict()
    (sessions_dir / "sessions.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _make_config(accounts: dict[str, dict], default_account: str | None = None):
    configs = {}
    for account_id, item in accounts.items():
        configs[account_id] = SimpleNamespace(
            enabled=True,
            extra={"account_id": account_id, "admins": item.get("admins", [])},
            home_channel=SimpleNamespace(
                chat_id=item["home_chat_id"],
                name=item.get("home_name", item["home_chat_id"]),
            ) if item.get("home_chat_id") else None,
        )

    return SimpleNamespace(
        get_default_account_id=lambda platform: default_account if platform == Platform.FEISHU else None,
        get_platform_config=lambda platform, account_id=None: (
            configs.get(account_id or default_account) if platform == Platform.FEISHU else None
        ),
        iter_platform_account_configs=lambda platform: (
            list(configs.items()) if platform == Platform.FEISHU else []
        ),
        get_connected_platforms=lambda: {Platform.FEISHU},
        platforms={Platform.FEISHU: next(iter(configs.values())) if configs else None},
    )


class TestFeishuIdTool:
    def test_whois_user_prefers_official_api_and_merges_observed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_sessions(
            tmp_path,
            [
                {
                    "session_key": "agent:main:feishu[laok-personal]:dm:oc_dm",
                    "session_id": "sess-official-user",
                    "source": {
                        "platform": Platform.FEISHU,
                        "account_id": "laok-personal",
                        "chat_id": "oc_dm",
                        "chat_name": "Ethan",
                        "chat_type": "dm",
                        "user_id": "ou_ethan",
                        "user_id_alt": "on_ethan",
                        "user_name": "Ethan Observed",
                    },
                }
            ],
        )
        config = _make_config(
            {"laok-personal": {"home_chat_id": "oc_home"}},
            default_account="laok-personal",
        )
        fake_client = _make_fake_client(
            user=SimpleNamespace(
                open_id="ou_ethan",
                user_id="u_ethan",
                union_id="on_ethan",
                name="Ethan",
                nickname="E",
                en_name="Ethan Z",
                job_title="Operator",
                department_ids=["od_dept"],
                email="ethan@example.com",
                enterprise_email="ethan@corp.example.com",
                mobile="+8613800000000",
            )
        )

        with (
            patch("gateway.config.load_gateway_config", return_value=config),
            patch("tools.feishu_id_tool._build_official_client", return_value=(fake_client, None)),
        ):
            result = json.loads(
                feishu_id_tool(
                    {
                        "action": "whois_user",
                        "open_id": "ou_ethan",
                    }
                )
            )

        assert result["success"] is True
        assert result["source"] == "official_api"
        assert result["authoritative"] is True
        assert result["user"]["open_ids"] == ["ou_ethan"]
        assert result["user"]["user_ids"] == ["u_ethan"]
        assert result["user"]["union_ids"] == ["on_ethan"]
        assert "Ethan Observed" in result["user"]["names"]
        assert result["user"]["job_title"] == "Operator"
        assert result["user"]["email"] == "ethan@example.com"

    def test_whois_chat_returns_members_and_home_channel(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_sessions(
            tmp_path,
            [
                {
                    "session_key": "agent:main:feishu[laok-personal]:group:oc_team:ou_ethan",
                    "session_id": "sess-1",
                    "source": {
                        "platform": Platform.FEISHU,
                        "account_id": "laok-personal",
                        "chat_id": "oc_team",
                        "chat_name": "Team CLAIRE",
                        "chat_type": "group",
                        "user_id": "ou_ethan",
                        "user_id_alt": "on_ethan",
                        "user_name": "Ethan",
                    },
                }
            ],
        )
        config = _make_config(
            {"laok-personal": {"home_chat_id": "oc_team", "home_name": "Team CLAIRE"}},
            default_account="laok-personal",
        )

        with patch("gateway.config.load_gateway_config", return_value=config):
            result = json.loads(feishu_id_tool({"action": "whois_chat", "chat_id": "oc_team"}))

        assert result["success"] is True
        assert result["chat"]["account_id"] == "laok-personal"
        assert result["chat"]["is_home_channel"] is True
        assert "Team CLAIRE" in result["chat"]["names"]
        member = result["chat"]["observed_members"][0]
        assert member["open_ids"] == ["ou_ethan"]
        assert member["union_ids"] == ["on_ethan"]
        assert member["names"] == ["Ethan"]

    def test_members_prefers_official_api_and_enriches_member_ids(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_sessions(
            tmp_path,
            [
                {
                    "session_key": "agent:main:feishu[laok-personal]:group:oc_team:ou_ethan",
                    "session_id": "sess-official-members",
                    "source": {
                        "platform": Platform.FEISHU,
                        "account_id": "laok-personal",
                        "chat_id": "oc_team",
                        "chat_name": "Team CLAIRE",
                        "chat_type": "group",
                        "user_id": "ou_ethan",
                        "user_id_alt": "on_ethan",
                        "user_name": "Ethan Observed",
                    },
                }
            ],
        )
        config = _make_config(
            {"laok-personal": {"home_chat_id": "oc_team", "home_name": "Team CLAIRE"}},
            default_account="laok-personal",
        )
        fake_client = _make_fake_client(
            chat=SimpleNamespace(
                name="Team CLAIRE",
                description="Core room",
                chat_type="group",
                owner_id="ou_owner",
                owner_id_type="open_id",
                external=False,
                tenant_key="tenant_a",
                labels=["core"],
                user_count="2",
                bot_count="1",
                chat_status="normal",
            ),
            members=[
                SimpleNamespace(member_id="ou_ethan", name="Ethan", tenant_key="tenant_a"),
                SimpleNamespace(member_id="ou_zoe", name="Zoe", tenant_key="tenant_a"),
            ],
            batch_users=[
                SimpleNamespace(
                    open_id="ou_ethan",
                    user_id="u_ethan",
                    union_id="on_ethan",
                    name="Ethan",
                    job_title="Operator",
                    department_ids=["od_ops"],
                ),
                SimpleNamespace(
                    open_id="ou_zoe",
                    user_id="u_zoe",
                    union_id="on_zoe",
                    name="Zoe",
                    department_ids=["od_research"],
                ),
            ],
        )

        with (
            patch("gateway.config.load_gateway_config", return_value=config),
            patch("tools.feishu_id_tool._build_official_client", return_value=(fake_client, None)),
        ):
            result = json.loads(
                feishu_id_tool(
                    {
                        "action": "members",
                        "chat_id": "oc_team",
                    }
                )
            )

        assert result["success"] is True
        assert result["source"] == "official_api"
        assert result["authoritative"] is True
        assert result["chat"]["chat_id"] == "oc_team"
        assert result["chat"]["is_home_channel"] is True
        assert result["chat"]["member_count"] == 2
        assert result["member_total"] == 2
        assert result["members"][0]["open_ids"] == ["ou_ethan"]
        assert result["members"][0]["user_ids"] == ["u_ethan"]
        assert result["members"][0]["union_ids"] == ["on_ethan"]
        assert "Ethan Observed" in result["members"][0]["names"]

    def test_search_chats_prefers_official_api(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_sessions(tmp_path, [])
        config = _make_config(
            {"laok-personal": {"home_chat_id": "oc_team", "home_name": "Team CLAIRE"}},
            default_account="laok-personal",
        )
        fake_client = _make_fake_client(
            search_chats=[
                SimpleNamespace(
                    chat_id="oc_team",
                    name="Team CLAIRE",
                    description="Core room",
                    owner_id="ou_owner",
                    owner_id_type="open_id",
                    external=False,
                    tenant_key="tenant_a",
                    labels=["core"],
                    chat_status="normal",
                ),
                SimpleNamespace(
                    chat_id="oc_team_archive",
                    name="Team CLAIRE Archive",
                    description="Archive room",
                    owner_id="ou_owner",
                    owner_id_type="open_id",
                    external=False,
                    tenant_key="tenant_a",
                    labels=["archive"],
                    chat_status="normal",
                ),
            ]
        )

        with (
            patch("gateway.config.load_gateway_config", return_value=config),
            patch("tools.feishu_id_tool._build_official_client", return_value=(fake_client, None)),
        ):
            result = json.loads(
                feishu_id_tool(
                    {
                        "action": "search_chats",
                        "query": "Team CLAIRE",
                    }
                )
            )

        assert result["success"] is True
        assert result["source"] == "official_api"
        assert result["authoritative"] is True
        assert [item["chat_id"] for item in result["matches"]] == ["oc_team", "oc_team_archive"]
        assert result["matches"][0]["is_home_channel"] is True

    def test_resolve_user_by_name_returns_observed_ids(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_sessions(
            tmp_path,
            [
                {
                    "session_key": "agent:main:feishu[laok-personal]:dm:oc_dm",
                    "session_id": "sess-2",
                    "source": {
                        "platform": Platform.FEISHU,
                        "account_id": "laok-personal",
                        "chat_id": "oc_dm",
                        "chat_name": "oc_dm",
                        "chat_type": "dm",
                        "user_id": "ou_ethan",
                        "user_id_alt": "on_ethan",
                        "user_name": "Ethan",
                    },
                }
            ],
        )
        config = _make_config(
            {"laok-personal": {"home_chat_id": "oc_home"}},
            default_account="laok-personal",
        )

        with patch("gateway.config.load_gateway_config", return_value=config):
            result = json.loads(
                feishu_id_tool(
                    {
                        "action": "resolve_user",
                        "query": "eth",
                        "account_id": "laok-personal",
                    }
                )
            )

        assert result["success"] is True
        assert len(result["matches"]) == 1
        assert result["matches"][0]["names"] == ["Ethan"]
        assert result["matches"][0]["open_ids"] == ["ou_ethan"]
        assert result["matches"][0]["union_ids"] == ["on_ethan"]

    def test_session_lookup_infers_account_from_scoped_session_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_sessions(
            tmp_path,
            [
                {
                    "session_key": "agent:main:feishu[laok-gradients]:dm:oc_dm",
                    "session_id": "sess-3",
                    "source": {
                        "platform": Platform.FEISHU,
                        "chat_id": "oc_dm",
                        "chat_name": "oc_dm",
                        "chat_type": "dm",
                        "user_id": "ou_jiawen",
                        "user_name": "嘉文",
                    },
                }
            ],
        )
        config = _make_config(
            {
                "laok-personal": {"home_chat_id": "oc_home"},
                "laok-gradients": {"home_chat_id": "oc_grad"},
            },
            default_account="laok-personal",
        )

        with patch("gateway.config.load_gateway_config", return_value=config):
            result = json.loads(
                feishu_id_tool(
                    {
                        "action": "session_lookup",
                        "session_key": "agent:main:feishu[laok-gradients]:dm:oc_dm",
                    }
                )
            )

        assert result["success"] is True
        assert result["session"]["account_id"] == "laok-gradients"
        assert result["session"]["account_id_inferred"] is False

    def test_my_chats_uses_current_sender_context(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "feishu")
        monkeypatch.setenv("HERMES_SESSION_ACCOUNT_ID", "laok-personal")
        monkeypatch.setenv("HERMES_SESSION_USER_ID", "ou_ethan")
        monkeypatch.setenv("HERMES_SESSION_USER_ID_ALT", "on_ethan")
        monkeypatch.setenv("HERMES_SESSION_USER_NAME", "Ethan")
        _write_sessions(
            tmp_path,
            [
                {
                    "session_key": "agent:main:feishu[laok-personal]:group:oc_team:ou_ethan",
                    "session_id": "sess-4",
                    "source": {
                        "platform": Platform.FEISHU,
                        "account_id": "laok-personal",
                        "chat_id": "oc_team",
                        "chat_name": "Team CLAIRE",
                        "chat_type": "group",
                        "user_id": "ou_ethan",
                        "user_id_alt": "on_ethan",
                        "user_name": "Ethan",
                    },
                },
                {
                    "session_key": "agent:main:feishu[laok-personal]:dm:oc_dm",
                    "session_id": "sess-5",
                    "source": {
                        "platform": Platform.FEISHU,
                        "account_id": "laok-personal",
                        "chat_id": "oc_dm",
                        "chat_name": "oc_dm",
                        "chat_type": "dm",
                        "user_id": "ou_ethan",
                        "user_id_alt": "on_ethan",
                        "user_name": "Ethan",
                    },
                },
            ],
        )
        config = _make_config(
            {"laok-personal": {"home_chat_id": "oc_home"}},
            default_account="laok-personal",
        )

        with patch("gateway.config.load_gateway_config", return_value=config):
            result = json.loads(feishu_id_tool({"action": "my_chats"}))

        assert result["success"] is True
        chat_ids = [item["chat_id"] for item in result["chats"]]
        assert chat_ids == ["oc_dm", "oc_team"]

    def test_cross_account_query_denied_for_non_admin_feishu_session(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "feishu")
        monkeypatch.setenv("HERMES_SESSION_ACCOUNT_ID", "laok-personal")
        monkeypatch.setenv("HERMES_SESSION_USER_ID", "ou_viewer")
        _write_sessions(tmp_path, [])
        config = _make_config(
            {
                "laok-personal": {"home_chat_id": "oc_home", "admins": ["ou_admin"]},
                "laok-gradients": {"home_chat_id": "oc_grad"},
            },
            default_account="laok-personal",
        )

        with patch("gateway.config.load_gateway_config", return_value=config):
            result = json.loads(
                feishu_id_tool(
                    {
                        "action": "search_chats",
                        "query": "team",
                        "as_account_id": "laok-gradients",
                    }
                )
            )

        assert "error" in result
        assert "Cross-account feishu_id query denied" in result["error"]

    def test_cross_account_query_allowed_for_admin_and_scoped_to_requested_account(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "feishu")
        monkeypatch.setenv("HERMES_SESSION_ACCOUNT_ID", "laok-personal")
        monkeypatch.setenv("HERMES_SESSION_USER_ID", "u_owner")
        _write_sessions(tmp_path, [])
        config = _make_config(
            {
                "laok-personal": {"home_chat_id": "oc_home", "admins": ["u_owner"]},
                "laok-gradients": {"home_chat_id": "oc_grad"},
            },
            default_account="laok-personal",
        )

        clients = {
            "laok-personal": _make_fake_client(
                chat=SimpleNamespace(
                    name="Personal Chat",
                    description="personal",
                    chat_type="group",
                    owner_id="ou_owner",
                    owner_id_type="open_id",
                    external=False,
                    tenant_key="tenant_personal",
                    labels=["personal"],
                    chat_status="normal",
                    user_count="3",
                    bot_count="1",
                )
            ),
            "laok-gradients": _make_fake_client(
                chat=SimpleNamespace(
                    name="Team CLAIRE",
                    description="gradients",
                    chat_type="group",
                    owner_id="ou_owner",
                    owner_id_type="open_id",
                    external=False,
                    tenant_key="tenant_gradients",
                    labels=["team"],
                    chat_status="normal",
                    user_count="8",
                    bot_count="2",
                )
            ),
        }

        def _build_client(_config, account_id):
            return clients[account_id], None

        with (
            patch("gateway.config.load_gateway_config", return_value=config),
            patch("tools.feishu_id_tool._build_official_client", side_effect=_build_client),
        ):
            result = json.loads(
                feishu_id_tool(
                    {
                        "action": "whois_chat",
                        "chat_id": "oc_team_claire",
                        "as_account_id": "laok-gradients",
                    }
                )
            )

        assert result["success"] is True
        assert result["source"] == "official_api"
        assert result["chat"]["account_id"] == "laok-gradients"
        assert result["chat"]["names"] == ["Team CLAIRE"]
        assert result["chat"]["tenant_key"] == "tenant_gradients"
