import pytest

from tools.tlon_tool import (
    TlonHttpError,
    TlonGroups,
    TlonHooks,
    TlonMessages,
    _encode_cord,
    _expand_cite_path,
    _format_post_id,
    _cite_to_url_path,
    _search_path,
)


class FakeTlonClient:
    ship_name = "~bot-palnet"
    ship_no_sig = "bot-palnet"

    def __init__(self):
        self.pokes = []
        self.threads = []
        self.scries = []

    async def poke(self, app, mark, json_data, **_kwargs):
        self.pokes.append({"app": app, "mark": mark, "json": json_data})
        return {"success": True}

    async def thread(self, **kwargs):
        self.threads.append(kwargs)
        return {"created": True}

    async def scry(self, app, path, **_kwargs):
        self.scries.append({"app": app, "path": path})
        if app == "groups" and path == "/v2/groups":
            return {
                "~host/group": {
                    "meta": {"title": "Test Group"},
                    "channels": {
                        "chat/~host/test": {"meta": {"title": "General"}},
                    },
                }
            }
        if app == "groups" and path.startswith("/v2/ui/groups/"):
            return {"roles": {}, "admins": []}
        if app == "channels-server" and path == "/v0/hooks":
            return {"hooks": {"0vabc": {"id": "0vabc", "name": "Old", "src": ":: old", "meta": {}}}}
        return {}


def test_encode_cord_matches_tlon_safe_text_shape():
    assert _encode_cord("some Chars!") == "~.some.~43.hars~21."
    assert _encode_cord("hello") == "~.hello"


def test_search_path_uses_channels_v5_and_t_encoding():
    assert _search_path("chat/~zod/general", "hello", None, 500) == (
        "/v5/chat/~zod/general/search/bounded/text//500/~.hello"
    )


def test_format_post_id_dots_bare_ud():
    assert _format_post_id("170141184507800833818237178278053937152") == (
        "170.141.184.507.800.833.818.237.178.278.053.937.152"
    )


def test_expose_cite_expansion_and_url_path():
    full = _expand_cite_path("diary/~zod/blog/170.141")
    assert full == "/1/chan/diary/~zod/blog/note/170.141"
    assert _cite_to_url_path(full) == "/chan/diary/~zod/blog/note/170.141"


@pytest.mark.asyncio
async def test_group_create_owned_creates_group_and_assigns_admin():
    client = FakeTlonClient()
    groups = TlonGroups(client)

    result = await groups.handle(
        "group_create_owned",
        {
            "title": "Hermes Group",
            "description": "Test group",
            "ship": "~malmur-halmex",
        },
    )

    assert result["success"] is True
    assert result["owner_ship"] == "~malmur-halmex"
    assert client.threads[0]["desk"] == "groups"
    assert client.threads[0]["input_mark"] == "group-create-thread"
    assert client.threads[0]["body"]["guestList"] == ["~malmur-halmex"]
    assert any(
        poke["json"] == {
            "group": {
                "flag": result["group_id"],
                "a-group": {
                    "role": {
                        "roles": ["admin"],
                        "a-role": {
                            "add": {
                                "title": "Admin",
                                "description": "Group administrator",
                                "image": "",
                                "cover": "",
                            }
                        },
                    }
                },
            }
        }
        for poke in client.pokes
    )
    assert any(
        poke["json"] == {
            "group": {
                "flag": result["group_id"],
                "a-group": {
                    "role": {
                        "roles": ["admin"],
                        "a-role": {"set-admin": None},
                    }
                },
            }
        }
        for poke in client.pokes
    )
    assert any(
        poke["json"] == {
            "group": {
                "flag": result["group_id"],
                "a-group": {
                    "seat": {
                        "ships": ["~malmur-halmex"],
                        "a-seat": {"add-roles": ["admin"]},
                    }
                },
            }
        }
        for poke in client.pokes
    )


@pytest.mark.asyncio
async def test_group_info_resolves_tlon_url_channel_to_parent_group():
    client = FakeTlonClient()
    groups = TlonGroups(client)

    result = await groups.handle(
        "group_info",
        {
            "group_id": (
                "https://host.tlon.network/apps/groups/Messages/Channel/ChannelRoot"
                "?channelId=chat%2F~host%2Ftest&groupId=~host%2Fwrong"
            ),
        },
    )

    assert result["success"] is True
    assert result["group_id"] == "~host/group"
    assert result["channel_id"] == "chat/~host/test"
    assert result["resolved_from"] == "channel_id"


@pytest.mark.asyncio
async def test_group_info_returns_candidates_instead_of_raising_404():
    class NotFoundClient(FakeTlonClient):
        async def scry(self, app, path, **kwargs):
            if app == "groups" and path.startswith("/v2/ui/groups/"):
                raise TlonHttpError(
                    "not found",
                    status=404,
                    app=app,
                    path=path,
                )
            return await super().scry(app, path, **kwargs)

    client = NotFoundClient()
    groups = TlonGroups(client)

    result = await groups.handle("group_info", {"group_id": "~host/wrong"})

    assert result["success"] is True
    assert result["found"] is False
    assert result["requested_group_id"] == "~host/wrong"
    assert result["candidates"][0]["group_id"] == "~host/group"


@pytest.mark.asyncio
async def test_notebook_post_uses_diary_metadata():
    client = FakeTlonClient()
    messages = TlonMessages(client)

    result = await messages.handle(
        "notebook_post",
        {
            "channel_id": "diary/~bot-palnet/notes",
            "title": "Notebook Title",
            "message": "Body text",
            "image": "https://example.com/cover.png",
        },
    )

    assert result["success"] is True
    poke = client.pokes[0]
    assert poke["app"] == "channels"
    post = poke["json"]["channel"]["action"]["post"]["add"]
    assert post["kind"] == "/diary"
    assert post["meta"]["title"] == "Notebook Title"
    assert post["meta"]["image"] == "https://example.com/cover.png"


@pytest.mark.asyncio
async def test_hook_add_uses_channels_server_hook_action():
    client = FakeTlonClient()
    hooks = TlonHooks(client)

    result = await hooks.handle(
        "hook_add",
        {"title": "Auto React", "source": ":: hook source"},
    )

    assert result["success"] is True
    assert client.pokes[0] == {
        "app": "channels-server",
        "mark": "hook-action-0",
        "json": {"add": {"name": "Auto React", "src": ":: hook source"}},
    }
