"""Focused seam contract for the extracted webhook dashboard router."""


def test_webhook_router_preserves_routes_and_web_server_compatibility(monkeypatch):
    import asyncio

    import hermes_cli.web_server as web_server
    from hermes_cli.web_routers import webhooks

    route_methods = {(route.path, tuple(sorted(route.methods))) for route in webhooks.router.routes}
    assert route_methods == {
        ("/api/webhooks", ("GET",)),
        ("/api/webhooks", ("POST",)),
        ("/api/webhooks/enable", ("POST",)),
        ("/api/webhooks/{name}", ("DELETE",)),
        ("/api/webhooks/{name}/enabled", ("PUT",)),
    }
    app_route_methods = {
        (route.path, tuple(sorted(route.methods)))
        for route in web_server.app.routes
        if hasattr(route, "methods")
    }
    assert route_methods <= app_route_methods

    for name in (
        "list_webhooks",
        "enable_webhooks",
        "create_webhook",
        "delete_webhook",
        "set_webhook_enabled",
        "_webhook_route_summary",
    ):
        assert getattr(web_server, name) is getattr(webhooks, name)

    calls = []
    monkeypatch.setattr(web_server, "_write_platform_enabled", lambda *args: calls.append(("write", args)))
    monkeypatch.setattr(
        web_server,
        "_restart_gateway_after_webhook_enable",
        lambda: {"restart_started": True, "restart_action": "gateway-restart", "restart_pid": 7},
    )

    result = asyncio.run(webhooks.enable_webhooks())

    assert calls == [("write", ("webhook", True))]
    assert result["restart_started"] is True
    assert result["restart_pid"] == 7


def test_webhook_list_route_uses_web_server_summary_seam(monkeypatch):
    import asyncio

    import hermes_cli.web_server as web_server
    import hermes_cli.webhook as webhook

    monkeypatch.setattr(webhook, "_get_webhook_base_url", lambda: "https://hooks.test")
    monkeypatch.setattr(
        webhook,
        "_load_subscriptions",
        lambda: {"build": {"description": "Build events"}},
    )
    monkeypatch.setattr(webhook, "_is_webhook_enabled", lambda: True)

    patched_summary = {"name": "patched-by-web-server"}
    monkeypatch.setattr(
        web_server,
        "_webhook_route_summary",
        lambda *args: patched_summary,
    )

    result = asyncio.run(web_server.list_webhooks())

    assert result["subscriptions"] == [patched_summary]
