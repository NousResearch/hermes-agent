import types


def test_check_yuanbao_does_not_import_gateway_platform(monkeypatch):
    from tools import yuanbao_tools

    monkeypatch.delitem(yuanbao_tools.sys.modules, "gateway.platforms.yuanbao", raising=False)
    monkeypatch.setattr(
        "gateway.session_context.get_session_env",
        lambda key, default="": default,
    )
    monkeypatch.setattr(
        yuanbao_tools,
        "_get_active_adapter",
        lambda *, import_if_needed=True: (_ for _ in ()).throw(
            AssertionError("check should not import the Yuanbao adapter")
        )
        if import_if_needed
        else None,
    )

    assert yuanbao_tools._check_yuanbao() is False


def test_check_yuanbao_uses_loaded_adapter_without_import(monkeypatch):
    from tools import yuanbao_tools

    adapter = object()
    module = types.SimpleNamespace(get_active_adapter=lambda: adapter)
    monkeypatch.setitem(yuanbao_tools.sys.modules, "gateway.platforms.yuanbao", module)
    monkeypatch.setattr(
        "gateway.session_context.get_session_env",
        lambda key, default="": default,
    )

    assert yuanbao_tools._check_yuanbao() is True
