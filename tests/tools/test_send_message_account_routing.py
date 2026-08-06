"""Per-account `send_message` routing — #8287.

The multi-account gateway lets one process host several bots. `send_message`
targets could not address them: `telegram@support:123` was rejected, so the
`send` consumer was the one reviewed path that stayed account-blind. These
tests cover the target parsing, the account-config derivation shared with the
gateway's adapter startup, and the fail-closed behavior for an unknown or
token-less account (never silently fall back to the default bot, which would
deliver to the wrong audience).
"""

import pytest

from gateway.config import (
    HomeChannel,
    Platform,
    PlatformConfig,
    derive_account_platform_config,
    resolve_platform_account,
)


# ---------------------------------------------------------------------------
# target parsing
# ---------------------------------------------------------------------------


def test_plain_platform_has_no_account():
    assert resolve_platform_account("telegram") == ("telegram", None)


def test_named_account_is_split_and_lowercased():
    assert resolve_platform_account("telegram@Support") == ("telegram", "support")


def test_default_account_spelling_resolves_to_none():
    """`@default` means the platform's default bot, spelled the same as omitting it."""
    assert resolve_platform_account("telegram@default") == ("telegram", None)


def test_empty_account_suffix_is_ignored():
    assert resolve_platform_account("telegram@") == ("telegram", None)


def test_empty_input_is_safe():
    assert resolve_platform_account("") == ("", None)


# ---------------------------------------------------------------------------
# derived per-account config (shared with the gateway's adapter startup)
# ---------------------------------------------------------------------------


def test_account_token_overrides_and_accounts_map_is_stripped():
    base = PlatformConfig(
        enabled=True,
        token="123:default",
        extra={
            "accounts": {"support": {"token": "456:support"}},
            "fallback_ips": ["1.2.3.4"],
        },
    )
    derived = derive_account_platform_config(
        Platform.TELEGRAM, base, {"token": "456:support"}
    )

    assert derived.token == "456:support"
    assert derived.extra["fallback_ips"] == ["1.2.3.4"]  # platform extra inherited
    # A derived config can never recurse into another account.
    assert "accounts" not in derived.extra
    # The base config is untouched (dataclasses.replace, not mutation).
    assert base.token == "123:default"
    assert "accounts" in base.extra


def test_account_home_channel_platform_is_implicit():
    base = PlatformConfig(enabled=True, token="123:default")
    derived = derive_account_platform_config(
        Platform.TELEGRAM, base, {"home_channel": {"chat_id": "-100999"}}
    )
    assert isinstance(derived.home_channel, HomeChannel)
    assert derived.home_channel.chat_id == "-100999"
    assert derived.home_channel.platform == Platform.TELEGRAM


def test_account_block_overrides_platform_extra():
    base = PlatformConfig(
        enabled=True, token="t", extra={"allowed_users": [1], "keep": "yes"}
    )
    derived = derive_account_platform_config(
        Platform.TELEGRAM, base, {"allowed_users": [2, 3]}
    )
    assert derived.extra["allowed_users"] == [2, 3]
    assert derived.extra["keep"] == "yes"


def test_empty_account_block_inherits_everything():
    base = PlatformConfig(enabled=True, token="123:default", extra={"a": 1})
    derived = derive_account_platform_config(Platform.TELEGRAM, base, {})
    assert derived.token == "123:default"
    assert derived.extra["a"] == 1


def test_runner_helper_delegates_to_the_shared_function():
    """The gateway's account startup and this send path must resolve an
    account identically — one implementation, two callers."""
    from gateway.run import GatewayRunner

    base = PlatformConfig(
        enabled=True, token="123:default", extra={"accounts": {"s": {}}}
    )
    block = {"token": "456:support", "home_channel": {"chat_id": "-100777"}}

    via_runner = GatewayRunner._account_platform_config(
        Platform.TELEGRAM, base, "support", block
    )
    via_shared = derive_account_platform_config(Platform.TELEGRAM, base, block)

    assert via_runner.token == via_shared.token == "456:support"
    assert via_runner.home_channel.chat_id == via_shared.home_channel.chat_id
    assert via_runner.extra == via_shared.extra


# ---------------------------------------------------------------------------
# fail-closed on a bad account (never fall back to the default bot)
# ---------------------------------------------------------------------------


def _send(target, monkeypatch, accounts=None, default_token="123:default"):
    """Drive send_message_tool far enough to hit account resolution, with the
    gateway config stubbed so no network or live adapter is involved."""
    import tools.send_message_tool as smt

    extra = {"accounts": accounts} if accounts is not None else {}
    pconfig = PlatformConfig(enabled=True, token=default_token, extra=extra)

    class _Cfg:
        platforms = {Platform.TELEGRAM: pconfig}

        def get_home_channel(self, platform):
            return None

    monkeypatch.setattr(smt, "load_gateway_config", lambda: _Cfg(), raising=False)
    import gateway.config as gwc

    monkeypatch.setattr(gwc, "load_gateway_config", lambda: _Cfg(), raising=False)
    return smt.send_message_tool({"target": target, "message": "hi"})


def test_unknown_account_is_rejected_with_the_configured_list(monkeypatch):
    out = _send(
        "telegram@nope:123", monkeypatch, accounts={"support": {"token": "t"}}
    )
    assert "nope" in out
    assert "support" in out  # tells the user what IS configured
    assert "TELEGRAM_BOT_TOKEN_NOPE" in out  # and how to add it


def test_account_with_no_token_is_rejected(monkeypatch):
    out = _send(
        "telegram@support:123", monkeypatch, accounts={"support": {"display_name": "S"}}
    )
    assert "support" in out
    assert "no token" in out.lower()
    assert "TELEGRAM_BOT_TOKEN_SUPPORT" in out


def test_account_target_on_platform_without_accounts_is_rejected(monkeypatch):
    """Fail closed rather than silently using the default bot's credential."""
    out = _send("telegram@support:123", monkeypatch, accounts=None)
    assert "support" in out
    assert "none" in out.lower()  # no accounts configured
