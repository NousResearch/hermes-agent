"""
test_yuanbao_pipeline.py - Unit tests for the inbound middleware pipeline.

Tests cover:
  1. InboundPipeline engine (use, use_before, use_after, remove, execute)
  2. InboundContext dataclass
  3. Individual middlewares (DecodeMiddleware, DedupMiddleware, SkipSelfMiddleware, etc.)
  4. InboundPipelineBuilder
  5. End-to-end pipeline integration
  6. OOP middleware ABC and class tests
"""

import asyncio
import sys
import os
import json

# Ensure project root is on the path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.platforms.yuanbao import (
    InboundContext,
    InboundMiddleware,
    InboundPipeline,
    DecodeMiddleware,
    ExtractFieldsMiddleware,
    DedupMiddleware,
    SkipSelfMiddleware,
    ChatRoutingMiddleware,
    AccessPolicy,
    AccessGuardMiddleware,
    AutoSetHomeMiddleware,
    ExtractContentMiddleware,
    PlaceholderFilterMiddleware,
    OwnerCommandMiddleware,
    BuildSourceMiddleware,
    GroupAtGuardMiddleware,
    QuoteContextMiddleware,
    MediaResolveMiddleware,
    PatchAnchorsMiddleware,
    DispatchMiddleware,
    InboundPipelineBuilder,
    YuanbaoAdapter,
    _MIN_RESOLVE_CONCURRENCY,
    _MAX_RESOLVE_CONCURRENCY,
)
from gateway.config import PlatformConfig


# ============================================================
# Helpers
# ============================================================

def make_config(**kwargs):
    extra = kwargs.pop("extra", {})
    extra.setdefault("app_id", "test_key")
    extra.setdefault("app_secret", "test_secret")
    extra.setdefault("ws_url", "wss://test.example.com/ws")
    extra.setdefault("api_domain", "https://test.example.com")
    return PlatformConfig(
        extra=extra,
        **kwargs,
    )


def make_adapter(**kwargs) -> YuanbaoAdapter:
    """Create a YuanbaoAdapter with test config."""
    config = make_config(**kwargs)
    adapter = YuanbaoAdapter(config)
    adapter._bot_id = "bot_123"
    return adapter


def make_ctx(adapter=None, conn_data=b"", **overrides) -> InboundContext:
    """Create an InboundContext with sensible defaults for testing."""
    if adapter is None:
        adapter = make_adapter()
    raw_frames = [conn_data] if conn_data else []
    ctx = InboundContext(adapter=adapter, raw_frames=raw_frames)
    for k, v in overrides.items():
        setattr(ctx, k, v)
    return ctx


def make_json_push(
    from_account="alice",
    to_account="bot_123",
    group_code="",
    text="Hello!",
    msg_id="msg-001",
) -> bytes:
    """Build a JSON callback_command push payload.

    Note: MsgContent inner fields use lowercase ("text" not "Text")
    because _extract_text() looks for lowercase keys.
    """
    msg_body = [{"MsgType": "TIMTextElem", "MsgContent": {"text": text}}]
    push = {
        "CallbackCommand": "C2C.CallbackAfterSendMsg",
        "From_Account": from_account,
        "To_Account": to_account,
        "MsgBody": msg_body,
        "MsgKey": msg_id,
    }
    if group_code:
        push["CallbackCommand"] = "Group.CallbackAfterSendMsg"
        push["GroupId"] = group_code
    return json.dumps(push).encode("utf-8")


# ============================================================
# 1. InboundPipeline Engine Tests
# ============================================================

class TestInboundPipeline:
    """Test the pipeline engine itself."""

    @pytest.mark.asyncio
    async def test_empty_pipeline(self):
        """Empty pipeline executes without error."""
        pipeline = InboundPipeline()
        ctx = make_ctx()
        await pipeline.execute(ctx)  # Should not raise




    @pytest.mark.asyncio
    async def test_conditional_guard_skip(self):
        """Middleware with when=False is skipped."""
        order = []

        async def mw_a(ctx, next_fn):
            order.append("a")
            await next_fn()

        async def mw_skipped(ctx, next_fn):
            order.append("skipped")
            await next_fn()

        async def mw_c(ctx, next_fn):
            order.append("c")
            await next_fn()

        pipeline = (
            InboundPipeline()
            .use("a", mw_a)
            .use("skipped", mw_skipped, when=lambda ctx: False)
            .use("c", mw_c)
        )
        await pipeline.execute(make_ctx())
        assert order == ["a", "c"]


    def test_use_before(self):
        """use_before inserts middleware before the target."""
        async def noop(ctx, next_fn):
            await next_fn()

        pipeline = InboundPipeline().use("a", noop).use("c", noop)
        pipeline.use_before("c", "b", noop)
        assert pipeline.middleware_names == ["a", "b", "c"]








    @pytest.mark.asyncio
    async def test_onion_model(self):
        """Middlewares support before/after processing (onion model)."""
        order = []

        async def mw_outer(ctx, next_fn):
            order.append("outer-before")
            await next_fn()
            order.append("outer-after")

        async def mw_inner(ctx, next_fn):
            order.append("inner")
            await next_fn()

        pipeline = InboundPipeline().use("outer", mw_outer).use("inner", mw_inner)
        await pipeline.execute(make_ctx())
        assert order == ["outer-before", "inner", "outer-after"]


# ============================================================
# 2. Individual Middleware Tests
# ============================================================

class TestDecodeMiddleware:
    @pytest.mark.asyncio
    async def test_json_decode(self):
        """DecodeMiddleware parses JSON push correctly."""
        push_data = make_json_push(from_account="alice", text="hi")
        ctx = make_ctx(conn_data=push_data)
        next_fn = AsyncMock()

        await DecodeMiddleware()(ctx, next_fn)

        assert ctx.push is not None
        assert ctx.decoded_via == "json"
        assert ctx.push.get("from_account") == "alice"
        next_fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_data_stops_pipeline(self):
        """DecodeMiddleware stops pipeline on empty conn_data."""
        ctx = make_ctx(conn_data=b"")
        next_fn = AsyncMock()

        await DecodeMiddleware()(ctx, next_fn)

        assert ctx.push is None
        next_fn.assert_not_awaited()



class TestExtractFieldsMiddleware:
    @pytest.mark.asyncio
    async def test_extracts_fields(self):
        """ExtractFieldsMiddleware populates ctx from push dict."""
        ctx = make_ctx(push={
            "from_account": "alice",
            "group_code": "grp-1",
            "group_name": "Test Group",
            "sender_nickname": "Alice",
            "msg_body": [{"msg_type": "TIMTextElem", "msg_content": {"text": "hi"}}],
            "msg_id": "msg-001",
            "cloud_custom_data": '{"key": "val"}',
        })
        next_fn = AsyncMock()

        await ExtractFieldsMiddleware()(ctx, next_fn)

        assert ctx.from_account == "alice"
        assert ctx.group_code == "grp-1"
        assert ctx.group_name == "Test Group"
        assert ctx.sender_nickname == "Alice"
        assert len(ctx.msg_body) == 1
        assert ctx.msg_id == "msg-001"
        assert ctx.cloud_custom_data == '{"key": "val"}'
        next_fn.assert_awaited_once()


class TestDedupMiddleware:
    @pytest.mark.asyncio
    async def test_new_message_passes(self):
        """DedupMiddleware passes new messages through."""
        adapter = make_adapter()
        ctx = make_ctx(adapter=adapter, msg_id="unique-msg-001")
        next_fn = AsyncMock()

        await DedupMiddleware()(ctx, next_fn)
        next_fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_duplicate_stops_pipeline(self):
        """DedupMiddleware stops pipeline for duplicate messages."""
        adapter = make_adapter()
        # Mark message as seen
        adapter._dedup.is_duplicate("dup-msg-001")

        ctx = make_ctx(adapter=adapter, msg_id="dup-msg-001")
        next_fn = AsyncMock()

        await DedupMiddleware()(ctx, next_fn)
        next_fn.assert_not_awaited()



class TestSkipSelfMiddleware:
    @pytest.mark.asyncio
    async def test_self_message_stops(self):
        """SkipSelfMiddleware stops pipeline for bot's own messages."""
        adapter = make_adapter()
        adapter._bot_id = "bot_123"
        ctx = make_ctx(adapter=adapter, from_account="bot_123")
        next_fn = AsyncMock()

        await SkipSelfMiddleware()(ctx, next_fn)
        next_fn.assert_not_awaited()



class TestChatRoutingMiddleware:
    @pytest.mark.asyncio
    async def test_group_routing(self):
        """ChatRoutingMiddleware sets group chat fields."""
        ctx = make_ctx(group_code="grp-1", group_name="Test Group")
        next_fn = AsyncMock()

        await ChatRoutingMiddleware()(ctx, next_fn)

        assert ctx.chat_id == "group:grp-1"
        assert ctx.chat_type == "group"
        assert ctx.chat_name == "Test Group"
        next_fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_dm_routing(self):
        """ChatRoutingMiddleware sets DM chat fields."""
        ctx = make_ctx(from_account="alice", sender_nickname="Alice")
        next_fn = AsyncMock()

        await ChatRoutingMiddleware()(ctx, next_fn)

        assert ctx.chat_id == "direct:alice"
        assert ctx.chat_type == "dm"
        assert ctx.chat_name == "Alice"
        next_fn.assert_awaited_once()



class TestAccessGuardMiddleware:
    @pytest.mark.asyncio
    async def test_open_policy_passes_with_opt_in(self, monkeypatch):
        """AccessGuardMiddleware passes open policy only with explicit opt-in."""
        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        adapter = make_adapter()
        adapter._access_policy = AccessPolicy(dm_policy="open", dm_allow_from=[], group_policy="open", group_allow_from=[])
        ctx = make_ctx(adapter=adapter, chat_type="dm", from_account="alice")
        next_fn = AsyncMock()

        await AccessGuardMiddleware()(ctx, next_fn)
        next_fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_open_policy_blocked_without_opt_in(self, monkeypatch):
        """AccessGuardMiddleware blocks open policy without explicit opt-in."""
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
        monkeypatch.delenv("YUANBAO_ALLOW_ALL_USERS", raising=False)
        adapter = make_adapter()
        adapter._access_policy = AccessPolicy(dm_policy="open", dm_allow_from=[], group_policy="open", group_allow_from=[])
        ctx = make_ctx(adapter=adapter, chat_type="dm", from_account="alice")
        next_fn = AsyncMock()

        await AccessGuardMiddleware()(ctx, next_fn)
        next_fn.assert_not_awaited()



    @pytest.mark.asyncio
    async def test_allowlist_dm_blocked(self):
        """AccessGuardMiddleware blocks DM when sender is not in allowlist."""
        adapter = make_adapter()
        adapter._access_policy = AccessPolicy(dm_policy="allowlist", dm_allow_from=["bob"], group_policy="open", group_allow_from=[])
        ctx = make_ctx(adapter=adapter, chat_type="dm", from_account="alice")
        next_fn = AsyncMock()

        await AccessGuardMiddleware()(ctx, next_fn)
        next_fn.assert_not_awaited()



    @pytest.mark.asyncio
    async def test_open_group_blocked_without_opt_in(self, monkeypatch):
        """AccessGuardMiddleware blocks open group policy without explicit opt-in."""
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
        monkeypatch.delenv("YUANBAO_ALLOW_ALL_USERS", raising=False)
        adapter = make_adapter()
        adapter._access_policy = AccessPolicy(
            dm_policy="pairing", dm_allow_from=[],
            group_policy="open", group_allow_from=[],
        )
        ctx = make_ctx(adapter=adapter, chat_type="group", group_code="grp-1")
        next_fn = AsyncMock()

        await AccessGuardMiddleware()(ctx, next_fn)
        next_fn.assert_not_awaited()


    @pytest.mark.asyncio
    async def test_unknown_group_policy_blocked(self, monkeypatch):
        """AccessGuardMiddleware blocks unrecognized group_policy values."""
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
        monkeypatch.delenv("YUANBAO_ALLOW_ALL_USERS", raising=False)
        adapter = make_adapter()
        adapter._access_policy = AccessPolicy(
            dm_policy="pairing", dm_allow_from=[],
            group_policy="typo", group_allow_from=[],
        )
        ctx = make_ctx(adapter=adapter, chat_type="group", group_code="grp-1")
        next_fn = AsyncMock()

        await AccessGuardMiddleware()(ctx, next_fn)
        next_fn.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("blank_sender", ["", "   ", None])
    async def test_pairing_blank_dm_blocked(self, monkeypatch, blank_sender):
        """AccessGuardMiddleware blocks pairing DMs with blank sender principals."""
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
        monkeypatch.delenv("YUANBAO_ALLOW_ALL_USERS", raising=False)
        adapter = make_adapter()
        adapter._access_policy = AccessPolicy(
            dm_policy="pairing", dm_allow_from=[],
            group_policy="pairing", group_allow_from=[],
        )
        ctx = make_ctx(adapter=adapter, chat_type="dm", from_account=blank_sender)
        next_fn = AsyncMock()

        await AccessGuardMiddleware()(ctx, next_fn)
        next_fn.assert_not_awaited()


class TestAccessPolicy:
    def test_open_group_requires_opt_in(self, monkeypatch):
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
        monkeypatch.delenv("YUANBAO_ALLOW_ALL_USERS", raising=False)
        policy = AccessPolicy(
            dm_policy="pairing", dm_allow_from=[],
            group_policy="open", group_allow_from=[],
        )
        assert policy.is_group_allowed("unknown-group") is False



    def test_unknown_group_policy_denies(self, monkeypatch):
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
        monkeypatch.delenv("YUANBAO_ALLOW_ALL_USERS", raising=False)
        policy = AccessPolicy(
            dm_policy="pairing", dm_allow_from=[],
            group_policy="typo", group_allow_from=[],
        )
        assert policy.is_group_allowed("unknown-group") is False

    @pytest.mark.parametrize("blank_sender", ["", "   ", None])
    def test_pairing_dm_intake_denies_blank_principal(self, monkeypatch, blank_sender):
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
        monkeypatch.delenv("YUANBAO_ALLOW_ALL_USERS", raising=False)
        policy = AccessPolicy(
            dm_policy="pairing", dm_allow_from=[],
            group_policy="pairing", group_allow_from=[],
        )
        assert policy.is_dm_intake_allowed(blank_sender) is False



class TestAutoSetHomeMiddleware:
    @pytest.mark.asyncio
    async def test_pairing_unapproved_dm_does_not_set_home(self, monkeypatch, tmp_path):
        """Intake-only pairing DMs must not claim YUANBAO_HOME_CHANNEL."""
        monkeypatch.delenv("YUANBAO_HOME_CHANNEL", raising=False)
        monkeypatch.delenv("YUANBAO_ALLOW_ALL_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)

        adapter = make_adapter()
        adapter._auto_sethome_done = False
        adapter._access_policy = AccessPolicy(
            dm_policy="pairing",
            dm_allow_from=[],
            group_policy="pairing",
            group_allow_from=[],
        )
        ctx = make_ctx(
            adapter=adapter,
            chat_type="dm",
            chat_id="direct:unapproved-sender",
            from_account="unapproved-sender",
        )
        next_fn = AsyncMock()

        with patch("gateway.pairing.PairingStore") as mock_store_cls:
            mock_store_cls.return_value.is_approved.return_value = False
            await AutoSetHomeMiddleware()(ctx, next_fn)

        assert "YUANBAO_HOME_CHANNEL" not in os.environ
        assert not (tmp_path / "config.yaml").exists()
        next_fn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_pairing_approved_dm_sets_home(self, monkeypatch, tmp_path):
        """Pairing-approved senders may auto-designate the home channel."""
        monkeypatch.delenv("YUANBAO_HOME_CHANNEL", raising=False)
        monkeypatch.setattr(
            "hermes_constants.get_hermes_home",
            lambda: tmp_path,
        )

        adapter = make_adapter()
        adapter._auto_sethome_done = False
        adapter._access_policy = AccessPolicy(
            dm_policy="pairing",
            dm_allow_from=[],
            group_policy="pairing",
            group_allow_from=[],
        )
        ctx = make_ctx(
            adapter=adapter,
            chat_type="dm",
            chat_id="direct:approved-sender",
            from_account="approved-sender",
            chat_name="Approved",
        )
        next_fn = AsyncMock()

        with patch("gateway.pairing.PairingStore") as mock_store_cls:
            mock_store_cls.return_value.is_approved.return_value = True
            await AutoSetHomeMiddleware()(ctx, next_fn)

        assert os.environ.get("YUANBAO_HOME_CHANNEL") == "direct:approved-sender"
        next_fn.assert_awaited_once()



class TestSenderMayDesignateHome:
    def test_pairing_unapproved_sender_denied(self, monkeypatch):
        monkeypatch.delenv("YUANBAO_ALLOW_ALL_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)

        adapter = make_adapter()
        adapter._access_policy = AccessPolicy(
            dm_policy="pairing",
            dm_allow_from=[],
            group_policy="pairing",
            group_allow_from=[],
        )
        ctx = make_ctx(
            adapter=adapter,
            chat_type="dm",
            from_account="unapproved-sender",
        )

        with patch("gateway.pairing.PairingStore") as mock_store_cls:
            mock_store_cls.return_value.is_approved.return_value = False
            assert adapter._sender_may_designate_home(ctx) is False

    def test_pairing_approved_sender_allowed(self):
        adapter = make_adapter()
        adapter._access_policy = AccessPolicy(
            dm_policy="pairing",
            dm_allow_from=[],
            group_policy="pairing",
            group_allow_from=[],
        )
        ctx = make_ctx(
            adapter=adapter,
            chat_type="dm",
            from_account="approved-sender",
        )

        with patch("gateway.pairing.PairingStore") as mock_store_cls:
            mock_store_cls.return_value.is_approved.return_value = True
            assert adapter._sender_may_designate_home(ctx) is True



class TestExtractContentMiddleware:
    @pytest.mark.asyncio
    async def test_extracts_text_and_media(self):
        """ExtractContentMiddleware extracts text and media refs."""
        adapter = make_adapter()
        msg_body = [
            {"msg_type": "TIMTextElem", "msg_content": {"text": "Hello!"}},
            {"msg_type": "TIMImageElem", "msg_content": {
                "image_info_array": [{"url": "https://img.example.com/1.jpg"}]
            }},
        ]
        ctx = make_ctx(adapter=adapter, msg_body=msg_body)
        next_fn = AsyncMock()

        await ExtractContentMiddleware()(ctx, next_fn)

        assert "Hello!" in ctx.raw_text
        assert len(ctx.media_refs) == 1
        assert ctx.media_refs[0]["kind"] == "image"
        next_fn.assert_awaited_once()


class TestPlaceholderFilterMiddleware:
    @pytest.mark.asyncio
    async def test_placeholder_stops(self):
        """PlaceholderFilterMiddleware stops on pure placeholder."""
        ctx = make_ctx(raw_text="[image]", media_refs=[])
        next_fn = AsyncMock()

        await PlaceholderFilterMiddleware()(ctx, next_fn)
        next_fn.assert_not_awaited()




class TestGroupAtGuardMiddleware:
    @pytest.mark.asyncio
    async def test_dm_passes(self):
        """GroupAtGuardMiddleware passes DM messages."""
        adapter = make_adapter()
        ctx = make_ctx(adapter=adapter, chat_type="dm")
        next_fn = AsyncMock()

        await GroupAtGuardMiddleware()(ctx, next_fn)
        next_fn.assert_awaited_once()


    @pytest.mark.asyncio
    async def test_group_without_at_bot_observes(self):
        """GroupAtGuardMiddleware observes group messages without @bot."""
        adapter = make_adapter()
        adapter._bot_id = "bot_123"
        adapter._session_store = None  # No session store -> observe is a no-op
        ctx = make_ctx(
            adapter=adapter,
            chat_type="group",
            chat_id="group:grp-1",
            msg_body=[{"msg_type": "TIMTextElem", "msg_content": {"text": "hi"}}],
            from_account="alice",
            sender_nickname="Alice",
            raw_text="hi",
            source=MagicMock(),
        )
        next_fn = AsyncMock()

        await GroupAtGuardMiddleware()(ctx, next_fn)

        next_fn.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_owner_command_skips_at_check(self):
        """GroupAtGuardMiddleware passes when owner_command is set."""
        adapter = make_adapter()
        adapter._bot_id = "bot_123"
        ctx = make_ctx(
            adapter=adapter,
            chat_type="group",
            msg_body=[],
            owner_command="/new",
            source=MagicMock(),
        )
        next_fn = AsyncMock()

        await GroupAtGuardMiddleware()(ctx, next_fn)
        next_fn.assert_awaited_once()


class TestDispatchMiddleware:
    @pytest.mark.asyncio
    async def test_addressed_group_message_bypasses_serial_queue(self):
        """A second @bot turn must reach busy-input handling immediately."""
        adapter = make_adapter()
        adapter._bot_id = "bot_123"
        first_started = asyncio.Event()
        second_started = asyncio.Event()
        release_first = asyncio.Event()

        async def handle_message(event):
            if event.message_id == "first":
                first_started.set()
                await release_first.wait()
            elif event.message_id == "second":
                second_started.set()

        adapter.handle_message = handle_message
        source = adapter.build_source(
            chat_id="group:grp-1",
            chat_type="group",
            user_id="alice",
            user_name="Alice",
            thread_id="main",
        )
        mention = {
            "msg_type": "TIMCustomElem",
            "msg_content": {
                "data": json.dumps({"elem_type": 1002, "text": "@bot", "user_id": "bot_123"})
            },
        }

        def addressed_context(message_id):
            return make_ctx(
                adapter=adapter,
                chat_type="group",
                chat_id="group:grp-1",
                msg_body=[mention],
                msg_id=message_id,
                raw_text=f"message {message_id}",
                source=source,
            )

        middleware = DispatchMiddleware()
        await middleware(addressed_context("first"), AsyncMock())
        await asyncio.wait_for(first_started.wait(), timeout=1)

        await middleware(addressed_context("second"), AsyncMock())
        await asyncio.wait_for(second_started.wait(), timeout=1)

        assert not adapter._group_queues
        release_first.set()
        await asyncio.gather(*list(adapter._inbound_tasks))


class TestAutoSetHomeAfterGroupAtGuard:
    @pytest.mark.asyncio
    async def test_unaddressed_group_does_not_set_home(self, monkeypatch, tmp_path):
        """Group traffic dropped by GroupAtGuard must not persist YUANBAO_HOME_CHANNEL."""
        monkeypatch.delenv("YUANBAO_HOME_CHANNEL", raising=False)
        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        monkeypatch.setattr(
            "hermes_constants.get_hermes_home",
            lambda: tmp_path,
        )

        adapter = make_adapter()
        adapter._auto_sethome_done = False
        adapter._access_policy = AccessPolicy(
            dm_policy="pairing",
            dm_allow_from=[],
            group_policy="open",
            group_allow_from=[],
        )
        adapter._session_store = None

        push_data = make_json_push(
            from_account="alice",
            group_code="grp-1",
            text="hello group",
            msg_id="msg-group-001",
        )
        ctx = InboundContext(adapter=adapter, raw_frames=[push_data])
        pipeline = InboundPipelineBuilder.build()
        await pipeline.execute(ctx)

        assert "YUANBAO_HOME_CHANNEL" not in os.environ
        assert not (tmp_path / "config.yaml").exists()


# ============================================================
# 4. Factory Tests
# ============================================================

class TestCreateInboundPipeline:
    def test_default_pipeline_has_all_middlewares(self):
        """InboundPipelineBuilder.build() creates pipeline with all expected middlewares."""
        pipeline = InboundPipelineBuilder.build()
        expected = [
            "decode",
            "extract-fields",
            "recall_guard",
            "dedup",
            "skip-self",
            "chat-routing",
            "access-guard",
            "extract-content",
            "placeholder-filter",
            "owner-command",
            "build-source",
            "group-at-guard",
            "auto-sethome",
            "group-attribution",
            "classify-msg-type",
            "quote-context",
            "forwarded-records-parse",
            "media-resolve",
            "patch-anchors",
            "dispatch",
        ]
        assert pipeline.middleware_names == expected


# ============================================================
# 5. End-to-End Pipeline Integration Tests
# ============================================================

class TestPipelineIntegration:
    @pytest.mark.asyncio
    async def test_full_dm_message_flow(self, monkeypatch):
        """Full pipeline processes a DM message end-to-end."""
        monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
        adapter = make_adapter()
        adapter._bot_id = "bot_123"
        adapter._access_policy = AccessPolicy(dm_policy="open", dm_allow_from=[], group_policy="open", group_allow_from=[])
        adapter.handle_message = AsyncMock()
        adapter._resolve_inbound_media_urls = AsyncMock(return_value=([], []))

        push_data = make_json_push(
            from_account="alice",
            to_account="bot_123",
            text="Hello bot!",
            msg_id="msg-e2e-001",
        )

        ctx = InboundContext(adapter=adapter, raw_frames=[push_data])
        pipeline = InboundPipelineBuilder.build()
        await pipeline.execute(ctx)

        # Verify context was populated correctly
        assert ctx.decoded_via == "json"
        assert ctx.from_account == "alice"
        assert ctx.chat_type == "dm"
        assert ctx.chat_id == "direct:alice"
        assert "Hello bot!" in ctx.raw_text
        assert ctx.source is not None

    @pytest.mark.asyncio
    async def test_pairing_blank_sender_stops_at_access_guard(self, monkeypatch):
        """Whitespace-only C2C senders must not pass pairing intake into dispatch."""
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
        monkeypatch.delenv("YUANBAO_ALLOW_ALL_USERS", raising=False)
        adapter = make_adapter()
        adapter._bot_id = "bot_123"
        adapter._access_policy = AccessPolicy(
            dm_policy="pairing", dm_allow_from=[],
            group_policy="pairing", group_allow_from=[],
        )
        adapter.handle_message = AsyncMock()

        push_data = make_json_push(
            from_account="   ",
            to_account="bot_123",
            text="Hello bot!",
            msg_id="msg-blank-001",
        )

        ctx = InboundContext(adapter=adapter, raw_frames=[push_data])
        pipeline = InboundPipelineBuilder.build()
        await pipeline.execute(ctx)

        assert ctx.from_account == "   "
        assert ctx.chat_type == "dm"
        assert ctx.chat_id == "direct:   "
        assert ctx.source is None
        adapter.handle_message.assert_not_awaited()




    @pytest.mark.asyncio
    async def test_adapter_has_pipeline(self):
        """YuanbaoAdapter.__init__ creates an inbound pipeline."""
        adapter = make_adapter()
        assert hasattr(adapter, "_inbound_pipeline")
        assert isinstance(adapter._inbound_pipeline, InboundPipeline)



if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ============================================================
# 6. OOP Middleware Tests
# ============================================================

class TestInboundMiddlewareABC:
    """Test the InboundMiddleware OOP protocol (callable + named)."""

    def test_subclass_with_handle_works(self):
        """Subclass with handle() can be instantiated."""
        class GoodMiddleware(InboundMiddleware):
            name = "good"
            async def handle(self, ctx, next_fn):
                await next_fn()
        mw = GoodMiddleware()
        assert mw.name == "good"

    @pytest.mark.asyncio
    async def test_callable_protocol(self):
        """Middleware instances are callable via __call__."""
        class TestMW(InboundMiddleware):
            name = "test"
            async def handle(self, ctx, next_fn):
                ctx.raw_text = "called"
                await next_fn()

        mw = TestMW()
        ctx = make_ctx()
        next_fn = AsyncMock()
        await mw(ctx, next_fn)  # Call via __call__
        assert ctx.raw_text == "called"
        next_fn.assert_awaited_once()


class TestMiddlewareClasses:
    """Pin the canonical ``name`` of each concrete middleware class.

    These names are referenced by ``InboundPipelineBuilder.build()`` ordering,
    by ``use_before`` / ``use_after`` insertion in extensions, and by log
    messages — so they're a real downstream contract worth pinning.
    """

    MIDDLEWARE_CLASSES = [
        (DecodeMiddleware, "decode"),
        (ExtractFieldsMiddleware, "extract-fields"),
        (DedupMiddleware, "dedup"),
        (SkipSelfMiddleware, "skip-self"),
        (ChatRoutingMiddleware, "chat-routing"),
        (AccessGuardMiddleware, "access-guard"),
        (ExtractContentMiddleware, "extract-content"),
        (PlaceholderFilterMiddleware, "placeholder-filter"),
        (OwnerCommandMiddleware, "owner-command"),
        (BuildSourceMiddleware, "build-source"),
        (GroupAtGuardMiddleware, "group-at-guard"),
        (DispatchMiddleware, "dispatch"),
    ]

    @pytest.mark.parametrize("cls,expected_name", MIDDLEWARE_CLASSES)
    def test_has_correct_name(self, cls, expected_name):
        """Each middleware class has the expected name."""
        mw = cls()
        assert mw.name == expected_name


class TestPipelineOOPRegistration:
    """Test that InboundPipeline works with OOP middleware instances."""

    @pytest.mark.asyncio
    async def test_use_with_middleware_instance(self):
        """pipeline.use(SomeMiddleware()) auto-extracts name."""
        class TestMW(InboundMiddleware):
            name = "test-mw"
            async def handle(self, ctx, next_fn):
                ctx.raw_text = "oop-works"
                await next_fn()

        pipeline = InboundPipeline().use(TestMW())
        assert pipeline.middleware_names == ["test-mw"]

        ctx = make_ctx()
        await pipeline.execute(ctx)
        assert ctx.raw_text == "oop-works"



# ============================================================
# QuoteContextMiddleware Tests
# ============================================================
#
# Quote-media resolution used to depend on a process-local
# msg_id→resids cache populated by ExtractContentMiddleware. After #27866
# made gateway/run.py write @bot user transcript entries with
# message_id (symmetric with the observed-group writer at yuanbao.py:2091),
# QuoteContextMiddleware's transcript-lookup path covers every quote case
# we used to rely on the cache for, so the cache (and those tests) were
# removed. ``_extract_quote_context()`` is now a pure (quote_id, quote_text)
# extractor; quote media references are populated separately by
# ``_extract_media_refs_from_transcript()`` against the transcript store.

class TestQuoteContextMiddleware:
    """Tests for QuoteContextMiddleware._extract_quote_context."""

    def test_extract_quote_context_no_cloud_data(self):
        """Returns (None, None) when cloud_custom_data is empty."""
        result = QuoteContextMiddleware()._extract_quote_context("")
        assert result == (None, None)





    @pytest.mark.asyncio
    async def test_handle_sets_ctx_fields(self):
        """QuoteContextMiddleware.handle() sets ctx.reply_to_message_id, reply_to_text, quote_media_refs.

        With no transcript store wired up, quote_media_refs falls back to []
        — media resolution from transcript is covered by separate tests.
        """
        cloud_data = json.dumps({
            "quote": {
                "id": "quoted-msg-004",
                "desc": "Check this image",
                "sender_nickname": "Dave",
            }
        })
        adapter = make_adapter()
        adapter._session_store = None  # no transcript lookup path
        ctx = make_ctx(adapter=adapter, cloud_custom_data=cloud_data)
        next_fn = AsyncMock()

        await QuoteContextMiddleware()(ctx, next_fn)

        assert ctx.reply_to_message_id == "quoted-msg-004"
        assert ctx.reply_to_text == "Dave: Check this image"
        assert ctx.quote_media_refs == []
        next_fn.assert_awaited_once()


# ============================================================
# MediaResolveMiddleware Tests
# ============================================================
#
# After the dispatch refactor, MediaResolveMiddleware is the single entry
# point for all inbound media downloads. It merges up to three sources
# into ``ctx.media_urls`` / ``ctx.media_types`` (deduped, in this order):
#
#   1) media carried by the current message itself (always),
#   2) quote_media_refs (when reply_to_message_id is set),
#   3) recent group-observed media (only when chat_type == "group" and
#      no quote is present).
#
# Direct messages skip the observed backfill entirely.

class TestResolveYbresRefs:
    """Direct tests for ``MediaResolveMiddleware._resolve_ybres_refs``.

    This classmethod is the shared engine for both ``_resolve_quote_media``
    and ``_collect_observed_media``. Patching the two upstream callers from
    routing tests doesn't exercise its filtering / error-swallowing
    behavior, so we pin those contracts directly here.
    """

    @pytest.mark.asyncio
    async def test_resolves_each_ref_in_order(self):
        """Successful resolution returns ``(paths, mimes)`` aligned with input order."""
        adapter = make_adapter()
        refs = [
            ("rid-1", "image", "a.jpg"),
            ("rid-2", "file", "doc.pdf"),
        ]

        with patch.object(
            MediaResolveMiddleware, "_fetch_resource_url",
            new=AsyncMock(side_effect=["https://fresh/1", "https://fresh/2"]),
        ) as p_fetch, patch.object(
            MediaResolveMiddleware, "_download_and_cache",
            new=AsyncMock(side_effect=[
                ("/cache/a.jpg", "image/jpeg"),
                ("/cache/doc.pdf", "application/pdf"),
            ]),
        ) as p_cache:
            paths, mimes = await MediaResolveMiddleware._resolve_ybres_refs(
                adapter, refs, log_prefix="test",
            )

        assert paths == ["/cache/a.jpg", "/cache/doc.pdf"]
        assert mimes == ["image/jpeg", "application/pdf"]
        assert p_fetch.await_count == 2
        # filename from the ref tuple is forwarded to download_and_cache
        cache_kwargs = [c.kwargs for c in p_cache.await_args_list]
        assert cache_kwargs[0]["file_name"] == "a.jpg"
        assert cache_kwargs[0]["kind"] == "image"
        assert cache_kwargs[0]["resource_id"] == "rid-1"
        assert cache_kwargs[1]["file_name"] == "doc.pdf"



    @pytest.mark.asyncio
    async def test_cache_miss_drops_ref(self):
        """If ``_download_and_cache`` returns None, the ref is dropped."""
        adapter = make_adapter()
        refs = [("rid-1", "image", "")]

        with patch.object(
            MediaResolveMiddleware, "_fetch_resource_url",
            new=AsyncMock(return_value="https://fresh/1"),
        ), patch.object(
            MediaResolveMiddleware, "_download_and_cache",
            new=AsyncMock(return_value=None),
        ):
            paths, mimes = await MediaResolveMiddleware._resolve_ybres_refs(
                adapter, refs, log_prefix="test",
            )

        assert paths == []
        assert mimes == []

    @pytest.mark.asyncio
    async def test_cache_hit_skips_resource_url_resolve(self, tmp_path):
        """A resourceId cache hit must not await ``_fetch_resource_url`` at all."""
        adapter = make_adapter()
        cached_file = tmp_path / "rid-cached.jpg"
        cached_file.write_bytes(b"cached-image")
        MediaResolveMiddleware._resource_cache.clear()
        try:
            MediaResolveMiddleware._put_cached_resource(
                "rid-cached", str(cached_file), "image/jpeg",
            )

            with patch.object(
                MediaResolveMiddleware, "_fetch_resource_url",
                new=AsyncMock(return_value="https://fresh/never"),
            ) as p_fetch:
                paths, mimes = await MediaResolveMiddleware._resolve_ybres_refs(
                    adapter, [("rid-cached", "image", "")], log_prefix="test",
                )

            assert paths == [str(cached_file)]
            assert mimes == ["image/jpeg"]
            p_fetch.assert_not_awaited()
        finally:
            MediaResolveMiddleware._resource_cache.clear()



class TestResolveMediaUrlsCacheHit:
    """Current-message media cache hits must skip the download-URL resolve."""

    @pytest.mark.asyncio
    async def test_cache_hit_skips_resolve_download_url(self, tmp_path):
        adapter = make_adapter()
        cached_file = tmp_path / "rid-cached.jpg"
        cached_file.write_bytes(b"cached-image")
        MediaResolveMiddleware._resource_cache.clear()
        try:
            MediaResolveMiddleware._put_cached_resource(
                "rid-cached", str(cached_file), "image/jpeg",
            )

            with patch.object(
                MediaResolveMiddleware, "_resolve_download_url",
                new=AsyncMock(return_value="https://fresh/never"),
            ) as p_resolve, patch.object(
                MediaResolveMiddleware, "_fetch_resource_url",
                new=AsyncMock(return_value="https://fresh/never"),
            ) as p_fetch:
                paths, mimes = await MediaResolveMiddleware._resolve_media_urls(
                    adapter,
                    [{
                        "kind": "image",
                        "url": "https://hunyuan.tencent.com/api/resource/download?resourceId=rid-cached",
                    }],
                )

            assert paths == [str(cached_file)]
            assert mimes == ["image/jpeg"]
            p_resolve.assert_not_awaited()
            p_fetch.assert_not_awaited()
        finally:
            MediaResolveMiddleware._resource_cache.clear()


class TestResolveYbresRefsConcurrency:
    """Bounded-concurrency contracts for ``_resolve_ybres_refs``."""

    # ------------------------------------------------------------------
    # Bounded-concurrency contracts (issue 3 in
    # yuanbao-media-pipeline-optimizations.md). These are behavior
    # contracts, not implementation snapshots — they assert the
    # invariants the new gather()-based path must hold, not how it's
    # wired internally.
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_concurrent_resolve_preserves_input_order(self):
        """Order of returned (paths, mimes) must match input ``refs`` order
        even when later refs finish downloading first.
        """
        adapter = make_adapter(extra={"media_resolve_concurrency": 4})
        refs = [
            ("rid-A", "image", ""),
            ("rid-B", "image", ""),
            ("rid-C", "image", ""),
        ]

        # _fetch is fast and uniform; the interesting variation is in
        # _download_and_cache, where rid-A is the slowest. If results
        # were assembled by completion order, rid-A would land last.
        async def slow_fetch(_adapter, rid):
            return f"https://fresh/{rid}"

        delays = {"rid-A": 0.06, "rid-B": 0.02, "rid-C": 0.0}
        results_by_rid = {
            "rid-A": ("/cache/A.jpg", "image/jpeg"),
            "rid-B": ("/cache/B.jpg", "image/jpeg"),
            "rid-C": ("/cache/C.jpg", "image/jpeg"),
        }

        async def slow_download(_adapter, *, fetch_url, kind, file_name, log_tag, resource_id):
            await asyncio.sleep(delays[resource_id])
            return results_by_rid[resource_id]

        with patch.object(
            MediaResolveMiddleware, "_fetch_resource_url",
            new=AsyncMock(side_effect=slow_fetch),
        ), patch.object(
            MediaResolveMiddleware, "_download_and_cache",
            new=AsyncMock(side_effect=slow_download),
        ):
            paths, mimes = await MediaResolveMiddleware._resolve_ybres_refs(
                adapter, refs, log_prefix="test",
            )

        assert paths == ["/cache/A.jpg", "/cache/B.jpg", "/cache/C.jpg"]
        assert mimes == ["image/jpeg", "image/jpeg", "image/jpeg"]

    @pytest.mark.asyncio
    async def test_concurrency_one_equivalent_to_sequential(self):
        """``media_resolve_concurrency = 1`` must behave like the legacy
        sequential path — at any moment at most one ``_download_and_cache``
        is in flight.
        """
        adapter = make_adapter(extra={"media_resolve_concurrency": 1})
        refs = [("rid-A", "image", ""), ("rid-B", "image", ""), ("rid-C", "image", "")]

        in_flight = 0
        max_in_flight = 0

        async def tracked_download(_adapter, *, fetch_url, kind, file_name, log_tag, resource_id):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            try:
                # Yield to the event loop so any concurrent coroutine
                # would have a chance to also enter the critical section.
                await asyncio.sleep(0.01)
                return (f"/cache/{resource_id}.jpg", "image/jpeg")
            finally:
                in_flight -= 1

        with patch.object(
            MediaResolveMiddleware, "_fetch_resource_url",
            new=AsyncMock(side_effect=lambda _a, rid: f"https://fresh/{rid}"),
        ), patch.object(
            MediaResolveMiddleware, "_download_and_cache",
            new=AsyncMock(side_effect=tracked_download),
        ):
            paths, _ = await MediaResolveMiddleware._resolve_ybres_refs(
                adapter, refs, log_prefix="test",
            )

        assert max_in_flight == 1
        assert paths == ["/cache/rid-A.jpg", "/cache/rid-B.jpg", "/cache/rid-C.jpg"]

    @pytest.mark.asyncio
    async def test_concurrency_caps_inflight_downloads(self):
        """Configured concurrency bounds the number of in-flight downloads."""
        adapter = make_adapter(extra={"media_resolve_concurrency": 2})
        refs = [(f"rid-{i}", "image", "") for i in range(6)]

        in_flight = 0
        max_in_flight = 0

        async def tracked_download(_adapter, *, fetch_url, kind, file_name, log_tag, resource_id):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            try:
                await asyncio.sleep(0.01)
                return (f"/cache/{resource_id}.jpg", "image/jpeg")
            finally:
                in_flight -= 1

        with patch.object(
            MediaResolveMiddleware, "_fetch_resource_url",
            new=AsyncMock(side_effect=lambda _a, rid: f"https://fresh/{rid}"),
        ), patch.object(
            MediaResolveMiddleware, "_download_and_cache",
            new=AsyncMock(side_effect=tracked_download),
        ):
            paths, _ = await MediaResolveMiddleware._resolve_ybres_refs(
                adapter, refs, log_prefix="test",
            )

        assert max_in_flight == 2
        assert paths == [f"/cache/rid-{i}.jpg" for i in range(6)]


    @pytest.mark.asyncio
    async def test_misconfigured_concurrency_clamped(self):
        """Out-of-range or non-int concurrency values are clamped, not crashed."""
        # Negative -> clamped up to MIN
        adapter_low = make_adapter(extra={"media_resolve_concurrency": -3})
        assert adapter_low.media_resolve_concurrency >= _MIN_RESOLVE_CONCURRENCY

        # Huge -> clamped down to MAX
        adapter_high = make_adapter(extra={"media_resolve_concurrency": 9999})
        assert adapter_high.media_resolve_concurrency <= _MAX_RESOLVE_CONCURRENCY

        # Non-int garbage -> falls back to default, doesn't raise
        adapter_garbage = make_adapter(extra={"media_resolve_concurrency": "fast"})
        assert (
            _MIN_RESOLVE_CONCURRENCY
            <= adapter_garbage.media_resolve_concurrency
            <= _MAX_RESOLVE_CONCURRENCY
        )




class TestMediaResolveMiddlewareRouting:
    """Branch-routing tests for MediaResolveMiddleware.handle()."""

    def _make_resolved_ctx(self, *, chat_type: str, reply_to: str = None,
                            quote_media_refs=None, raw_text: str = "hello"):
        adapter = make_adapter()
        ctx = make_ctx(
            adapter=adapter,
            chat_type=chat_type,
            reply_to_message_id=reply_to,
            quote_media_refs=list(quote_media_refs or []),
            raw_text=raw_text,
            media_refs=[],  # no own attachments by default
        )
        return adapter, ctx

    @pytest.mark.asyncio
    async def test_dm_no_quote_skips_observed_backfill(self):
        """In dm chats, observed-media backfill is never invoked."""
        _adapter, ctx = self._make_resolved_ctx(chat_type="dm")

        with patch.object(
            MediaResolveMiddleware, "_resolve_media_urls",
            new=AsyncMock(return_value=([], [])),
        ) as p_own, patch.object(
            MediaResolveMiddleware, "_resolve_quote_media",
            new=AsyncMock(return_value=([], [])),
        ) as p_quote, patch.object(
            MediaResolveMiddleware, "_collect_observed_media",
            new=AsyncMock(return_value=([], [])),
        ) as p_observed:
            next_fn = AsyncMock()
            await MediaResolveMiddleware()(ctx, next_fn)

        p_own.assert_awaited_once()
        p_quote.assert_not_awaited()
        p_observed.assert_not_awaited()
        next_fn.assert_awaited_once()


    @pytest.mark.asyncio
    async def test_group_no_quote_runs_observed_backfill(self):
        """In group chats without quote, observed-media backfill is invoked."""
        _adapter, ctx = self._make_resolved_ctx(chat_type="group")

        with patch.object(
            MediaResolveMiddleware, "_resolve_media_urls",
            new=AsyncMock(return_value=([], [])),
        ), patch.object(
            MediaResolveMiddleware, "_resolve_quote_media",
            new=AsyncMock(return_value=([], [])),
        ) as p_quote, patch.object(
            MediaResolveMiddleware, "_collect_observed_media",
            new=AsyncMock(return_value=(["/cache/o1.jpg"], ["image/jpeg"])),
        ) as p_observed:
            next_fn = AsyncMock()
            await MediaResolveMiddleware()(ctx, next_fn)

        p_quote.assert_not_awaited()
        p_observed.assert_awaited_once()
        assert ctx.media_urls == ["/cache/o1.jpg"]



    @pytest.mark.asyncio
    async def test_placeholder_recheck_uses_own_count_only(self):
        """Placeholder retry-skip uses ``own_count``, not the merged total.

        A bare placeholder text (e.g. ``[image]``) accompanied only by a
        quote-resolved image is still skippable — quote media must not
        flip a placeholder into a non-placeholder.
        """
        _adapter, ctx = self._make_resolved_ctx(
            chat_type="dm",
            reply_to="quoted-004",
            quote_media_refs=[("rid", "image", "")],
            raw_text="[image]",
        )

        with patch.object(
            MediaResolveMiddleware, "_resolve_media_urls",
            new=AsyncMock(return_value=([], [])),  # no own media
        ), patch.object(
            MediaResolveMiddleware, "_resolve_quote_media",
            new=AsyncMock(return_value=(["/cache/q.jpg"], ["image/jpeg"])),
        ), patch.object(
            PlaceholderFilterMiddleware, "is_skippable_placeholder",
            return_value=True,
        ) as p_check:
            next_fn = AsyncMock()
            await MediaResolveMiddleware()(ctx, next_fn)

        # Pipeline short-circuited despite quote media being present.
        next_fn.assert_not_awaited()
        # And the second-pass check was called with own_count == 0.
        p_check.assert_called_once()
        _text_arg, count_arg = p_check.call_args.args
        assert count_arg == 0


# ============================================================
# PatchAnchorsMiddleware Tests
# ============================================================

class TestPatchAnchorsMiddleware:
    """Tests for PatchAnchorsMiddleware._patch()."""

    def test_no_op_when_text_or_urls_empty(self):
        assert PatchAnchorsMiddleware._patch("", [], []) == ""
        assert PatchAnchorsMiddleware._patch("hello", [], []) == "hello"


    def test_replaces_file_anchor_with_filename_label(self):
        text = "see [file:doc.pdf|ybres:rid-1]"
        out = PatchAnchorsMiddleware._patch(
            text, ["/cache/doc.pdf"], ["application/pdf"],
        )
        assert "[file: doc.pdf → /cache/doc.pdf]" in out


    def test_anchor_kind_image_requires_image_mime(self):
        """An [image|...] anchor with a non-image mime is left alone."""
        text = "[image|ybres:rid]"
        out = PatchAnchorsMiddleware._patch(
            text, ["/cache/odd.bin"], ["application/octet-stream"],
        )
        assert out == text

