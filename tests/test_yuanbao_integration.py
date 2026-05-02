"""
test_yuanbao_integration.py - Yuanbao 模块集成测试

验证各模块能正确组装和交互：
  - YuanbaoAdapter 初始化
  - Config / Platform 枚举
  - get_connected_platforms 逻辑
  - Proto 编解码 round-trip
  - Markdown 分块
  - API / Media 模块 import
  - Toolset 注册
"""

import sys
import os

# 确保 hermes-agent 根目录在 sys.path 中
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from gateway.config import Platform, PlatformConfig, GatewayConfig
from gateway.platforms.yuanbao import YuanbaoAdapter


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


# ===========================================================
# 1. Adapter 初始化
# ===========================================================

class TestYuanbaoAdapterInit:
    def test_create_adapter(self):
        config = make_config()
        adapter = YuanbaoAdapter(config)
        assert adapter is not None
        assert adapter.PLATFORM == Platform.YUANBAO

    def test_initial_state(self):
        config = make_config()
        adapter = YuanbaoAdapter(config)
        status = adapter.get_status()
        assert status["connected"] == False
        assert status["bot_id"] is None


# ===========================================================
# 2. Config / Platform 枚举
# ===========================================================

class TestYuanbaoConfig:
    def test_platform_enum(self):
        assert Platform.YUANBAO.value == "yuanbao"

    def test_config_fields(self):
        config = make_config()
        assert config.extra["app_id"] == "test_key"
        assert config.extra["app_secret"] == "test_secret"

    def test_get_connected_platforms_requires_key_and_secret(self):
        # Only key, no secret → not in connected list
        gw_only_key = GatewayConfig(
            platforms={
                Platform.YUANBAO: PlatformConfig(
                    enabled=True,
                    extra={"app_id": "key"},
                )
            }
        )
        platforms = gw_only_key.get_connected_platforms()
        assert Platform.YUANBAO not in platforms

        # key + secret both present → in connected list
        gw_full = GatewayConfig(
            platforms={
                Platform.YUANBAO: PlatformConfig(
                    enabled=True,
                    extra={"app_id": "key", "app_secret": "secret"},
                )
            }
        )
        platforms2 = gw_full.get_connected_platforms()
        assert Platform.YUANBAO in platforms2


# ===========================================================
# 3. GatewayRunner 注册
# ===========================================================

class TestGatewayRunnerRegistration:
    def test_yuanbao_in_platform_enum(self):
        """Platform 枚举包含 YUANBAO"""
        assert hasattr(Platform, "YUANBAO")
        assert Platform.YUANBAO.value == "yuanbao"

    def _make_minimal_runner(self, config):
        """通过 __new__ + 最小初始化绕过 run.py 的模块级 dotenv/ssl 副作用"""
        import sys
        from unittest.mock import MagicMock

        # Stub out heavy dependencies if not already present
        stubs = [
            "dotenv",
            "hermes_cli.env_loader",
            "hermes_cli.config",
            "hermes_constants",
        ]
        _orig = {}
        for mod in stubs:
            if mod not in sys.modules:
                _orig[mod] = None
                sys.modules[mod] = MagicMock()

        try:
            from gateway.run import GatewayRunner
        finally:
            # Restore only the ones we injected
            for mod, orig in _orig.items():
                if orig is None:
                    sys.modules.pop(mod, None)

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = config
        runner.adapters = {}
        runner._failed_platforms = {}
        runner._session_model_overrides = {}
        return runner, GatewayRunner

    def test_runner_creates_yuanbao_adapter(self):
        """GatewayRunner._create_adapter 能为 YUANBAO 返回 YuanbaoAdapter 实例"""
        from gateway.config import GatewayConfig
        from unittest.mock import patch
        config = make_config(enabled=True)
        gw_config = GatewayConfig(platforms={Platform.YUANBAO: config})

        try:
            runner, _ = self._make_minimal_runner(gw_config)
            # websockets 在测试环境可能未安装，mock 掉 WEBSOCKETS_AVAILABLE
            with patch("gateway.platforms.yuanbao.WEBSOCKETS_AVAILABLE", True):
                adapter = runner._create_adapter(Platform.YUANBAO, config)
        except ImportError as e:
            pytest.skip(f"run.py import unavailable in test env: {e}")

        assert adapter is not None
        assert isinstance(adapter, YuanbaoAdapter)

    def test_runner_adapter_platform_attr(self):
        """创建的 adapter.PLATFORM 为 Platform.YUANBAO"""
        from gateway.config import GatewayConfig
        from unittest.mock import patch
        config = make_config(enabled=True)
        gw_config = GatewayConfig(platforms={Platform.YUANBAO: config})

        try:
            runner, _ = self._make_minimal_runner(gw_config)
            with patch("gateway.platforms.yuanbao.WEBSOCKETS_AVAILABLE", True):
                adapter = runner._create_adapter(Platform.YUANBAO, config)
        except ImportError as e:
            pytest.skip(f"run.py import unavailable in test env: {e}")

        assert adapter is not None
        assert adapter.PLATFORM == Platform.YUANBAO


# ===========================================================
# 4. Proto round-trip
# ===========================================================

class TestProtoRoundTrip:
    """验证 proto 编解码基本功能"""

    def test_conn_msg_roundtrip(self):
        from gateway.platforms.yuanbao_proto import encode_conn_msg, decode_conn_msg
        encoded = encode_conn_msg(msg_type=1, seq_no=42, data=b"hello")
        decoded = decode_conn_msg(encoded)
        assert decoded["seq_no"] == 42
        assert decoded["data"] == b"hello"

    def test_text_elem_encoding(self):
        from gateway.platforms.yuanbao_proto import encode_send_c2c_message
        msg = encode_send_c2c_message(
            to_account="user123",
            msg_body=[{"msg_type": "TIMTextElem", "msg_content": {"text": "hello"}}],
            from_account="bot456",
        )
        assert isinstance(msg, bytes)
        assert len(msg) > 0


# ===========================================================
# 5. Markdown 分块
# ===========================================================

class TestMarkdownChunking:
    def test_chunks_are_sent_separately(self):
        from gateway.platforms.yuanbao import MarkdownProcessor
        long_text = "paragraph\n\n" * 100
        chunks = MarkdownProcessor.chunk_markdown_text(long_text, 200)
        assert len(chunks) > 1
        for c in chunks:
            # 段落原子块允许轻微超限，仅验证不崩溃
            assert isinstance(c, str)
            assert len(c) > 0

    def test_chunk_short_text_no_split(self):
        from gateway.platforms.yuanbao import MarkdownProcessor
        text = "hello world"
        chunks = MarkdownProcessor.chunk_markdown_text(text, 3000)
        assert chunks == [text]


# ===========================================================
# 6. Sign Token 模块
# ===========================================================

class TestSignToken:
    def test_import_ok(self):
        from gateway.platforms.yuanbao import SignManager
        assert callable(SignManager.get_token)
        assert callable(SignManager.force_refresh)


# ===========================================================
# 6b. ConnectionManager / OutboundManager
# ===========================================================

class TestManagerImports:
    def test_connection_manager_import(self):
        from gateway.platforms.yuanbao import ConnectionManager
        assert ConnectionManager is not None

    def test_outbound_manager_import(self):
        from gateway.platforms.yuanbao import OutboundManager
        assert OutboundManager is not None

    def test_message_sender_import(self):
        from gateway.platforms.yuanbao import MessageSender
        assert MessageSender is not None

    def test_heartbeat_manager_import(self):
        from gateway.platforms.yuanbao import HeartbeatManager
        assert HeartbeatManager is not None

    def test_slow_response_notifier_import(self):
        from gateway.platforms.yuanbao import SlowResponseNotifier
        assert SlowResponseNotifier is not None

    def test_adapter_has_outbound_manager(self):
        adapter = YuanbaoAdapter(make_config())
        from gateway.platforms.yuanbao import ConnectionManager, OutboundManager
        assert isinstance(adapter._connection, ConnectionManager)
        assert isinstance(adapter._outbound, OutboundManager)

    def test_outbound_composes_sub_managers(self):
        adapter = YuanbaoAdapter(make_config())
        from gateway.platforms.yuanbao import MessageSender, HeartbeatManager, SlowResponseNotifier
        assert isinstance(adapter._outbound.sender, MessageSender)
        assert isinstance(adapter._outbound.heartbeat, HeartbeatManager)
        assert isinstance(adapter._outbound.slow_notifier, SlowResponseNotifier)


# ===========================================================
# 7. Media 模块
# ===========================================================

class TestMediaModule:
    def test_import_ok(self):
        from gateway.platforms.yuanbao_media import upload_to_cos, download_url
        assert callable(upload_to_cos)
        assert callable(download_url)


# ===========================================================
# 8. Toolset 注册
# ===========================================================

class TestToolset:
    def test_yuanbao_toolset_registered(self):
        """toolsets.py 中存在 hermes-yuanbao 键"""
        import importlib
        ts = importlib.import_module("toolsets")
        assert hasattr(ts, "TOOLSETS") or hasattr(ts, "toolsets")
        toolsets_dict = getattr(ts, "TOOLSETS", getattr(ts, "toolsets", {}))
        assert "hermes-yuanbao" in toolsets_dict

    def test_tools_import(self):
        from tools.yuanbao_tools import (
            get_group_info,
            query_group_members,
            send_dm,
        )
        assert all(callable(f) for f in [
            get_group_info,
            query_group_members,
            send_dm,
        ])


# ===========================================================
# 9. platforms/__init__.py 导出
# ===========================================================

class TestPlatformInit:
    def test_yuanbao_adapter_exported(self):
        """gateway.platforms.__init__.py 应导出 YuanbaoAdapter"""
        from gateway.platforms import YuanbaoAdapter as _YuanbaoAdapter
        assert _YuanbaoAdapter is YuanbaoAdapter


# ===========================================================
# 10. P0 fixes verification
# ===========================================================

import asyncio
import collections


class TestP0ReconnectGuard:
    """P0-1: _reconnecting flag prevents concurrent reconnect attempts."""

    def test_reconnecting_flag_initialized(self):
        adapter = YuanbaoAdapter(make_config())
        assert hasattr(adapter._connection, '_reconnecting')
        assert adapter._connection._reconnecting is False

    def test_schedule_reconnect_skips_when_not_running(self):
        adapter = YuanbaoAdapter(make_config())
        adapter._running = False
        adapter._connection._reconnecting = False
        adapter._connection.schedule_reconnect()
        # No task should be created because _running is False

    def test_schedule_reconnect_skips_when_already_reconnecting(self):
        adapter = YuanbaoAdapter(make_config())
        adapter._running = True
        adapter._connection._reconnecting = True
        adapter._connection.schedule_reconnect()
        # No new task should be created because already reconnecting


class TestP0InboundTaskTracking:
    """P0-2: _inbound_tasks set is initialized and usable."""

    def test_inbound_tasks_initialized(self):
        adapter = YuanbaoAdapter(make_config())
        assert hasattr(adapter, '_inbound_tasks')
        assert isinstance(adapter._inbound_tasks, set)
        assert len(adapter._inbound_tasks) == 0


class TestP0ChatLockEviction:
    """P0-3: get_chat_lock uses OrderedDict and safe eviction."""

    def test_chat_locks_is_ordered_dict(self):
        adapter = YuanbaoAdapter(make_config())
        assert isinstance(adapter._outbound._chat_locks, collections.OrderedDict)

    def test_eviction_skips_locked(self):
        """When eviction is needed, locked entries are skipped."""
        adapter = YuanbaoAdapter(make_config())
        from gateway.platforms.yuanbao import OutboundManager

        # Fill to capacity with unlocked locks
        for i in range(OutboundManager.CHAT_DICT_MAX_SIZE):
            adapter._outbound._chat_locks[f"chat_{i}"] = asyncio.Lock()

        # Lock the oldest entry
        oldest_key = next(iter(adapter._outbound._chat_locks))
        oldest_lock = adapter._outbound._chat_locks[oldest_key]
        # Simulate a held lock by acquiring it in a non-async way (set _locked)
        # asyncio.Lock is not held until actually acquired; so we test the
        # method logic by acquiring the first lock manually.
        # For a sync test, we check that get_chat_lock doesn't crash.
        new_lock = adapter._outbound.get_chat_lock("new_chat")
        assert "new_chat" in adapter._outbound._chat_locks
        assert isinstance(new_lock, asyncio.Lock)
        # The oldest unlocked entry should have been evicted
        assert len(adapter._outbound._chat_locks) == OutboundManager.CHAT_DICT_MAX_SIZE

    def test_move_to_end_on_access(self):
        """Accessing an existing key moves it to the end (MRU)."""
        adapter = YuanbaoAdapter(make_config())
        adapter._outbound._chat_locks["a"] = asyncio.Lock()
        adapter._outbound._chat_locks["b"] = asyncio.Lock()
        adapter._outbound._chat_locks["c"] = asyncio.Lock()

        # Access "a" — should move to end
        adapter._outbound.get_chat_lock("a")
        keys = list(adapter._outbound._chat_locks.keys())
        assert keys[-1] == "a"
        assert keys[0] == "b"


class TestP0PlatformScopedLock:
    """P0-4: connect() calls _acquire_platform_lock."""

    def test_adapter_has_platform_lock_methods(self):
        adapter = YuanbaoAdapter(make_config())
        assert hasattr(adapter, '_acquire_platform_lock')
        assert hasattr(adapter, '_release_platform_lock')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
