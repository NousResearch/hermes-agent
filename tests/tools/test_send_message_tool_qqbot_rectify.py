"""
QQBot media sending rectification tests for PR #40457.
Tests live adapter routing, standalone fallback, target resolution, and
file classification — without real network requests.

Uses asyncio.run() instead of @pytest.mark.asyncio to avoid a hard
dependency on pytest-asyncio (which may not be installed).
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform


def _run(async_fn):
    """Run an async test function synchronously via asyncio.run()."""
    return asyncio.run(async_fn)


# ============================================================================
# QQBot: live adapter routing + standalone fallback (rectification of PR #40457)
# ============================================================================

class TestQqbotLiveAdapterRouting:
    """Verify QQBot media sends are dispatched through the live adapter."""

    def test_live_adapter_used_when_available(self, monkeypatch):
        """When a live QQAdapter exists, it is preferred over standalone."""

        async def _body():
            from tools.send_message_tool import _send_via_adapter

            mock_adapter = MagicMock()
            mock_adapter.send = AsyncMock(return_value=MagicMock(success=True, message_id='msg-1'))

            mock_runner = MagicMock()
            mock_runner.adapters = {Platform.QQBOT: mock_adapter}
            monkeypatch.setattr('gateway.run._gateway_runner_ref', lambda: mock_runner)

            pconfig = SimpleNamespace(extra={}, token='')
            result = await _send_via_adapter(
                Platform.QQBOT, pconfig, 'c2c:test123', 'hello',
                media_files=[], force_document=False,
            )

            mock_adapter.send.assert_called_once()
            assert result.get('success') is True

        _run(_body())

    def test_live_path_does_not_call_standalone(self, monkeypatch):
        """When live adapter succeeds, standalone is never invoked."""

        async def _body():
            from tools.send_message_tool import _send_via_adapter

            standalone_called = []

            async def fake_standalone(*args, **kwargs):
                standalone_called.append(True)
                return {'success': True}

            mock_adapter = MagicMock()
            mock_adapter.send = AsyncMock(return_value=MagicMock(success=True, message_id='msg-1'))
            mock_runner = MagicMock()
            mock_runner.adapters = {Platform.QQBOT: mock_adapter}
            monkeypatch.setattr('gateway.run._gateway_runner_ref', lambda: mock_runner)

            from gateway.platform_registry import PlatformEntry, platform_registry
            platform_registry.register(PlatformEntry(
                name='qqbot', label='QQBot',
                adapter_factory=lambda cfg: None,
                check_fn=lambda: True,
                standalone_sender_fn=fake_standalone,
            ))

            pconfig = SimpleNamespace(extra={}, token='')
            result = await _send_via_adapter(
                Platform.QQBOT, pconfig, 'c2c:test123', 'hello',
                media_files=[], force_document=False,
            )

            assert len(standalone_called) == 0
            assert result.get('success') is True

        _run(_body())

    def test_live_failure_does_not_fallback_to_standalone(self, monkeypatch):
        """Live adapter failure returns error; does NOT retry via standalone."""

        async def _body():
            from tools.send_message_tool import _send_via_adapter

            standalone_called = []

            async def fake_standalone(*args, **kwargs):
                standalone_called.append(True)
                return {'success': True}

            mock_adapter = MagicMock()
            mock_adapter.send = AsyncMock(return_value=MagicMock(success=False, error='connection lost'))
            mock_runner = MagicMock()
            mock_runner.adapters = {Platform.QQBOT: mock_adapter}
            monkeypatch.setattr('gateway.run._gateway_runner_ref', lambda: mock_runner)

            from gateway.platform_registry import PlatformEntry, platform_registry
            platform_registry.register(PlatformEntry(
                name='qqbot', label='QQBot',
                adapter_factory=lambda cfg: None,
                check_fn=lambda: True,
                standalone_sender_fn=fake_standalone,
            ))

            pconfig = SimpleNamespace(extra={}, token='')
            result = await _send_via_adapter(
                Platform.QQBOT, pconfig, 'c2c:test123', 'hello',
            )

            assert len(standalone_called) == 0
            assert 'error' in result or result.get('success') is False

        _run(_body())

    def test_live_exception_propagates_without_standalone_fallback(self, monkeypatch):
        """Live adapter exception is caught and returned; no standalone fallback."""

        async def _body():
            from tools.send_message_tool import _send_via_adapter

            standalone_called = []

            async def fake_standalone(*args, **kwargs):
                standalone_called.append(True)
                return {'success': True}

            mock_adapter = MagicMock()
            mock_adapter.send = AsyncMock(side_effect=RuntimeError('boom'))
            mock_runner = MagicMock()
            mock_runner.adapters = {Platform.QQBOT: mock_adapter}
            monkeypatch.setattr('gateway.run._gateway_runner_ref', lambda: mock_runner)

            from gateway.platform_registry import PlatformEntry, platform_registry
            platform_registry.register(PlatformEntry(
                name='qqbot', label='QQBot',
                adapter_factory=lambda cfg: None,
                check_fn=lambda: True,
                standalone_sender_fn=fake_standalone,
            ))

            pconfig = SimpleNamespace(extra={}, token='')
            result = await _send_via_adapter(
                Platform.QQBOT, pconfig, 'c2c:test123', 'hello',
            )

            assert len(standalone_called) == 0
            assert 'error' in result

        _run(_body())


class TestQqbotStandaloneFallback:
    """Verify standalone sender is used when no live adapter exists."""

    def test_standalone_called_when_no_runner(self, monkeypatch):
        """When gateway runner is None, standalone_sender_fn is called."""

        async def _body():
            from tools.send_message_tool import _send_via_adapter

            standalone_calls = []

            async def fake_standalone(pconfig, chat_id, message, **kwargs):
                standalone_calls.append((chat_id, message, kwargs))
                return {'success': True, 'message_id': 'standalone-1'}

            monkeypatch.setattr('gateway.run._gateway_runner_ref', lambda: None)

            from gateway.platform_registry import PlatformEntry, platform_registry
            platform_registry.register(PlatformEntry(
                name='qqbot', label='QQBot',
                adapter_factory=lambda cfg: None,
                check_fn=lambda: True,
                standalone_sender_fn=fake_standalone,
            ))

            pconfig = SimpleNamespace(extra={}, token='')
            result = await _send_via_adapter(
                Platform.QQBOT, pconfig, 'c2c:test123', 'hello',
                media_files=[], force_document=False,
            )

            assert len(standalone_calls) == 1
            assert result.get('success') is True

        _run(_body())

    def test_standalone_receives_media_and_force_document(self, monkeypatch):
        """Standalone sender receives media_files and force_document params."""

        async def _body():
            from tools.send_message_tool import _send_via_adapter

            standalone_kwargs = {}

            async def fake_standalone(*args, **kwargs):
                standalone_kwargs.update(kwargs)
                return {'success': True}

            monkeypatch.setattr('gateway.run._gateway_runner_ref', lambda: None)

            from gateway.platform_registry import PlatformEntry, platform_registry
            platform_registry.register(PlatformEntry(
                name='qqbot', label='QQBot',
                adapter_factory=lambda cfg: None,
                check_fn=lambda: True,
                standalone_sender_fn=fake_standalone,
            ))

            media = [('/tmp/test.jpg', False)]
            pconfig = SimpleNamespace(extra={}, token='')
            await _send_via_adapter(
                Platform.QQBOT, pconfig, 'c2c:test123', 'hello',
                media_files=media, force_document=True,
            )

            assert standalone_kwargs.get('force_document') is True
            assert standalone_kwargs.get('media_files') == media

        _run(_body())


class TestQqbotNoRegression:
    """Verify QQBot MEDIA: routing no longer rejects or omits media."""

    def test_qqbot_media_not_unsupported(self, monkeypatch):
        """QQBot with MEDIA: tag is routed to _send_via_adapter, not rejected."""

        async def _body():
            from tools.send_message_tool import _send_to_platform

            adapter_calls = []

            async def fake_send_via_adapter(platform, pconfig, chat_id, chunk,
                                            thread_id=None, media_files=None,
                                            force_document=False):
                adapter_calls.append({
                    'platform': platform, 'chat_id': chat_id,
                    'media_files': media_files, 'force_document': force_document,
                })
                return {'success': True, 'message_id': 'via-adapter'}

            monkeypatch.setattr(
                'tools.send_message_tool._send_via_adapter',
                fake_send_via_adapter,
            )

            pconfig = SimpleNamespace(extra={}, token='')
            await _send_to_platform(
                Platform.QQBOT, pconfig, 'c2c:test123', 'hello',
                media_files=[('/tmp/test.jpg', False)],
                force_document=False,
            )

            assert len(adapter_calls) > 0
            assert adapter_calls[0]['media_files'] is not None

        _run(_body())

    def test_qqbot_text_only_routes_via_adapter(self, monkeypatch):
        """QQBot text-only message also routes through _send_via_adapter."""

        async def _body():
            from tools.send_message_tool import _send_to_platform

            adapter_calls = []

            async def fake_send_via_adapter(*args, **kwargs):
                adapter_calls.append(kwargs)
                return {'success': True}

            monkeypatch.setattr(
                'tools.send_message_tool._send_via_adapter',
                fake_send_via_adapter,
            )

            pconfig = SimpleNamespace(extra={}, token='')
            await _send_to_platform(
                Platform.QQBOT, pconfig, 'c2c:test123', 'hello',
                media_files=[], force_document=False,
            )

            assert len(adapter_calls) > 0
            assert adapter_calls[0].get('media_files') == []

        _run(_body())


class TestQqbotTargetResolution:
    """Verify standalone target resolution from chat_id prefixes."""

    def test_resolve_target_c2c_prefix(self):
        from gateway.platforms.qqbot.standalone import _resolve_target
        ttype, tid = _resolve_target('c2c:abc123')
        assert ttype == 'c2c'
        assert tid == 'abc123'

    def test_resolve_target_user_prefix(self):
        from gateway.platforms.qqbot.standalone import _resolve_target
        ttype, tid = _resolve_target('user:xyz')
        assert ttype == 'c2c'
        assert tid == 'xyz'

    def test_resolve_target_group_prefix(self):
        from gateway.platforms.qqbot.standalone import _resolve_target
        ttype, tid = _resolve_target('group:def456')
        assert ttype == 'group'
        assert tid == 'def456'

    def test_resolve_target_guild_prefix(self):
        from gateway.platforms.qqbot.standalone import _resolve_target
        ttype, tid = _resolve_target('guild:ch1')
        assert ttype == 'guild'
        assert tid == 'ch1'

    def test_resolve_target_raw_openid_defaults_to_c2c(self):
        from gateway.platforms.qqbot.standalone import _resolve_target
        ttype, tid = _resolve_target('openid_value')
        assert ttype == 'c2c'
        assert tid == 'openid_value'

    def test_resolve_target_empty(self):
        from gateway.platforms.qqbot.standalone import _resolve_target
        ttype, tid = _resolve_target('')
        assert ttype == 'unknown'
        assert tid == ''


class TestQqbotFileClassification:
    """Verify standalone file-type classification."""

    def test_image_ext_classifies_as_image(self):
        from gateway.platforms.qqbot.standalone import _classify_file
        from gateway.platforms.qqbot.constants import MEDIA_TYPE_IMAGE
        assert _classify_file('.jpg') == MEDIA_TYPE_IMAGE
        assert _classify_file('.png') == MEDIA_TYPE_IMAGE

    def test_video_ext_classifies_as_video(self):
        from gateway.platforms.qqbot.standalone import _classify_file
        from gateway.platforms.qqbot.constants import MEDIA_TYPE_VIDEO
        assert _classify_file('.mp4') == MEDIA_TYPE_VIDEO

    def test_voice_ext_classifies_as_voice(self):
        from gateway.platforms.qqbot.standalone import _classify_file
        from gateway.platforms.qqbot.constants import MEDIA_TYPE_VOICE
        assert _classify_file('.silk') == MEDIA_TYPE_VOICE
        assert _classify_file('.mp3') == MEDIA_TYPE_VOICE

    def test_unknown_ext_classifies_as_file(self):
        from gateway.platforms.qqbot.standalone import _classify_file
        from gateway.platforms.qqbot.constants import MEDIA_TYPE_FILE
        assert _classify_file('.pdf') == MEDIA_TYPE_FILE
        assert _classify_file('.zip') == MEDIA_TYPE_FILE

    def test_force_document_overrides_classification(self):
        from gateway.platforms.qqbot.standalone import _classify_file
        from gateway.platforms.qqbot.constants import MEDIA_TYPE_FILE
        assert _classify_file('.jpg', force_document=True) == MEDIA_TYPE_FILE
        assert _classify_file('.mp4', force_document=True) == MEDIA_TYPE_FILE
        assert _classify_file('.silk', force_document=True) == MEDIA_TYPE_FILE

    def test_case_insensitive_ext(self):
        from gateway.platforms.qqbot.standalone import _classify_file
        from gateway.platforms.qqbot.constants import MEDIA_TYPE_IMAGE
        assert _classify_file('.JPG') == MEDIA_TYPE_IMAGE
        assert _classify_file('.Png') == MEDIA_TYPE_IMAGE
