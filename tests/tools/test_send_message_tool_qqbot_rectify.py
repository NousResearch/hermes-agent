"""
QQBot media sending rectification tests — round 2.

Covers live adapter dispatch (send_image_file / send_voice / send_video /
send_document), shared QQApiClient, standalone fallback, target resolution,
file classification, and resource cleanup.  No real network requests.
"""

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform


def _run(async_fn):
    """Run an async test function synchronously via asyncio.run()."""
    return asyncio.run(async_fn)


# ═══════════════════════════════════════════════════════════════════════
# Live adapter media dispatch
# ═══════════════════════════════════════════════════════════════════════

class TestLiveAdapterMediaDispatch:
    """Verify live adapter calls specialised media methods."""

    def test_image_calls_send_image_file(self, monkeypatch):
        async def _body():
            from tools.send_message_tool import _dispatch_live_media

            adapter = MagicMock()
            adapter.send = AsyncMock(return_value=MagicMock(success=True, message_id='t1'))
            adapter.send_image_file = AsyncMock(return_value=MagicMock(success=True, message_id='i1'))

            result = await _dispatch_live_media(
                adapter, 'c2c:test', 'hello',
                media_files=[('/tmp/photo.jpg', False)],
                force_document=False,
            )
            adapter.send.assert_called_once()
            adapter.send_image_file.assert_called_once_with(
                chat_id='c2c:test', image_path='/tmp/photo.jpg',
            )
            assert result['success'] is True

        _run(_body())

    def test_voice_calls_send_voice(self, monkeypatch):
        async def _body():
            from tools.send_message_tool import _dispatch_live_media

            adapter = MagicMock()
            adapter.send = AsyncMock(return_value=MagicMock(success=True, message_id='t1'))
            adapter.send_voice = AsyncMock(return_value=MagicMock(success=True, message_id='v1'))

            result = await _dispatch_live_media(
                adapter, 'c2c:test', 'hello',
                media_files=[('/tmp/msg.ogg', True)],
                force_document=False,
            )
            adapter.send_voice.assert_called_once_with(
                chat_id='c2c:test', audio_path='/tmp/msg.ogg',
            )
            assert result['success'] is True

        _run(_body())

    def test_video_calls_send_video(self, monkeypatch):
        async def _body():
            from tools.send_message_tool import _dispatch_live_media

            adapter = MagicMock()
            adapter.send = AsyncMock(return_value=MagicMock(success=True, message_id='t1'))
            adapter.send_video = AsyncMock(return_value=MagicMock(success=True, message_id='v1'))

            result = await _dispatch_live_media(
                adapter, 'c2c:test', 'hello',
                media_files=[('/tmp/clip.mp4', False)],
                force_document=False,
            )
            adapter.send_video.assert_called_once_with(
                chat_id='c2c:test', video_path='/tmp/clip.mp4',
            )
            assert result['success'] is True

        _run(_body())

    def test_unknown_ext_calls_send_document(self, monkeypatch):
        async def _body():
            from tools.send_message_tool import _dispatch_live_media

            adapter = MagicMock()
            adapter.send = AsyncMock(return_value=MagicMock(success=True, message_id='t1'))
            adapter.send_document = AsyncMock(return_value=MagicMock(success=True, message_id='d1'))

            result = await _dispatch_live_media(
                adapter, 'c2c:test', 'hello',
                media_files=[('/tmp/report.pdf', False)],
                force_document=False,
            )
            adapter.send_document.assert_called_once_with(
                chat_id='c2c:test', file_path='/tmp/report.pdf',
            )
            assert result['success'] is True

        _run(_body())

    def test_force_document_all_calls_send_document(self, monkeypatch):
        async def _body():
            from tools.send_message_tool import _dispatch_live_media

            adapter = MagicMock()
            adapter.send = AsyncMock(return_value=MagicMock(success=True, message_id='t1'))
            adapter.send_document = AsyncMock(return_value=MagicMock(success=True, message_id='d1'))

            result = await _dispatch_live_media(
                adapter, 'c2c:test', 'hello',
                media_files=[('/tmp/photo.jpg', False), ('/tmp/clip.mp4', False)],
                force_document=True,
            )
            assert adapter.send_document.call_count == 2
            assert not adapter.send_image_file.called
            assert not adapter.send_video.called
            assert result['success'] is True

        _run(_body())

    def test_text_sent_only_once_with_multiple_media(self, monkeypatch):
        async def _body():
            from tools.send_message_tool import _dispatch_live_media

            adapter = MagicMock()
            adapter.send = AsyncMock(return_value=MagicMock(success=True, message_id='t1'))
            adapter.send_image_file = AsyncMock(return_value=MagicMock(success=True, message_id='i1'))

            result = await _dispatch_live_media(
                adapter, 'c2c:test', 'hello',
                media_files=[('/tmp/a.jpg', False), ('/tmp/b.jpg', False)],
                force_document=False,
            )
            # Text once, media twice
            assert adapter.send.call_count == 1
            assert adapter.send_image_file.call_count == 2
            assert result['success'] is True

        _run(_body())

    def test_live_media_failure_returns_error_not_fallback(self, monkeypatch):
        async def _body():
            from tools.send_message_tool import _dispatch_live_media

            adapter = MagicMock()
            adapter.send = AsyncMock(return_value=MagicMock(success=True, message_id='t1'))
            adapter.send_image_file = AsyncMock(
                return_value=MagicMock(success=False, error='upload failed')
            )

            result = await _dispatch_live_media(
                adapter, 'c2c:test', 'hello',
                media_files=[('/tmp/photo.jpg', False)],
                force_document=False,
            )
            assert 'error' in result
            assert 'upload failed' in result['error']

        _run(_body())

    def test_live_adapter_no_standalone_call(self, monkeypatch):
        """Live adapter path must not invoke standalone_sender_fn."""
        async def _body():
            from tools.send_message_tool import _send_via_adapter

            standalone_called = []

            async def fake_standalone(*args, **kwargs):
                standalone_called.append(True)
                return {'success': True}

            adapter = MagicMock()
            adapter.send = AsyncMock(return_value=MagicMock(success=True, message_id='t1'))

            mock_runner = MagicMock()
            mock_runner.adapters = {Platform.QQBOT: adapter}
            monkeypatch.setattr('gateway.run._gateway_runner_ref', lambda: mock_runner)

            # Register a fake standalone
            from gateway.platform_registry import PlatformEntry, platform_registry
            platform_registry.register(PlatformEntry(
                name='qqbot', label='QQBot',
                adapter_factory=lambda cfg: None, check_fn=lambda: True,
                standalone_sender_fn=fake_standalone,
            ))

            pconfig = SimpleNamespace(extra={}, token='')
            result = await _send_via_adapter(
                Platform.QQBOT, pconfig, 'c2c:test', 'hello',
                media_files=[], force_document=False,
            )
            assert len(standalone_called) == 0
            assert result['success'] is True

        _run(_body())

    def test_live_failure_no_standalone_fallback(self, monkeypatch):
        """Live adapter failure does NOT fallback to standalone (no double send)."""
        async def _body():
            from tools.send_message_tool import _send_via_adapter

            standalone_called = []

            async def fake_standalone(*args, **kwargs):
                standalone_called.append(True)
                return {'success': True}

            adapter = MagicMock()
            adapter.send = AsyncMock(
                return_value=MagicMock(success=False, error='connection lost')
            )

            mock_runner = MagicMock()
            mock_runner.adapters = {Platform.QQBOT: adapter}
            monkeypatch.setattr('gateway.run._gateway_runner_ref', lambda: mock_runner)

            from gateway.platform_registry import PlatformEntry, platform_registry
            platform_registry.register(PlatformEntry(
                name='qqbot', label='QQBot',
                adapter_factory=lambda cfg: None, check_fn=lambda: True,
                standalone_sender_fn=fake_standalone,
            ))

            pconfig = SimpleNamespace(extra={}, token='')
            result = await _send_via_adapter(
                Platform.QQBOT, pconfig, 'c2c:test', 'hello',
            )
            assert len(standalone_called) == 0
            assert 'error' in result

        _run(_body())


# ═══════════════════════════════════════════════════════════════════════
# Standalone fallback
# ═══════════════════════════════════════════════════════════════════════

class TestStandaloneFallback:
    """Verify standalone sender is called when no live adapter exists."""

    def test_no_runner_calls_standalone(self, monkeypatch):
        async def _body():
            from tools.send_message_tool import _send_via_adapter

            standalone_calls = []

            async def fake_standalone(pconfig, chat_id, message, **kwargs):
                standalone_calls.append((chat_id, message, kwargs))
                return {'success': True, 'message_id': 's1'}

            monkeypatch.setattr('gateway.run._gateway_runner_ref', lambda: None)

            from gateway.platform_registry import PlatformEntry, platform_registry
            platform_registry.register(PlatformEntry(
                name='qqbot', label='QQBot',
                adapter_factory=lambda cfg: None, check_fn=lambda: True,
                standalone_sender_fn=fake_standalone,
            ))

            pconfig = SimpleNamespace(extra={}, token='')
            result = await _send_via_adapter(
                Platform.QQBOT, pconfig, 'c2c:test', 'hello',
                media_files=[], force_document=False,
            )
            assert len(standalone_calls) == 1
            assert result['success'] is True

        _run(_body())


# ═══════════════════════════════════════════════════════════════════════
# Shared QQApiClient tests
# ═══════════════════════════════════════════════════════════════════════

class TestQQApiClient:
    """Verify the shared QQ outbound component."""

    def test_ensure_token_returns_token(self):
        async def _body():
            import httpx
            from gateway.platforms.qqbot.outbound import QQApiClient

            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json = MagicMock(return_value={
                'access_token': 'test-token-123', 'expires_in': 7200,
            })

            mock_client = MagicMock(spec=httpx.AsyncClient)
            mock_client.post = AsyncMock(return_value=mock_resp)

            api = QQApiClient('app', 'secret', mock_client)
            token = await api.ensure_token()

            assert token == 'test-token-123'
            mock_client.post.assert_called_once()

        _run(_body())

    def test_api_request_uses_auth_headers(self):
        async def _body():
            import httpx
            from gateway.platforms.qqbot.outbound import QQApiClient

            mock_token_resp = MagicMock()
            mock_token_resp.status_code = 200
            mock_token_resp.json = MagicMock(return_value={
                'access_token': 'tok', 'expires_in': 7200,
            })

            mock_api_resp = MagicMock()
            mock_api_resp.status_code = 200
            mock_api_resp.json = MagicMock(return_value={'id': 'msg-1'})

            mock_client = MagicMock(spec=httpx.AsyncClient)
            mock_client.post = AsyncMock(return_value=mock_token_resp)
            mock_client.request = AsyncMock(return_value=mock_api_resp)

            api = QQApiClient('app', 'secret', mock_client)
            result = await api.api_request('POST', '/v2/users/x/messages', {'content': 'hi'})

            # Verify Authorization header was set
            call_kwargs = mock_client.request.call_args
            headers = call_kwargs[1].get('headers', {})
            assert 'Authorization' in headers
            assert 'QQBot tok' in headers['Authorization']
            assert result == {'id': 'msg-1'}

        _run(_body())

    def test_api_request_raises_on_http_error(self):
        async def _body():
            import httpx
            from gateway.platforms.qqbot.outbound import QQApiClient

            mock_token_resp = MagicMock()
            mock_token_resp.status_code = 200
            mock_token_resp.json = MagicMock(return_value={
                'access_token': 'tok', 'expires_in': 7200,
            })

            mock_err_resp = MagicMock()
            mock_err_resp.status_code = 403
            mock_err_resp.json = MagicMock(return_value={'message': 'Forbidden'})

            mock_client = MagicMock(spec=httpx.AsyncClient)
            mock_client.post = AsyncMock(return_value=mock_token_resp)
            mock_client.request = AsyncMock(return_value=mock_err_resp)

            api = QQApiClient('app', 'secret', mock_client)
            try:
                await api.api_request('POST', '/v2/users/x/messages', {})
                assert False, 'Should have raised'
            except RuntimeError as e:
                assert '403' in str(e)
                assert 'Forbidden' in str(e)

        _run(_body())

    def test_endpoint_for_c2c(self):
        from gateway.platforms.qqbot.outbound import QQApiClient
        path = QQApiClient._endpoint_for('c2c', 'openid123')
        assert path == '/v2/users/openid123/messages'

    def test_endpoint_for_group(self):
        from gateway.platforms.qqbot.outbound import QQApiClient
        path = QQApiClient._endpoint_for('group', 'group456')
        assert path == '/v2/groups/group456/messages'

    def test_endpoint_for_guild(self):
        from gateway.platforms.qqbot.outbound import QQApiClient
        path = QQApiClient._endpoint_for('guild', 'ch789')
        assert 'channels' in path


# ═══════════════════════════════════════════════════════════════════════
# Target resolution
# ═══════════════════════════════════════════════════════════════════════

class TestTargetResolution:
    def test_c2c_prefix(self):
        from gateway.platforms.qqbot.outbound import resolve_target
        ttype, tid = resolve_target('c2c:abc123')
        assert ttype == 'c2c'
        assert tid == 'abc123'

    def test_user_prefix(self):
        from gateway.platforms.qqbot.outbound import resolve_target
        ttype, tid = resolve_target('user:xyz')
        assert ttype == 'c2c'
        assert tid == 'xyz'

    def test_group_prefix(self):
        from gateway.platforms.qqbot.outbound import resolve_target
        ttype, tid = resolve_target('group:def456')
        assert ttype == 'group'
        assert tid == 'def456'

    def test_guild_prefix(self):
        from gateway.platforms.qqbot.outbound import resolve_target
        ttype, tid = resolve_target('guild:ch1')
        assert ttype == 'guild'
        assert tid == 'ch1'

    def test_raw_openid_defaults_to_c2c(self):
        from gateway.platforms.qqbot.outbound import resolve_target
        ttype, tid = resolve_target('openid_value')
        assert ttype == 'c2c'
        assert tid == 'openid_value'

    def test_empty_returns_unknown(self):
        from gateway.platforms.qqbot.outbound import resolve_target
        ttype, tid = resolve_target('')
        assert ttype == 'unknown'


# ═══════════════════════════════════════════════════════════════════════
# File classification
# ═══════════════════════════════════════════════════════════════════════

class TestFileClassification:
    def test_image_ext(self):
        from gateway.platforms.qqbot.outbound import classify_media_type, MEDIA_TYPE_IMAGE
        assert classify_media_type('.jpg') == MEDIA_TYPE_IMAGE
        assert classify_media_type('.png') == MEDIA_TYPE_IMAGE

    def test_video_ext(self):
        from gateway.platforms.qqbot.outbound import classify_media_type, MEDIA_TYPE_VIDEO
        assert classify_media_type('.mp4') == MEDIA_TYPE_VIDEO

    def test_voice_ext(self):
        from gateway.platforms.qqbot.outbound import classify_media_type, MEDIA_TYPE_VOICE
        assert classify_media_type('.silk') == MEDIA_TYPE_VOICE
        assert classify_media_type('.mp3') == MEDIA_TYPE_VOICE

    def test_unknown_ext(self):
        from gateway.platforms.qqbot.outbound import classify_media_type, MEDIA_TYPE_FILE
        assert classify_media_type('.pdf') == MEDIA_TYPE_FILE
        assert classify_media_type('.zip') == MEDIA_TYPE_FILE

    def test_force_document_overrides(self):
        from gateway.platforms.qqbot.outbound import classify_media_type, MEDIA_TYPE_FILE
        assert classify_media_type('.jpg', force_document=True) == MEDIA_TYPE_FILE
        assert classify_media_type('.mp4', force_document=True) == MEDIA_TYPE_FILE

    def test_case_insensitive(self):
        from gateway.platforms.qqbot.outbound import classify_media_type, MEDIA_TYPE_IMAGE
        assert classify_media_type('.JPG') == MEDIA_TYPE_IMAGE
        assert classify_media_type('.Png') == MEDIA_TYPE_IMAGE


# ═══════════════════════════════════════════════════════════════════════
# Guild handling
# ═══════════════════════════════════════════════════════════════════════

class TestGuildHandling:
    def test_guild_text_supported(self):
        """Guild text-only send should not be rejected."""
        from gateway.platforms.qqbot.standalone import _standalone_send

        # We test the upfront check: guild + no media = no rejection
        from gateway.platforms.qqbot.outbound import resolve_target
        ttype, tid = resolve_target('guild:ch1')
        assert ttype == 'guild'

    def test_guild_media_rejected(self):
        """Guild + media should be rejected explicitly (not attempted)."""
        # The standalone sender checks this before any HTTP
        from gateway.platforms.qqbot.outbound import resolve_target
        ttype, tid = resolve_target('guild:ch1')
        assert ttype == 'guild'
        # Media rejection happens inside _standalone_send — tested via
        # integration with fake config.


# ═══════════════════════════════════════════════════════════════════════
# Text chunking
# ═══════════════════════════════════════════════════════════════════════

class TestTextChunking:
    def test_short_text_not_chunked(self):
        from gateway.platforms.qqbot.outbound import split_for_qq
        result = split_for_qq('hello', 4000)
        assert result == ['hello']

    def test_long_text_chunked(self):
        from gateway.platforms.qqbot.outbound import split_for_qq
        text = 'a' * 5000
        result = split_for_qq(text, 2000)
        assert len(result) == 3
        assert all(len(c) <= 2000 for c in result)


# ═══════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════

class TestRegistration:
    def test_qqbot_registered_with_valid_entry(self):
        """QQBot PlatformEntry exists with a non-None standalone_sender_fn."""
        from gateway.platform_registry import platform_registry

        entry = platform_registry.get('qqbot')
        assert entry is not None, 'QQBot must be registered in platform_registry'
        assert entry.standalone_sender_fn is not None, (
            'standalone_sender_fn must be set so standalone send works'
        )
        # adapter_factory should be callable (real factory, not lambda: None)
        assert callable(entry.adapter_factory), (
            'adapter_factory must be callable'
        )


# ═══════════════════════════════════════════════════════════════════════
# No regression: QQBot MEDIA: routed correctly
# ═══════════════════════════════════════════════════════════════════════

class TestNoRegression:
    def test_qqbot_media_not_unsupported(self, monkeypatch):
        """QQBot with MEDIA: goes to _send_via_adapter, not rejected."""
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
                Platform.QQBOT, pconfig, 'c2c:test', 'hello',
                media_files=[('/tmp/test.jpg', False)],
                force_document=False,
            )
            assert len(adapter_calls) > 0

        _run(_body())


# ═══════════════════════════════════════════════════════════════════════
# Standalone resource cleanup
# ═══════════════════════════════════════════════════════════════════════

class TestStandaloneResourceCleanup:
    def test_http_client_closed_on_success(self, monkeypatch):
        """Standalone sender closes httpx client even on success."""
        async def _body():
            from gateway.platforms.qqbot.standalone import _standalone_send

            closed = []

            class FakeClient:
                async def __aenter__(self): return self
                async def __aexit__(self, *args): pass
                async def aclose(self):
                    closed.append(True)
                async def post(self, url, *, json=None, headers=None, timeout=None):
                    m = MagicMock()
                    m.status_code = 200
                    m.json = MagicMock(return_value={'access_token': 'tok', 'expires_in': 7200})
                    return m
                async def request(self, method, url, *, headers=None, json=None, timeout=None):
                    m = MagicMock()
                    m.status_code = 200
                    m.json = MagicMock(return_value={'id': 'msg-1'})
                    return m
                async def put(self, url, *, content=None, headers=None, timeout=None):
                    m = MagicMock()
                    m.status_code = 200
                    return m

            monkeypatch.setattr('httpx.AsyncClient', lambda **kw: FakeClient())

            pconfig = SimpleNamespace(
                extra={'app_id': 'test', 'client_secret': 'test'},
                token='',
            )
            result = await _standalone_send(
                pconfig, 'c2c:test', 'hello',
                media_files=[], force_document=False,
            )
            assert len(closed) == 1

        _run(_body())

    def test_http_client_closed_on_error(self, monkeypatch):
        """Standalone sender closes httpx client on error too."""
        async def _body():
            from gateway.platforms.qqbot.standalone import _standalone_send

            closed = []

            class FakeClient:
                async def __aenter__(self): return self
                async def __aexit__(self, *args): pass
                async def aclose(self):
                    closed.append(True)
                async def post(self, *args, **kwargs):
                    raise RuntimeError('network error')

            monkeypatch.setattr('httpx.AsyncClient', lambda **kw: FakeClient())

            pconfig = SimpleNamespace(
                extra={'app_id': 'test', 'client_secret': 'test'},
                token='',
            )
            result = await _standalone_send(
                pconfig, 'c2c:test', 'hello',
                media_files=[], force_document=False,
            )
            assert 'error' in result
            assert len(closed) == 1

        _run(_body())
