"""Mattermost send-tool native and standalone file delivery contracts."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from tools.send_message_tool import _send_to_platform, send_message_tool


@pytest.fixture(autouse=True)
def _bind_media_test_root(tmp_path, monkeypatch):
    from gateway.platforms import base as platform_base
    monkeypatch.setattr(platform_base, "_HERMES_ROOT", tmp_path)
    monkeypatch.setattr(platform_base, "MEDIA_DELIVERY_SAFE_ROOTS",
                        tuple(tmp_path / "cache" / name for name in platform_base._MEDIA_DELIVERY_CACHE_SUBDIRS))


class TestMattermostFileDelivery:
    @pytest.mark.parametrize("native_path", [r"C:\images\space %23 #.png", r"\\server\share\space %23 #.png"])
    @pytest.mark.asyncio
    async def test_live_media_keeps_windows_native_paths(self, transport, monkeypatch, native_path):
        from plugins.platforms.mattermost.adapter import MattermostAdapter
        config, session, posts, uploads = transport
        adapter = MattermostAdapter(config)
        adapter._session = session
        monkeypatch.setattr("tools.send_message_tool._live_adapter", lambda platform: (None, adapter))
        opened = []
        def native_file(path):
            opened.append(path)
            assert path == native_path
            return SimpleNamespace(exists=lambda: True, read_bytes=lambda: b"image", name="space %23 #.png")
        monkeypatch.setattr("plugins.platforms.mattermost.adapter.Path", native_file)
        result = await _send_to_platform(Platform.MATTERMOST, config, "channel", "  caption\n",
                                         media_files=[(native_path, False)])
        assert result["success"], result
        assert opened == [native_path]
        assert result["delivered_media"] == 1
        assert len(uploads) == len(posts) == 1
        assert posts[0]["message"] == "  caption\n"

    @pytest.mark.parametrize("uri,native_path", [
        ("file:///C:/images/space%20%2523%20%23.png", r"C:\images\space %23 #.png"),
        ("file://server/share/space%20%2523%20%23.png", r"\\server\share\space %23 #.png"),
    ])
    @pytest.mark.asyncio
    async def test_gateway_windows_file_uri_uses_native_decoder(self, transport, monkeypatch, uri, native_path):
        import nturl2path
        from plugins.platforms.mattermost.adapter import MattermostAdapter
        config, session, posts, uploads = transport
        adapter = MattermostAdapter(config)
        adapter._session = session
        # Exercise Windows' standard-library decoder without claiming a Windows filesystem run.
        monkeypatch.setattr("plugins.platforms.mattermost.adapter.url2pathname", nturl2path.url2pathname, raising=False)
        opened = []
        def native_file(path):
            opened.append(path)
            assert path == native_path
            return SimpleNamespace(exists=lambda: True, read_bytes=lambda: b"image", name="space %23 #.png")
        monkeypatch.setattr("plugins.platforms.mattermost.adapter.Path", native_file)
        result = await adapter.send_multiple_images("channel", [(uri, "")])
        assert result.success
        assert opened == [native_path]
        assert len(uploads) == len(posts) == 1

    @pytest.mark.parametrize("live", [False, True])
    @pytest.mark.parametrize("caption", ["", "  explicit @here\n", " \t "])
    @pytest.mark.asyncio
    async def test_generated_filename_batch_fits_minimum_limit(self, tmp_path, transport, monkeypatch, live, caption):
        from plugins.platforms.mattermost.adapter import MattermostAdapter
        config, session, posts, uploads = transport
        config.extra["max_post_length"] = 500
        if live:
            adapter = MattermostAdapter(config)
            adapter._session = session
            adapter._api_get = AsyncMock(return_value={"root_id": "root"})
            monkeypatch.setattr("tools.send_message_tool._live_adapter", lambda platform: (None, adapter))
        paths = [tmp_path / (f"{i}-@here-" + "x" * 160 + ".png") for i in range(5)]
        for path in paths:
            path.write_bytes(b"image")
        result = await _send_to_platform(Platform.MATTERMOST, config, "channel", caption, thread_id="root",
                                         media_files=[(str(path), False) for path in paths])
        assert result["success"]
        assert result["delivered_media"] == 5
        assert len(posts) == 1
        assert len(posts[0]["message"]) <= 500
        assert posts[0]["root_id"] == "root"
        assert len(posts[0]["file_ids"]) == 5
        if caption:
            assert posts[0]["message"] == caption
        else:
            assert len(posts[0]["message"].splitlines()) == 5
            assert "@here" not in posts[0]["message"]

    @pytest.mark.parametrize("live", [False, True])
    @pytest.mark.parametrize("scenario", ["early_failure", "early_exception", "all_rejected"])
    def test_public_media_request_keeps_totals_and_text_receipts(self, tmp_path, transport, monkeypatch, live, scenario):
        from plugins.platforms.mattermost.adapter import MattermostAdapter
        config, session, posts, uploads = transport
        if live:
            adapter = MattermostAdapter(config)
            adapter._session = session
            monkeypatch.setattr("tools.send_message_tool._live_adapter", lambda platform: (None, adapter))
        path = tmp_path / "private.png"
        path.write_bytes(b"image")
        # A real strict filter rejects an existing file outside its roots. Never forward it.
        monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "1" if scenario == "all_rejected" else "0")
        monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", "0")
        monkeypatch.setattr("gateway.config.load_gateway_config",
                            lambda: SimpleNamespace(platforms={Platform.MATTERMOST: config}))
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
        monkeypatch.setattr("tools.send_message_tool._mirror_sent_message", lambda *args: False)
        monkeypatch.setattr("model_tools._run_async", lambda coro: asyncio.run(coro))
        original_post = session.post.side_effect
        def post(url, **kwargs):
            response = original_post(url, **kwargs)
            if url.endswith("/posts") and len(posts) == 2:
                if scenario == "early_failure":
                    response.status = 503
                elif scenario == "early_exception":
                    response.json.side_effect = TimeoutError()
            return response
        session.post.side_effect = post
        result = json.loads(send_message_tool({"target": "mattermost:aaaaaaaaaaaaaaaaaaaaaaaaaa",
                                               "message": "x" * 10000 + f"\nMEDIA:{path}"}))
        assert result["success"] is False
        assert result["total_media"] == 1
        assert result["delivered_media"] == 0
        assert result["media_delivered"] is False
        assert result["partial_failure"] is True
        expected_ids = [f"post-{i+1}" for i in range(len(posts))] if scenario == "all_rejected" else ["post-1"]
        assert len(posts) >= 2
        assert result["message_ids"] == expected_ids
        assert result["message_id"] == expected_ids[-1]
        assert len(uploads) == (0 if scenario == "all_rejected" else 1)
        assert all(not p.get("file_ids") for p in posts)

    @pytest.mark.parametrize("async_handler", [False, True])
    @pytest.mark.parametrize("text,file_exists", [("caption", True), ("caption", False), ("", False)])
    @pytest.mark.parametrize("handler_success", [False, True])
    def test_mattermost_custom_handler_keeps_typed_media_request(
            self, tmp_path, transport, monkeypatch, async_handler, text, file_exists, handler_success):
        from dataclasses import replace
        from gateway.platform_registry import platform_registry
        config, session, posts, uploads = transport
        path = tmp_path / "image.png"
        if file_exists:
            path.write_bytes(b"image")
        args = {"target": "mattermost:aaaaaaaaaaaaaaaaaaaaaaaaaa", "message": f"{text}\nMEDIA:{path}",
                "custom_options": {"enabled": True, "count": 7}}
        expected_result = ({"success": True, "message_id": "custom"} if handler_success
                           else {"success": False, "error": "Custom delivery declined"})
        seen = []
        def handler(request, chat_id, platform_name, pconfig):
            seen.append((request, chat_id, platform_name, pconfig))
            return dict(expected_result)
        async def async_send(*args):
            return handler(*args)
        original_entry = platform_registry.get("mattermost")
        platform_registry.register(replace(original_entry, send_message_handler=async_send if async_handler else handler))
        monkeypatch.setattr("gateway.config.load_gateway_config",
                            lambda: SimpleNamespace(platforms={Platform.MATTERMOST: config}))
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
        monkeypatch.setattr("tools.send_message_tool._mirror_sent_message", lambda *args: False)
        monkeypatch.setattr("model_tools._run_async", lambda coro: asyncio.run(coro))
        try:
            result = json.loads(send_message_tool(args))
        finally:
            platform_registry.register(original_entry)
        assert seen == [(args, "aaaaaaaaaaaaaaaaaaaaaaaaaa", "mattermost", config)]
        if not file_exists:
            # Native accounting still reports missing media after custom dispatch.
            expected_result["media_dropped"] = [{"path": str(path), "reason": "not found on this host"}]
            if handler_success:
                expected_result.update(
                    success=False, partial_success=True,
                    error="Delivery incomplete: 1 requested MEDIA attachment(s) "
                          "dropped before delivery (see media_dropped)")
        assert result == expected_result
        assert not posts and not uploads

    @pytest.mark.parametrize("scenario", ["six", "first_missing", "all_missing", "later_exception"])
    @pytest.mark.asyncio
    async def test_live_native_batch_complete_success_requires_all_files(self, tmp_path, transport, monkeypatch, scenario):
        from plugins.platforms.mattermost.adapter import MattermostAdapter
        config, session, posts, uploads = transport
        adapter = MattermostAdapter(config)
        adapter._session = session
        adapter._api_get = AsyncMock(return_value={"root_id": "root"})
        monkeypatch.setattr("tools.send_message_tool._live_adapter", lambda platform: (None, adapter))
        paths = [tmp_path / f"file-{i}.png" for i in range(6)]
        for i, path in enumerate(paths):
            if scenario != "all_missing" and not (scenario == "first_missing" and i < 5):
                path.write_bytes(b"image")
        original_post = session.post.side_effect
        def post(url, **kwargs):
            response = original_post(url, **kwargs)
            if scenario == "later_exception" and url.endswith("/posts") and len(posts) == 2:
                response.json.side_effect = RuntimeError("response lost")
            return response
        session.post.side_effect = post
        result = await _send_to_platform(Platform.MATTERMOST, config, "channel", "  caption\n", thread_id="root",
                                         media_files=[(str(path), False) for path in paths])
        assert result["success"] is (scenario == "six")
        delivered = {"six": 6, "first_missing": 1, "all_missing": 0, "later_exception": 5}[scenario]
        assert result["total_media"] == 6
        assert result["delivered_media"] == delivered
        assert result["media_delivered"] is bool(delivered)
        assert result["message_ids"] == (["post-1", "post-2"] if scenario == "six" else ["post-1"])
        assert all(len(post.get("file_ids", [])) <= 5 for post in posts)
        assert all(post["root_id"] == "root" for post in posts)
        assert sum(post["message"] == "  caption\n" for post in posts) == 1

    @pytest.mark.parametrize("text", ["", "useful text"])
    def test_public_rejected_media_is_not_total_success(self, tmp_path, transport, monkeypatch, text):
        config, session, posts, uploads = transport
        monkeypatch.setattr("gateway.config.load_gateway_config",
                            lambda: SimpleNamespace(platforms={Platform.MATTERMOST: config}))
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
        monkeypatch.setattr("tools.send_message_tool._mirror_sent_message", lambda *args: False)
        monkeypatch.setattr("model_tools._run_async", lambda coro: asyncio.run(coro))
        result = json.loads(send_message_tool({"target": "mattermost:aaaaaaaaaaaaaaaaaaaaaaaaaa",
                                               "message": f"{text}\nMEDIA:{tmp_path / 'missing.png'}"}))
        assert result.get("success") is False, result
        assert result["media_delivered"] is False
        assert result["total_media"] == 1
        assert result["delivered_media"] == 0
        assert result["partial_failure"] is bool(text)
        assert len(posts) == (1 if text else 0)
        assert result["media_dropped"] == [
            {"path": str(tmp_path / "missing.png"), "reason": "not found on this host"},
        ]

    @pytest.mark.parametrize("live", [False, True])
    @pytest.mark.parametrize("media_count", [0, 1, 6])
    @pytest.mark.parametrize("outcome", ["ok", "rejected", "timeout", "empty"])
    @pytest.mark.asyncio
    async def test_public_media_send_preserves_all_ids_on_long_text_partial_failure(
            self, tmp_path, transport, monkeypatch, live, media_count, outcome):
        from dataclasses import replace
        from gateway.platform_registry import platform_registry
        from plugins.platforms.mattermost.adapter import MattermostAdapter, _standalone_send
        config, session, posts, uploads = transport
        config.extra["max_post_length"] = 500
        adapter = MattermostAdapter(config)
        adapter._session = session
        adapter._api_get = AsyncMock(return_value={"root_id": "root"})
        runner = SimpleNamespace(_gateway_loop=asyncio.get_running_loop())
        monkeypatch.setattr("tools.send_message_tool._live_adapter", lambda p: (runner, adapter if live else None))
        native_sender = AsyncMock(wraps=_standalone_send)
        entry = platform_registry.get("mattermost")
        platform_registry.register(replace(entry, standalone_sender_fn=native_sender))
        paths = [tmp_path / f"image-{i}.png" for i in range(media_count)]
        for path in paths:
            path.write_bytes(b"image")
        original_post = session.post.side_effect
        def post(url, **kwargs):
            response = original_post(url, **kwargs)
            if url.endswith("/posts") and len(posts) == 3:
                if outcome == "rejected":
                    response.status = 413
                    response.text.return_value = "rejected"
                elif outcome == "timeout":
                    response.json.side_effect = TimeoutError()
                elif outcome == "empty":
                    response.json.return_value = {}
            return response
        session.post.side_effect = post
        message = "x" * 1300
        try:
            result = await _send_to_platform(Platform.MATTERMOST, config, "channel", message,
                                            thread_id="root", media_files=[str(p) for p in paths])
        finally:
            platform_registry.register(entry)
        native_sender.assert_awaited_once()
        assert native_sender.await_args.args[2] == message
        assert result["success"] is (outcome == "ok")
        assert result["delivered_media"] == (media_count if outcome == "ok" else 0)
        assert result["total_media"] == media_count
        expected_posts = 4 if outcome == "ok" and media_count == 6 else 3
        assert len(posts) == expected_posts
        expected_ids = [f"post-{i + 1}" for i in range(expected_posts if outcome == "ok" else 2)]
        assert result["message_ids"] == expected_ids
        assert result["message_id"] == expected_ids[-1]
        assert all(len(p["message"]) <= 500 for p in posts)
        assert all(p["root_id"] == "root" and p["props"]["disable_mentions"] for p in posts)
        if media_count:
            assert "".join(p["message"] for p in posts[:3]) == message
            assert not posts[0].get("file_ids") and not posts[1].get("file_ids")
            attached = [fid for p in posts for fid in p.get("file_ids", [])]
            assert attached == [f"file-{i + 1}" for i in range(media_count if outcome == "ok" else min(5, media_count))]

    @pytest.mark.parametrize("media", [False, True])
    @pytest.mark.asyncio
    async def test_registered_native_dispatch_cancellation_is_not_replayed(self, tmp_path, transport, monkeypatch, media):
        from plugins.platforms.mattermost.adapter import MattermostAdapter
        config, session, posts, uploads = transport
        adapter = MattermostAdapter(config)
        adapter._session = session
        runner = SimpleNamespace(_gateway_loop=asyncio.get_running_loop())
        monkeypatch.setattr("tools.send_message_tool._live_adapter", lambda p: (runner, adapter))
        original_post = session.post.side_effect
        def post(url, **kwargs):
            response = original_post(url, **kwargs)
            if url.endswith("/posts"):
                response.json.side_effect = asyncio.CancelledError()
            return response
        session.post.side_effect = post
        path = tmp_path / "image.png"
        path.write_bytes(b"image")
        with pytest.raises(asyncio.CancelledError):
            await _send_to_platform(Platform.MATTERMOST, config, "channel", "caption",
                                    media_files=[str(path)] if media else [])
        assert len(posts) == 1

    @pytest.mark.parametrize("media", [False, True])
    @pytest.mark.parametrize("cancel", [False, True])
    @pytest.mark.asyncio
    async def test_registered_sender_uses_gateway_loop_and_shields_enqueued_send(
            self, tmp_path, transport, monkeypatch, media, cancel):
        import threading
        from plugins.platforms.mattermost.adapter import MattermostAdapter
        config, session, posts, uploads = transport
        adapter = MattermostAdapter(config)
        adapter._session = session
        gateway_loop = asyncio.new_event_loop()
        ready, started, finished = threading.Event(), threading.Event(), threading.Event()
        release = asyncio.Event()
        native_results = []
        def run_loop():
            asyncio.set_event_loop(gateway_loop)
            gateway_loop.call_soon(ready.set)
            gateway_loop.run_forever()
        thread = threading.Thread(target=run_loop)
        thread.start()
        runner = SimpleNamespace(_gateway_loop=gateway_loop)
        monkeypatch.setattr("tools.send_message_tool._live_adapter", lambda p: (runner, adapter))
        native = adapter.send_multiple_images if media else adapter.send
        async def send(*args, **kwargs):
            assert asyncio.get_running_loop() is gateway_loop
            started.set()
            await release.wait()
            result = await native(*args, **kwargs)
            native_results.append(result)
            finished.set()
            return result
        monkeypatch.setattr(adapter, "send_multiple_images" if media else "send", send)
        path = tmp_path / "image.png"
        path.write_bytes(b"image")
        try:
            assert await asyncio.to_thread(ready.wait, 5)
            task = asyncio.create_task(_send_to_platform(
                Platform.MATTERMOST, config, "channel", "caption", media_files=[str(path)] if media else []))
            assert await asyncio.to_thread(started.wait, 5)
            if cancel:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            gateway_loop.call_soon_threadsafe(release.set)
            if not cancel:
                result = await task
                assert result["success"] and result["message_ids"] == ["post-1"]
            assert await asyncio.to_thread(finished.wait, 5)
            assert len(native_results) == 1 and native_results[0].success
            assert len(posts) == 1
        finally:
            gateway_loop.call_soon_threadsafe(release.set)
            gateway_loop.call_soon_threadsafe(gateway_loop.stop)
            thread.join(timeout=5)
            assert not thread.is_alive()
            gateway_loop.close()

    @pytest.fixture
    def transport(self, monkeypatch):
        from gateway.config import PlatformConfig
        from tools.send_message_tool import prepare_send_message_platforms
        config = PlatformConfig(enabled=True, token="test-mattermost-token", extra={"url": "https://mm.example.com"})
        prepare_send_message_platforms()
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        posts = []
        uploads = []

        def post(url, **kwargs):
            response = AsyncMock()
            response.status = 201
            if url.endswith("/files"):
                uploads.append(kwargs["data"])
                data = {"file_infos": [{"id": f"file-{len(uploads)}"}]}
            else:
                posts.append(kwargs["json"])
                data = {"id": f"post-{len(posts)}"}
            response.json.return_value = data
            response.__aenter__.return_value = response
            return response

        session.post.side_effect = post
        monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: session)
        monkeypatch.setattr("tools.send_message_tool._live_adapter", lambda platform: (None, None))
        monkeypatch.setattr("gateway.channel_directory.resolve_channel_name", lambda platform, ref: ref)
        return config, session, posts, uploads

    @pytest.mark.parametrize("caption", ["", "  caption @here\n", " \t "])
    @pytest.mark.parametrize("descriptor", ["tuple", "dict", "string"])
    @pytest.mark.asyncio
    async def test_six_files_caption_once_thread_each_batch(self, tmp_path, transport, caption, descriptor):
        from plugins.platforms.mattermost.adapter import _standalone_send
        config, session, posts, uploads = transport
        paths = [tmp_path / f"file-{i}.png" for i in range(6)]
        for path in paths:
            path.write_bytes(b"image")
        media = [{"tuple": (str(path), False), "dict": {"path": str(path)}, "string": str(path)}[descriptor]
                 for path in paths]
        result = await _standalone_send(config, "channel", caption, thread_id="root", media_files=media)
        assert result["success"] is True
        assert result["media_delivered"] is True
        assert result["delivered_media"] == result["total_media"] == 6
        assert result["message_ids"] == ["post-1", "post-2"]
        assert result["message_id"] == "post-2"
        assert [len(post["file_ids"]) for post in posts] == [5, 1]
        assert all(post["root_id"] == "root" for post in posts)
        assert all(post["props"]["disable_mentions"] for post in posts)
        if caption:
            assert posts[0]["message"] == caption
            assert posts[1]["message"] != caption
        else:
            assert "file-0.png" in posts[0]["message"]
        assert "file-5.png" in posts[1]["message"]

    @pytest.mark.parametrize("text", ["", "  useful text\n"])
    @pytest.mark.parametrize("missing", ["first", "all", "disappeared"])
    @pytest.mark.asyncio
    async def test_missing_files_are_partial_or_failed(self, tmp_path, transport, text, missing, monkeypatch):
        from plugins.platforms.mattermost.adapter import _standalone_send
        config, session, posts, uploads = transport
        paths = [tmp_path / f"file-{i}.png" for i in range(6)]
        for i, path in enumerate(paths):
            if missing == "disappeared" or (missing == "first" and i == 5):
                path.write_bytes(b"image")
        if missing == "disappeared":
            # Disappearance after dispatch validation but before open().
            original_open = open
            def vanished(file, *args, **kwargs):
                if str(file) == str(paths[0]):
                    raise FileNotFoundError(str(file))
                return original_open(file, *args, **kwargs)
            monkeypatch.setattr("builtins.open", vanished)
        result = await _standalone_send(config, "channel", text, thread_id="root",
                                        media_files=[str(path) for path in paths])
        delivered = {"first": 1, "all": 0, "disappeared": 5}[missing]
        assert result["success"] is False
        assert result["error"]
        assert result["delivered_media"] == delivered
        assert result["total_media"] == 6
        assert result["media_delivered"] is bool(delivered)
        assert result["partial_failure"] is bool(delivered or text)
        assert result["message_ids"] == [f"post-{i + 1}" for i in range(len(posts))]
        assert all(post["root_id"] == "root" for post in posts)
        if text:
            assert sum(post["message"] == text for post in posts) == 1
        assert all(post.get("file_ids") or post["message"] for post in posts)
        if missing == "all" and not text:
            assert not posts

    @pytest.mark.parametrize("upload_outcome, fail_at, caption_outcome", [
        (413, 1, "ok"), (503, 2, "ok"), ("missing_id", 1, "ok"),
        ("timeout", 1, "ok"), ("timeout", 7, "ok"),
        (413, 1, "timeout"), (503, 1, "empty"), ("cancelled", 1, "ok"),
    ])
    @pytest.mark.asyncio
    async def test_upload_failure_preserves_only_pending_caption(
        self, tmp_path, transport, upload_outcome, fail_at, caption_outcome,
    ):
        from plugins.platforms.mattermost.adapter import _standalone_send
        config, session, posts, uploads = transport
        config.extra["max_post_length"] = 500
        paths = [tmp_path / f"file-{i}.png" for i in range(8)]
        for path in paths:
            path.write_bytes(b"image")
        original_post = session.post.side_effect

        def post(url, **kwargs):
            response = original_post(url, **kwargs)
            if url.endswith("/files") and len(uploads) == fail_at:
                if isinstance(upload_outcome, int):
                    response.status = upload_outcome
                elif upload_outcome == "missing_id":
                    response.json.return_value = {"file_infos": [{}]}
                else:
                    response.__aenter__.side_effect = (
                        asyncio.CancelledError() if upload_outcome == "cancelled" else TimeoutError("upload lost"))
            elif url.endswith("/posts") and len(posts) == 2:
                if caption_outcome == "timeout":
                    response.json.side_effect = TimeoutError("caption ack lost")
                elif caption_outcome == "empty":
                    response.json.return_value = {}
            return response

        session.post.side_effect = post
        caption = "  explicit @here\n![keep](https://example.com/image)\t" + "x" * 550 + "  "
        delivery = _standalone_send(config, "channel", caption, thread_id="root",
                                    media_files=[str(path) for path in paths])
        if upload_outcome == "cancelled":
            with pytest.raises(asyncio.CancelledError):
                await delivery
            assert len(uploads) == 1 and not posts
            return
        result = await delivery
        assert "".join(p["message"] for p in posts) == caption
        assert len(posts) == 2
        assert len(uploads) == fail_at
        assert all(len(p["message"]) <= 500 and p["root_id"] == "root"
                   and p["props"]["disable_mentions"] for p in posts)
        delivered = 5 if fail_at > 5 else 0
        assert [p.get("file_ids") for p in posts] == [None, [f"file-{i}" for i in range(1, 6)] if delivered else None]
        ids = ["post-1", "post-2"] if caption_outcome == "ok" else ["post-1"]
        assert result["message_ids"] == ids
        assert result["message_id"] == ids[-1]
        assert not result["success"]
        assert result["total_media"] == 8
        assert result["delivered_media"] == delivered
        assert result["failed_media"] == 8 - delivered
        assert result["media_delivered"] is bool(delivered)
        assert result["partial_failure"]
        assert "upload" in result["error"].lower()
        if isinstance(upload_outcome, int):
            assert str(upload_outcome) in result["error"]
        if caption_outcome != "ok":
            assert result["_delivery_uncertain"]

    @pytest.mark.parametrize("failure", ["exception", "missing_id"])
    @pytest.mark.asyncio
    async def test_exception_after_ack_keeps_receipt_without_retry(self, tmp_path, transport, failure):
        from plugins.platforms.mattermost.adapter import _standalone_send
        config, session, posts, uploads = transport
        paths = [tmp_path / f"file-{i}.png" for i in range(6)]
        for path in paths:
            path.write_bytes(b"image")
        original_post = session.post.side_effect
        def lose_second_post(url, **kwargs):
            response = original_post(url, **kwargs)
            if url.endswith("/posts") and len(posts) == 2:
                if failure == "exception":
                    response.json.side_effect = RuntimeError("response lost")
                else:
                    response.json.return_value = {}
            return response
        session.post.side_effect = lose_second_post
        result = await _standalone_send(config, "channel", "caption", thread_id="root",
                                        media_files=[str(path) for path in paths])
        assert result["success"] is False
        assert result["partial_failure"] is True
        assert result["message_ids"] == ["post-1"]
        assert result["message_id"] == "post-1"
        assert result["delivered_media"] == 5
        assert result["total_media"] == 6
        assert result["media_delivered"] is True
        assert len(posts) == 2
        assert len(uploads) == 6

    @pytest.mark.parametrize("disappear", [False, True])
    def test_public_file_only_native_plugin_route(self, tmp_path, transport, monkeypatch, disappear):
        config, session, posts, uploads = transport
        path = tmp_path / "picture.png"
        path.write_bytes(b"image")
        gateway_config = SimpleNamespace(platforms={Platform.MATTERMOST: config})
        monkeypatch.setattr("gateway.config.load_gateway_config", lambda: gateway_config)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)
        monkeypatch.setattr("tools.send_message_tool._mirror_sent_message", lambda *args: False)
        def dispatch(coro):
            if disappear:
                path.unlink()
            return asyncio.run(coro)
        monkeypatch.setattr("model_tools._run_async", dispatch)
        result = json.loads(send_message_tool({"target": "mattermost:aaaaaaaaaaaaaaaaaaaaaaaaaa",
                                               "message": f"MEDIA:{path}"}))
        assert result["success"] is (not disappear)
        assert result["media_delivered"] is (not disappear)
        assert result["total_media"] == 1
        assert result["delivered_media"] == (0 if disappear else 1)
        assert len(posts) == (0 if disappear else 1)
        if posts:
            assert posts[0]["message"] == "📎 picture.png"
            assert posts[0]["file_ids"] == ["file-1"]
