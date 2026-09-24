"""Authenticated desktop PCM stream; connection ownership bounds recording lifetime.

Generic WebSocket handler for the live ``/api/audio/transcribe-stream`` endpoint. Speaks the
``ready`` / ``partial`` / ``final`` / ``error`` / ``unsupported`` wire protocol. The active
backend is resolved through the streaming transcription registry by ``stt.provider``;
provider-specific configuration is passed through under the provider's own name — Core never
knows a concrete provider.
"""
import asyncio


async def serve_transcription(ws, config):
    from fastapi import WebSocketDisconnect
    from agent.streaming_transcription_registry import get_provider as _get_streaming_provider
    from hermes_cli.plugins import _ensure_plugins_discovered
    from tools.transcription_tools import is_stt_enabled

    provider = None
    key = None
    total_bytes = 0
    try:
        if not is_stt_enabled(config):
            await ws.send_json({"type": "unsupported"})
            return
        _ensure_plugins_discovered()
        provider = _get_streaming_provider(config.get("provider"))
        if provider is None:
            await ws.send_json({"type": "unsupported"})
            return
        provider_config = config.get(provider.name) or {}
        key = await asyncio.to_thread(provider.start, config=provider_config)
        await ws.send_json({"type": "ready", "sample_rate": 16000})
        last_text = ""
        while True:
            message = await asyncio.wait_for(ws.receive(), timeout=60)
            if message["type"] == "websocket.disconnect":
                break
            pcm = message.get("bytes")
            if pcm is not None:
                total_bytes += len(pcm)
                if total_bytes > 16000 * 2 * 600:
                    raise ValueError("Recording exceeds ten minutes")
                text = await asyncio.to_thread(provider.feed, key, pcm)
                if text != last_text:
                    last_text = text
                    await ws.send_json({"type": "partial", "text": text})
            elif message.get("text") == "finish":
                text = await asyncio.to_thread(provider.finish, key)
                key = None
                await ws.send_json({"type": "final", "text": text})
                break
            elif message.get("text") == "cancel":
                break
            else:
                raise ValueError("Unknown ASR stream message")
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await ws.send_json({"type": "error", "error": str(exc)})
        except (RuntimeError, WebSocketDisconnect):
            pass
    finally:
        if key and provider is not None:
            await asyncio.to_thread(provider.cancel, key)