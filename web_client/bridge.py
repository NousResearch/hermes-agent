#!/usr/bin/env python3
"""WebGPU Bridge Server — connects browser-side WebLLM inference to Hermes Agent.

The browser loads a model via WebGPU and connects over WebSocket.
This bridge exposes an OpenAI-compatible HTTP endpoint that Hermes talks to.

Usage:
    python3 -m web_client.bridge              # Start bridge on default ports
    python3 -m web_client.bridge --port 8801  # Custom WebSocket port

Then open web_client/index.html in Chrome, load a model, and click Connect.
Hermes connects via:
    hermes --provider webgpu
"""

import argparse
import asyncio
import json
import logging
import time
import uuid
import webbrowser
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, HTMLResponse, StreamingResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("webgpu-bridge")

# State
_ws_client: WebSocket | None = None
_pending: dict[str, asyncio.Future] = {}
_client_model: str = "unknown"
_stats = {"requests": 0, "start_time": time.time()}


async def ws_endpoint(websocket: WebSocket):
    """WebSocket endpoint for the browser client."""
    global _ws_client, _client_model
    await websocket.accept()
    _ws_client = websocket
    logger.info("Browser client connected")

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "ready":
                _client_model = data.get("model", "unknown")
                logger.info(f"Client ready with model: {_client_model}")

            elif msg_type == "pong":
                pass

            elif msg_type == "response":
                req_id = data.get("id")
                if req_id in _pending:
                    _pending[req_id].set_result(data.get("data"))

            elif msg_type == "chunk":
                req_id = data.get("id")
                if req_id in _pending and hasattr(_pending[req_id], "_chunks"):
                    _pending[req_id]._chunks.put_nowait(data.get("data"))

            elif msg_type == "done":
                req_id = data.get("id")
                if req_id in _pending and hasattr(_pending[req_id], "_chunks"):
                    _pending[req_id]._chunks.put_nowait(None)  # sentinel

            elif msg_type == "error":
                req_id = data.get("id")
                if req_id in _pending:
                    _pending[req_id].set_exception(
                        RuntimeError(data.get("error", "inference failed"))
                    )

    except WebSocketDisconnect:
        logger.info("Browser client disconnected")
        _ws_client = None
        # Fail all pending requests
        for fut in _pending.values():
            if not fut.done():
                fut.set_exception(RuntimeError("browser client disconnected"))
        _pending.clear()


async def chat_completions(request: Request):
    """OpenAI-compatible /v1/chat/completions endpoint."""
    if _ws_client is None:
        return JSONResponse(
            {"error": {"message": "No browser client connected. Open web_client/index.html and load a model."}},
            status_code=503,
        )

    body = await request.json()
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    stream = body.get("stream", False)
    _stats["requests"] += 1

    if stream:
        return await _handle_streaming(req_id, body)
    else:
        return await _handle_non_streaming(req_id, body)


async def _handle_non_streaming(req_id: str, body: dict):
    """Non-streaming: send to browser, wait for response."""
    fut = asyncio.get_event_loop().create_future()
    _pending[req_id] = fut

    await _ws_client.send_json({
        "type": "inference",
        "data": {"id": req_id, "stream": False, **body},
    })

    try:
        result = await asyncio.wait_for(fut, timeout=300)
        return JSONResponse(result)
    except asyncio.TimeoutError:
        return JSONResponse({"error": {"message": "inference timed out"}}, status_code=504)
    except RuntimeError as e:
        return JSONResponse({"error": {"message": str(e)}}, status_code=502)
    finally:
        _pending.pop(req_id, None)


async def _handle_streaming(req_id: str, body: dict):
    """Streaming: SSE response, chunks forwarded from browser."""
    chunk_queue = asyncio.Queue()
    fut = asyncio.get_event_loop().create_future()
    fut._chunks = chunk_queue
    _pending[req_id] = fut

    await _ws_client.send_json({
        "type": "inference",
        "data": {"id": req_id, "stream": True, **body},
    })

    async def event_generator():
        try:
            while True:
                chunk = await asyncio.wait_for(chunk_queue.get(), timeout=120)
                if chunk is None:  # done sentinel
                    yield "data: [DONE]\n\n"
                    break
                yield f"data: {json.dumps(chunk)}\n\n"
        except asyncio.TimeoutError:
            yield f'data: {json.dumps({"error": "stream timeout"})}\n\n'
        finally:
            _pending.pop(req_id, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def models_list(request: Request):
    """GET /v1/models — report the loaded model."""
    return JSONResponse({
        "object": "list",
        "data": [
            {
                "id": _client_model,
                "object": "model",
                "owned_by": "webgpu-local",
            }
        ],
    })


async def health(request: Request):
    """Health check."""
    return JSONResponse({
        "status": "ok" if _ws_client else "waiting_for_client",
        "model": _client_model,
        "requests_served": _stats["requests"],
        "uptime_seconds": int(time.time() - _stats["start_time"]),
    })


async def serve_ui(request: Request):
    """Serve the web client HTML."""
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text())


app = Starlette(
    routes=[
        # Browser WebSocket
        WebSocketRoute("/ws", ws_endpoint),
        # OpenAI-compatible API (hermes talks to this)
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        Route("/v1/models", models_list, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
        # Serve the web UI
        Route("/", serve_ui, methods=["GET"]),
    ],
)


def main():
    p = argparse.ArgumentParser(description="WebGPU bridge — browser inference for Hermes Agent")
    p.add_argument("--port", type=int, default=8801, help="Port for both HTTP API and WebSocket (default: 8801)")
    p.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    args = p.parse_args()

    print(f"WebGPU Bridge Server")
    print(f"  HTTP API:    http://127.0.0.1:{args.port}/v1")
    print(f"  WebSocket:   ws://127.0.0.1:{args.port}/ws")
    print(f"  Web UI:      http://127.0.0.1:{args.port}/")
    print()
    print(f"  Connect hermes:")
    print(f"    hermes --provider webgpu")
    print()

    if not args.no_browser:
        import threading
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}/")).start()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
