"""Deterministic OpenAI-compatible stub used by the agent end-to-end test.

The end-to-end test needs the *real* Hermes agent loop (turn handling, tool
dispatch, result plumbing) without depending on an external model provider.
This stub implements just enough of ``POST /v1/chat/completions``:

  * first request  -> an assistant message with one ``tool_calls`` entry for the
    configured tool name/arguments
  * second request -> a final assistant message that echoes the tool result

That is the smallest exchange that proves a tool call made the full round trip.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional


class OpenAIStubServer:
    def __init__(
        self,
        *,
        tool_name: str,
        tool_arguments: dict[str, Any],
        model: str = "stub-model",
    ) -> None:
        self.tool_name = tool_name
        self.tool_arguments = tool_arguments
        self.model = model
        self.requests: list[dict[str, Any]] = []
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.base_url = ""

    # ── response construction ────────────────────────────────────────────
    def _completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages") or []
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        if not tool_messages:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_utcp_e2e",
                        "type": "function",
                        "function": {
                            "name": self.tool_name,
                            "arguments": json.dumps(self.tool_arguments),
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
        else:
            tool_content = tool_messages[-1].get("content")
            message = {
                "role": "assistant",
                "content": f"TOOL_RESULT {tool_content}",
            }
            finish_reason = "stop"
        return {
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model") or self.model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self) -> OpenAIStubServer:
        stub = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format, *args):
                pass

            def _send_json(self, payload: dict[str, Any]) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_stream(self, completion: dict[str, Any]) -> None:
                """Emit the completion as OpenAI chat-completion SSE chunks."""
                message = completion["choices"][0]["message"]
                finish_reason = completion["choices"][0]["finish_reason"]
                base = {
                    "id": completion["id"],
                    "object": "chat.completion.chunk",
                    "created": completion["created"],
                    "model": completion["model"],
                }

                def chunk(delta: dict[str, Any], finish: Optional[str] = None) -> bytes:
                    payload = dict(base)
                    payload["choices"] = [
                        {"index": 0, "delta": delta, "finish_reason": finish}
                    ]
                    return f"data: {json.dumps(payload)}\n\n".encode()

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                chunks: list[bytes] = [chunk({"role": "assistant", "content": ""})]
                if message.get("tool_calls"):
                    for index, tool_call in enumerate(message["tool_calls"]):
                        chunks.append(
                            chunk(
                                {
                                    "tool_calls": [
                                        {
                                            "index": index,
                                            "id": tool_call["id"],
                                            "type": "function",
                                            "function": {
                                                "name": tool_call["function"]["name"],
                                                "arguments": tool_call["function"]["arguments"],
                                            },
                                        }
                                    ]
                                }
                            )
                        )
                else:
                    content = message.get("content") or ""
                    # Two chunks so the client exercises mid-stream assembly.
                    midpoint = len(content) // 2
                    chunks.append(chunk({"content": content[:midpoint]}))
                    chunks.append(chunk({"content": content[midpoint:]}))
                chunks.append(chunk({}, finish_reason))
                chunks.append(b"data: [DONE]\n\n")

                for item in chunks:
                    if item != b"data: [DONE]\n\n":
                        item = b"%X\r\n" % len(item) + item + b"\r\n"
                    else:
                        item = b"%X\r\n" % len(item) + item + b"\r\n"
                    self.wfile.write(item)
                self.wfile.write(b"0\r\n\r\n")

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    payload = json.loads(raw.decode("utf-8") or "{}")
                except ValueError:
                    payload = {}
                stub.requests.append(payload)
                completion = stub._completion(payload)
                if payload.get("stream"):
                    self._send_stream(completion)
                else:
                    self._send_json(completion)

            def do_GET(self) -> None:
                # Model discovery endpoints some providers probe.
                self._send_json({"data": [{"id": stub.model, "object": "model"}]})

        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        server.daemon_threads = True
        self._server = server
        self.base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
        thread = threading.Thread(target=server.serve_forever, name="openai-stub", daemon=True)
        thread.start()
        self._thread = thread
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> OpenAIStubServer:
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()
