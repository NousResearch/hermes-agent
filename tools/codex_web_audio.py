"""ChatGPT web read-aloud transport for Codex OAuth TTS.

The documented OpenAI speech API rejects ChatGPT/Codex OAuth tokens because
those tokens do not carry ``api.model.audio.request``. ChatGPT's subscription
read-aloud path instead synthesizes an existing assistant message. This module
creates a temporary, history-disabled conversation, verifies that the assistant
repeated the requested text exactly, then reads that message through
``/backend-api/synthesize``.

The Sentinel fingerprint/PoW shape and browser request profile are adapted from
starbaser/ccproxy (MIT, commit f2c47695b0835da023257aee0ac2a3dffd9fe570),
itself cross-checked against gproxy and aurora. Hermes identifies itself with an
``originator: hermes`` header and never presents an Android package, mobile app
identity, or forged Play Integrity payload.
"""

from __future__ import annotations

import base64
import json
import math
import random
import time
import uuid
from dataclasses import dataclass
from threading import Event
from typing import Any, Dict, Iterable, Optional

import requests

_BASE_URL = "https://chatgpt.com"
_SENTINEL_PREPARE = "/backend-api/sentinel/chat-requirements/prepare"
_SENTINEL_FINALIZE = "/backend-api/sentinel/chat-requirements/finalize"
_CONVERSATION_PREPARE = "/backend-api/f/conversation/prepare"
_CONVERSATION = "/backend-api/f/conversation"
_SYNTHESIZE = "/backend-api/synthesize"
_MAX_JSON_BYTES = 2 * 1024 * 1024
_MAX_SSE_BYTES = 4 * 1024 * 1024
_MAX_AUDIO_BYTES = 16 * 1024 * 1024
_MAX_POW_ATTEMPTS = 500_000

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
)
_BUILD_ID = "prod-d7545204e22cb990d0245281e6550977d93b6a81"
_SCRIPT_SRC = f"https://chatgpt.com/_next/static/chunks/{_BUILD_ID}.js"
_NATIVE_VOICES = frozenset({
    "juniper",
    "cove",
    "ember",
    "breeze",
    "maple",
    "vale",
    "glimmer",
    "orbit",
    "fathom",
    "ridge",
})
_VOICE_ALIASES = {
    "alloy": "juniper",
    "echo": "cove",
    "fable": "vale",
    "onyx": "ember",
    "nova": "breeze",
    "shimmer": "maple",
    "ash": "ridge",
    "ballad": "glimmer",
    "coral": "orbit",
    "sage": "fathom",
    "verse": "vale",
    "marin": "juniper",
    "cedar": "cove",
}


@dataclass(frozen=True)
class CodexSpeech:
    audio: bytes
    content_type: str
    conversation_id: str
    message_id: str
    spoken_text: str


@dataclass(frozen=True)
class _OperationBudget:
    deadline: float
    cancel_event: Optional[Event] = None

    def remaining(self) -> float:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise InterruptedError("Codex subscription speech was cancelled")
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Codex subscription speech exceeded its timeout")
        return remaining


def resolve_voice(raw: str) -> str:
    """Resolve ChatGPT-native and OpenAI-compatible voice names fail-closed."""
    voice = str(raw or "juniper").strip().lower()
    if voice in _NATIVE_VOICES:
        return voice
    if voice in _VOICE_ALIASES:
        return _VOICE_ALIASES[voice]
    raise ValueError(
        f"Unsupported Codex subscription voice {raw!r}. Choose one of: "
        + ", ".join(sorted(_NATIVE_VOICES | _VOICE_ALIASES.keys()))
    )


def _browser_date(now: float) -> str:
    # Match an English browser configured for UTC, independently of the
    # gateway host's locale/DST. The conversation uses the same timezone.
    local = time.gmtime(now)
    weekdays = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
    months = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
    return (
        f"{weekdays[local.tm_wday]} {months[local.tm_mon - 1]} {local.tm_mday:02d} "
        f"{local.tm_year:04d} {local.tm_hour:02d}:{local.tm_min:02d}:{local.tm_sec:02d} "
        "GMT+0000 (Coordinated Universal Time)"
    )


def _fingerprint_array(
    device_id: str, *, nonce: int = 1, elapsed_ms: int = 30_412
) -> list[Any]:
    now = time.time()
    return [
        "2774",
        _browser_date(now),
        "4294967296",
        nonce,
        random.random(),
        _USER_AGENT,
        _SCRIPT_SRC,
        _BUILD_ID,
        "en",
        elapsed_ms,
        "windowControlsOverlay−[object WindowControlsOverlay]",  # noqa: RUF001
        "location",
        "outerWidth",
        30412.5,
        device_id,
        "",
        32,
        now * 1000 - 30412.5,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]


def _encode_fingerprint(config: list[Any]) -> str:
    return base64.b64encode(json.dumps(config, separators=(",", ":")).encode()).decode()


def _requirements_token(config: list[Any]) -> str:
    return "gAAAAAC" + _encode_fingerprint(config)


def _pow_hash_hex(value: str) -> str:
    result = 2_166_136_261
    for byte in value.encode("utf-8"):
        result ^= byte
        result = (result * 16_777_619) & 0xFFFF_FFFF
    result ^= result >> 16
    result = (result * 2_246_822_507) & 0xFFFF_FFFF
    result ^= result >> 13
    result = (result * 3_266_489_909) & 0xFFFF_FFFF
    result ^= result >> 16
    return f"{result:08x}"


def _solve_pow(
    config: list[Any], seed: str, difficulty: str, budget: _OperationBudget
) -> str:
    started = time.monotonic()
    proof_config = list(config)
    for nonce in range(_MAX_POW_ATTEMPTS):
        budget.remaining()
        proof_config[3] = nonce
        proof_config[9] = int((time.monotonic() - started) * 1000)
        payload = _encode_fingerprint(proof_config)
        if _pow_hash_hex(seed + payload)[: len(difficulty)] <= difficulty:
            return f"gAAAAAB{payload}~S"
    raise RuntimeError(
        f"ChatGPT Sentinel proof-of-work exceeded {_MAX_POW_ATTEMPTS} attempts"
    )


def _headers(
    token: str, device_id: str, *, account_id: Optional[str] = None, final: bool = False
) -> Dict[str, str]:
    headers = {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9",
        "authorization": f"Bearer {token}",
        "oai-language": "en-US",
        "oai-device-id": device_id,
        "oai-session-id": device_id,
        "origin": _BASE_URL,
        "originator": "hermes",
        "priority": "u=1, i",
        "referer": f"{_BASE_URL}/",
        "sec-ch-ua": '"Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"',
        "sec-ch-ua-arch": '"x86"',
        "sec-ch-ua-bitness": '"64"',
        "sec-ch-ua-full-version": '"148.0.7778.98"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-model": '""',
        "sec-ch-ua-platform": '"Windows"',
        "sec-ch-ua-platform-version": '"15.0.0"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": _USER_AGENT,
        "oai-client-version": _BUILD_ID,
        "oai-client-build-number": "7646290",
    }
    if account_id:
        headers["ChatGPT-Account-ID"] = account_id
    if final:
        headers.update({"oai-echo-logs": "0,3352,1,4100", "oai-telemetry": "[1,null]"})
    return headers


def _bounded_bytes(
    response: requests.Response, limit: int, label: str, budget: _OperationBudget
) -> bytes:
    budget.remaining()
    length = response.headers.get("content-length")
    if length and length.isdigit() and int(length) > limit:
        raise ValueError(f"{label} response exceeds {limit} bytes")
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(64 * 1024):
        budget.remaining()
        if not chunk:
            continue
        total += len(chunk)
        if total > limit:
            raise ValueError(f"{label} response exceeds {limit} bytes")
        chunks.append(chunk)
    budget.remaining()
    return b"".join(chunks)


def _request(
    session: requests.Session,
    method: str,
    path: str,
    *,
    headers: Dict[str, str],
    budget: _OperationBudget,
    json_body: Optional[dict] = None,
    params: Optional[dict] = None,
    limit: int = _MAX_JSON_BYTES,
) -> tuple[requests.Response, bytes]:
    response = session.request(
        method,
        _BASE_URL + path,
        headers=headers,
        json=json_body,
        params=params,
        timeout=budget.remaining(),
        stream=True,
        allow_redirects=False,
    )
    try:
        budget.remaining()
        if 300 <= response.status_code < 400:
            raise RuntimeError(
                f"ChatGPT {path} refused redirect (HTTP {response.status_code})"
            )
        body = _bounded_bytes(response, limit, path, budget)
        if response.status_code >= 400:
            detail = body.decode("utf-8", "replace")[:300]
            raise RuntimeError(
                f"ChatGPT {path} failed (HTTP {response.status_code}): {detail}"
            )
        return response, body
    finally:
        response.close()


def _prepare_body(final_body: dict, state: str, partial_text: str) -> dict:
    stripped = {
        "messages",
        "enable_message_followups",
        "paragen_cot_summary_display_override",
        "force_parallel_switch",
        "client_contextual_info",
        "client_prepare_state",
    }
    body = {key: value for key, value in final_body.items() if key not in stripped}
    body.update({
        "fork_from_shared_post": False,
        "client_prepare_state": state,
        "client_contextual_info": {"app_name": "chatgpt.com"},
    })
    if state in {"sent", "success"}:
        body["partial_query"] = {
            "id": str(uuid.uuid4()),
            "author": {"role": "user"},
            "content": {"content_type": "text", "parts": [partial_text[:5]]},
        }
    return body


def _conversation_body(prompt: str) -> dict:
    return {
        "action": "next",
        "messages": [
            {
                "id": str(uuid.uuid4()),
                "author": {"role": "user"},
                "create_time": time.time(),
                "content": {"content_type": "text", "parts": [prompt]},
                "metadata": {
                    "developer_mode_connector_ids": [],
                    "selected_connector_ids": [],
                    "selected_sync_knowledge_store_ids": [],
                    "selected_sources": [],
                    "selected_github_repos": [],
                    "selected_all_github_repos": False,
                    "serialization_metadata": {"custom_symbol_offsets": []},
                },
            }
        ],
        "parent_message_id": "client-created-root",
        "model": "auto",
        "client_prepare_state": "sent",
        "timezone_offset_min": 0,
        "timezone": "UTC",
        "conversation_mode": {"kind": "primary_assistant"},
        "enable_message_followups": False,
        "system_hints": [],
        "supports_buffering": True,
        "supported_encodings": ["v1"],
        "client_contextual_info": {
            "is_dark_mode": True,
            "time_since_loaded": 5000,
            "page_height": 900,
            "page_width": 1400,
            "pixel_ratio": 1.0,
            "screen_height": 1080,
            "screen_width": 1920,
            "app_name": "chatgpt.com",
        },
        "paragen_cot_summary_display_override": "allow",
        "force_parallel_switch": "auto",
        "history_and_training_disabled": True,
    }


def _parse_conversation_sse(lines: Iterable[str]) -> tuple[str, str, str]:
    conversation_id = ""
    message_id = ""
    current_assistant = ""
    reply = ""
    for line in lines:
        if not line.startswith("data: "):
            continue
        try:
            event = json.loads(line[6:])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(event, dict):
            continue
        if event.get("conversation_id"):
            conversation_id = str(event["conversation_id"])
        value = event.get("v")
        if isinstance(value, dict):
            if value.get("conversation_id"):
                conversation_id = str(value["conversation_id"])
            message = value.get("message")
            if (
                isinstance(message, dict)
                and (message.get("author") or {}).get("role") == "assistant"
                and (message.get("content") or {}).get("content_type") == "text"
            ):
                current_assistant = str(message.get("id") or "")
                message_id = current_assistant
                parts = (message.get("content") or {}).get("parts") or []
                reply = parts[0] if parts and isinstance(parts[0], str) else ""
        elif isinstance(value, list) and current_assistant:
            for patch in value:
                if (
                    isinstance(patch, dict)
                    and patch.get("p") == "/message/content/parts/0"
                    and patch.get("o") == "append"
                    and isinstance(patch.get("v"), str)
                ):
                    reply += patch["v"]
    return conversation_id, message_id, reply.strip()


def _poll_conversation_message(
    session: requests.Session,
    token: str,
    device_id: str,
    conversation_id: str,
    budget: _OperationBudget,
    *,
    account_id: Optional[str] = None,
) -> tuple[str, str]:
    """Recover a completed assistant message when the SSE switched transports early."""
    if not conversation_id:
        return "", ""
    path = f"/backend-api/conversation/{conversation_id}"
    poll_budget = _OperationBudget(
        min(budget.deadline, time.monotonic() + 15.0), budget.cancel_event
    )
    while time.monotonic() < poll_budget.deadline:
        try:
            _, raw = _request(
                session,
                "GET",
                path,
                headers={**_headers(token, device_id, account_id=account_id), "x-openai-target-path": path},
                budget=poll_budget,
                limit=_MAX_JSON_BYTES,
            )
            data = json.loads(raw)
        except (ValueError, json.JSONDecodeError):
            data = {}
        mapping = data.get("mapping") if isinstance(data, dict) else None
        if not isinstance(mapping, dict):
            mapping = {}
        ordered = []
        current = str(data.get("current_node") or "") if isinstance(data, dict) else ""
        if current in mapping:
            ordered.append(mapping[current])
        ordered.extend(node for key, node in mapping.items() if key != current)
        for node in ordered:
            message = node.get("message") if isinstance(node, dict) else None
            if (
                not isinstance(message, dict)
                or (message.get("author") or {}).get("role") != "assistant"
            ):
                continue
            content = message.get("content") or {}
            parts = content.get("parts") or []
            if (
                content.get("content_type") == "text"
                and parts
                and isinstance(parts[0], str)
                and parts[0].strip()
            ):
                return str(message.get("id") or ""), parts[0].strip()
        delay = min(0.35, max(0, poll_budget.deadline - time.monotonic()))
        if budget.cancel_event is None:
            time.sleep(delay)
        else:
            budget.cancel_event.wait(delay)
        budget.remaining()
    return "", ""


def _sentinel(
    session: requests.Session, token: str, device_id: str, budget: _OperationBudget,
    *, account_id: Optional[str] = None,
) -> tuple[str, str]:
    config = _fingerprint_array(device_id)
    headers = {
        **_headers(token, device_id, account_id=account_id),
        "content-type": "application/json",
        "accept": "*/*",
    }
    _, raw = _request(
        session,
        "POST",
        _SENTINEL_PREPARE,
        headers=headers,
        budget=budget,
        json_body={"p": _requirements_token(config)},
    )
    prepared = json.loads(raw)
    challenge = prepared.get("proofofwork") or {}
    proof = ""
    if challenge.get("required"):
        seed = str(challenge.get("seed") or "")
        difficulty = str(challenge.get("difficulty") or "")
        if not seed or not difficulty:
            raise RuntimeError("ChatGPT Sentinel required PoW without seed/difficulty")
        proof = _solve_pow(config, seed, difficulty, budget)
    finalize = {"prepare_token": str(prepared.get("prepare_token") or "")}
    if proof:
        finalize["proofofwork"] = proof
    _, raw = _request(
        session,
        "POST",
        _SENTINEL_FINALIZE,
        headers=headers,
        budget=budget,
        json_body=finalize,
    )
    requirement = str(json.loads(raw).get("token") or "")
    if not requirement:
        raise RuntimeError("ChatGPT Sentinel finalize returned no requirements token")
    return requirement, proof


def _conduit(
    session: requests.Session,
    token: str,
    device_id: str,
    trace_id: str,
    final_body: dict,
    state: str,
    conduit_token: str,
    budget: _OperationBudget,
    *,
    account_id: Optional[str] = None,
) -> str:
    headers = {
        **_headers(token, device_id, account_id=account_id),
        "content-type": "application/json",
        "accept": "*/*",
        "x-oai-turn-trace-id": trace_id,
        "x-openai-target-path": _CONVERSATION_PREPARE,
        "x-conduit-token": conduit_token,
    }
    _, raw = _request(
        session,
        "POST",
        _CONVERSATION_PREPARE,
        headers=headers,
        budget=budget,
        json_body=_prepare_body(
            final_body, state, final_body["messages"][0]["content"]["parts"][0]
        ),
    )
    return str(json.loads(raw).get("conduit_token") or "")


def synthesize_codex_speech(
    text: str,
    token: str,
    *,
    voice: str = "juniper",
    account_id: Optional[str] = None,
    timeout: float = 120,
    cancel_event: Optional[Event] = None,
    _retry_mismatch: bool = True,
) -> CodexSpeech:
    """Create verified, history-disabled read-aloud audio within one operation budget.

    Cancellation and deadline checks run between requests and streamed chunks;
    an in-flight synchronous socket read remains bounded by its request timeout.
    """
    requested = text.strip()
    if not requested:
        raise ValueError("Codex subscription TTS input is empty")
    if len(requested) > 4096:
        raise ValueError("Codex subscription TTS input exceeds 4096 characters")
    selected_voice = resolve_voice(voice)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Codex subscription TTS timeout must be positive and finite")
    budget = _OperationBudget(time.monotonic() + timeout, cancel_event)
    budget.remaining()
    prompt = "Return exactly this text, with no quotes or changes: " + requested
    device_id = str(uuid.uuid4())
    with requests.Session() as session:
        # Cookies are obtained from response headers: never download warmup HTML.
        # Share a small warmup budget and retain its cookies for the mismatch retry.
        warmup_deadline = min(budget.deadline, time.monotonic() + 5.0)
        for path in ("/", "/api/auth/session", "/cdn-cgi/trace"):
            budget.remaining()
            remaining = warmup_deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                response = session.get(
                    _BASE_URL + path,
                    timeout=min(remaining, budget.remaining()),
                    allow_redirects=False,
                    stream=True,
                )
                response.close()
            except requests.RequestException:
                pass

        for attempt in range(2 if _retry_mismatch else 1):
            budget.remaining()
            trace_id = str(uuid.uuid4())
            final_body = _conversation_body(prompt)
            conduit = _conduit(
                session, token, device_id, trace_id, final_body, "none", "no-token", budget,
                account_id=account_id,
            )
            requirement, proof = _sentinel(
                session, token, device_id, budget, account_id=account_id
            )
            for _ in range(4):
                if not conduit:
                    break
                conduit = _conduit(
                    session, token, device_id, trace_id, final_body, "success", conduit, budget,
                    account_id=account_id,
                )

            headers = {
                **_headers(token, device_id, account_id=account_id, final=True),
                "content-type": "application/json",
                "accept": "text/event-stream",
                "openai-sentinel-chat-requirements-token": requirement,
                "openai-sentinel-proof-token": proof,
                "x-oai-turn-trace-id": trace_id,
                "x-openai-target-path": _CONVERSATION,
            }
            _, raw = _request(
                session,
                "POST",
                _CONVERSATION,
                headers=headers,
                budget=budget,
                json_body=final_body,
                limit=_MAX_SSE_BYTES,
            )
            conversation_id, message_id, spoken = _parse_conversation_sse(
                raw.decode("utf-8", "replace").splitlines()
            )
            if conversation_id and not spoken:
                polled_id, polled_text = _poll_conversation_message(
                    session, token, device_id, conversation_id, budget, account_id=account_id
                )
                message_id = polled_id or message_id
                spoken = polled_text or spoken
            if not conversation_id or not message_id:
                raise RuntimeError(
                    "ChatGPT conversation returned no synthesizable assistant message"
                )
            if spoken != requested:
                if _retry_mismatch and attempt == 0:
                    continue
                raise RuntimeError(
                    "ChatGPT changed the requested TTS text; refusing mismatched content: "
                    f"{spoken!r}"
                )

            synth_headers = {
                **_headers(token, device_id, account_id=account_id),
                "accept": "*/*",
                "x-openai-target-path": _SYNTHESIZE,
                "referer": f"{_BASE_URL}/c/{conversation_id}",
            }
            response, audio = _request(
                session,
                "GET",
                _SYNTHESIZE,
                headers=synth_headers,
                budget=budget,
                params={
                    "conversation_id": conversation_id,
                    "message_id": message_id,
                    "voice": selected_voice,
                    "format": "mp3",
                },
                limit=_MAX_AUDIO_BYTES,
            )
            content_type = (response.headers.get("content-type") or "").split(";", 1)[0].lower()
            if not content_type.startswith("audio/"):
                raise RuntimeError(
                    f"ChatGPT synthesize returned non-audio content type {content_type!r}"
                )
            if not audio:
                raise RuntimeError("ChatGPT synthesize returned empty audio")
            return CodexSpeech(audio, content_type, conversation_id, message_id, spoken)
