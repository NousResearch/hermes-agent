"""
wot_engine.py — Web-of-Thought multi-agent communication engine for Hermes.

Generic agents talk to each other through a shared message bus. No role taxonomy
hardcoded — agents are just (name, system_prompt) slots. Caller decides what each
agent does via its system_prompt.

Four communication modes:
  - "parallel"   : all agents run concurrently; each sees others' completed
                   messages on round boundaries
  - "streaming"  : like parallel, but agents see PARTIAL CoT tokens from peers
                   via chunked polling (works with any OpenAI-compat backend
                   without requiring server-side mid-generation injection)
  - "sequential" : round-robin; each agent gets the full prior transcript
  - "queue"      : topic-driven; agents pull from a shared queue when they
                   declare interest in a tag

Backbone: any OpenAI-compatible /v1/chat/completions endpoint, plus Ollama's
native /api/chat for thinking-mode models (where /v1/ silently drops thinking).
Auto-detected at construction time:
  Ollama, llama.cpp's llama-server, vLLM, LM Studio, OpenAI, OpenRouter, etc.

Termination: 3 layers — explicit DONE marker, max_rounds cap, per-agent timeout.
Cost control: optional token_budget per channel.
CoT propagation: default "strip" (cross-family-safe). Opt in to "raw" for
same-family debate where shared reasoning amplifies; this is informed by
arxiv 2503.13657 and the MemoryTrap (Apr 2026) attack surface.

Grounded in Artificial Neural Mesh V0 (Ali, 2026)
  https://doi.org/10.5281/zenodo.18112435

License: MIT
"""
from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Set, Tuple

import httpx

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration (env-overridable, OpenAI-compatible)
# ─────────────────────────────────────────────────────────────────────────────
LLM_BASE_URL = os.getenv("LLM_BASE_URL", os.getenv("OLLAMA_URL", "http://localhost:11434"))
LLM_DEFAULT_MODEL = os.getenv("LLM_DEFAULT_MODEL", "qwen3:4b-instruct")
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-no-key-required")  # most local backends don't check

DEFAULT_MAX_TOKENS_PER_TURN = 800
DEFAULT_STREAM_CHUNK_TOKENS = 32     # how many tokens of peer-stream a listener absorbs at a time
DEFAULT_REQUEST_TIMEOUT = 300        # seconds — overall HTTP read timeout
DEFAULT_AGENT_TURN_TIMEOUT = 240     # seconds — per-turn cap; sub-LL_REQUEST_TIMEOUT so we cancel before httpx
DEFAULT_TOKEN_BUDGET = 0             # 0 = unbounded (in chars; ~4 chars/token estimate)

# Type aliases
PropagateMode = Literal["strip", "raw", "summary"]
BackendKind = Literal["llama-server", "ollama", "vllm", "openai-compat"]


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Message:
    """One message on the bus.

    `content` is the agent's user-facing answer. `reasoning` is the agent's
    chain-of-thought (separately tracked so we can choose how / whether to
    propagate it to peers — see PropagateMode).
    """
    from_agent: str
    content: str
    reasoning: str = ""                  # CoT, separate from final content
    to_agent: Optional[str] = None       # None = broadcast
    tags: List[str] = field(default_factory=list)
    ts: float = field(default_factory=time.time)
    round: int = 0
    seq: int = 0                         # per-agent monotonic sequence for stream debugging
    is_chunk: bool = False               # True = streaming partial; False = completed turn
    chunk_index: int = 0
    is_final_chunk: bool = False
    chunk_kind: Literal["content", "reasoning"] = "content"  # which channel a streaming chunk came from

    def to_dict(self, include_reasoning: bool = True) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "from": self.from_agent,
            "to": self.to_agent,
            "content": self.content,
            "tags": self.tags,
            "ts": datetime.datetime.fromtimestamp(self.ts).isoformat(timespec="seconds"),
            "round": self.round,
            "seq": self.seq,
            "is_chunk": self.is_chunk,
            "chunk_index": self.chunk_index,
            "is_final_chunk": self.is_final_chunk,
            "chunk_kind": self.chunk_kind,
        }
        if include_reasoning and self.reasoning:
            d["reasoning"] = self.reasoning
        return d

    def char_len(self) -> int:
        return len(self.content) + len(self.reasoning)


def _is_thinking_model(name: str) -> bool:
    """Detect models that emit <think>...</think> reasoning by default.

    These need much higher max_tokens because budget is consumed inside
    the thinking block before any answer is produced.
    Covers: DeepSeek-R1 + distills, QwQ family, *-thinking. Does NOT match
    Qwen3.5/3.6 — those default to thinking but we disable via
    chat_template_kwargs.enable_thinking=false at the request layer.
    """
    n = (name or "").lower().replace("_", "-")
    if "deepseek-r1" in n or "qwq" in n:
        return True
    if "thinking" in n and "qwen3.5" not in n and "qwen3.6" not in n:
        return True
    return False


def _is_qwen35_or_36(model: str) -> bool:
    n = (model or "").lower().replace("_", ".")
    return "qwen3.5" in n or "qwen3.6" in n


@dataclass
class AgentSpec:
    """Caller-supplied agent definition. NO role taxonomy — caller writes system_prompt."""
    name: str
    system_prompt: str
    model: Optional[str] = None
    interests: List[str] = field(default_factory=list)   # tags this agent listens for in queue mode
    max_tokens: int = DEFAULT_MAX_TOKENS_PER_TURN
    temperature: float = 0.7
    turn_timeout: float = DEFAULT_AGENT_TURN_TIMEOUT     # per-turn cap

    def __post_init__(self):
        # Auto-sanitize: collapse whitespace runs to underscores, drop other
        # disallowed characters. Real LLM callers emit names like
        # "Critical Thinker" or "agent A" — we accept that and normalize
        # rather than error-out, since name is a label not an identifier.
        if isinstance(self.name, str):
            sanitized = re.sub(r"\s+", "_", self.name.strip())
            sanitized = re.sub(r"[^A-Za-z0-9_\-]", "", sanitized)
            if sanitized:
                self.name = sanitized
        if not re.match(r"^[A-Za-z0-9_\-]+$", self.name or ""):
            raise ValueError(f"agent name empty or invalid after sanitization: {self.name!r}")
        if not self.model:
            self.model = LLM_DEFAULT_MODEL
        # Auto-bump max_tokens for thinking-mode models if caller used the default.
        if (self.max_tokens == DEFAULT_MAX_TOKENS_PER_TURN
                and _is_thinking_model(self.model)):
            self.max_tokens = 2500


# ─────────────────────────────────────────────────────────────────────────────
# Backend detection — probe once at construction
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BackendInfo:
    """What we learned about the OpenAI-compat endpoint at the other end.

    Drives per-backend code paths:
      - llama-server: id_slot pinning, cache_prompt, /props for confirmation
      - ollama:       /api/chat for thinking models (the /v1/ path drops
                      reasoning_content for many models — issues #15293, #15635)
      - vllm:         standard /v1/, reasoning_content reliable, no slot ctrl
      - openai-compat: minimum-feature fallback
    """
    kind: BackendKind = "openai-compat"
    base_url: str = ""
    supports_slots: bool = False         # llama-server only
    has_native_thinking_api: bool = False  # ollama only
    raw_props: Optional[Dict[str, Any]] = None


async def _probe_backend(base_url: str, api_key: str, client: httpx.AsyncClient) -> BackendInfo:
    """Best-effort backend identification. Falls back to openai-compat on any failure."""
    base = base_url.rstrip("/")
    info = BackendInfo(base_url=base)
    headers = {"Authorization": f"Bearer {api_key}"}
    # Try llama-server's GET /props
    try:
        r = await client.get(f"{base}/props", headers=headers, timeout=5.0)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict) and ("default_generation_settings" in data
                                            or "build_info" in data
                                            or "model_path" in data):
                info.kind = "llama-server"
                info.supports_slots = True
                info.raw_props = {"detected": "llama-server"}
                return info
    except Exception:
        pass
    # Try Ollama's GET /api/version
    try:
        r = await client.get(f"{base}/api/version", timeout=5.0)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict) and "version" in data:
                info.kind = "ollama"
                info.has_native_thinking_api = True
                info.raw_props = {"detected": "ollama", "version": data.get("version")}
                return info
    except Exception:
        pass
    # Try vLLM's /version (vLLM exposes it; also returns at /v1/models with vllm-specific fields)
    try:
        r = await client.get(f"{base}/version", timeout=5.0)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict) and "version" in data and "vllm" in str(data).lower():
                info.kind = "vllm"
                info.raw_props = {"detected": "vllm"}
                return info
    except Exception:
        pass
    # Default: assume OpenAI-compat
    return info


# ─────────────────────────────────────────────────────────────────────────────
# Message bus — async pub/sub with token-budget tracking
# ─────────────────────────────────────────────────────────────────────────────
class Channel:
    """Async multi-subscriber message bus.

    Each agent gets its own asyncio.Queue. Tracks cumulative chars (≈tokens÷4)
    so the engine can abort runaway loops on a hard budget — which round-caps
    alone don't bound (one round can blow up context).
    """
    def __init__(self) -> None:
        self.history: List[Message] = []
        self._subscribers: Dict[str, asyncio.Queue[Message]] = {}
        self._seq_per_agent: Dict[str, int] = {}
        self._lock = asyncio.Lock()
        self.total_chars: int = 0   # cumulative across all messages, including chunks

    def subscribe(self, agent_name: str) -> asyncio.Queue[Message]:
        if agent_name not in self._subscribers:
            self._subscribers[agent_name] = asyncio.Queue()
        return self._subscribers[agent_name]

    def next_seq(self, agent_name: str) -> int:
        n = self._seq_per_agent.get(agent_name, 0) + 1
        self._seq_per_agent[agent_name] = n
        return n

    async def publish(self, msg: Message) -> None:
        async with self._lock:
            if msg.seq == 0:
                msg.seq = self.next_seq(msg.from_agent)
            self.history.append(msg)
            self.total_chars += msg.char_len()
            for sub_name, queue in self._subscribers.items():
                if sub_name == msg.from_agent:
                    continue
                if msg.to_agent is not None and msg.to_agent != sub_name:
                    continue
                await queue.put(msg)

    def transcript(self, include_chunks: bool = False,
                   include_reasoning: bool = True) -> List[Dict[str, Any]]:
        return [m.to_dict(include_reasoning=include_reasoning)
                for m in self.history if include_chunks or not m.is_chunk]


# ─────────────────────────────────────────────────────────────────────────────
# LLM client — backend-aware async client
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LLMResponse:
    """Structured response so the agent layer can keep reasoning + content separate."""
    content: str = ""
    reasoning: str = ""

    def is_empty(self) -> bool:
        return not (self.content.strip() or self.reasoning.strip())


class _LLMClient:
    """Async, streaming + non-streaming OpenAI-compat client with per-backend dispatch.

    On llama-server: passes id_slot for KV-cache pinning + cache_prompt: true.
    On Ollama with thinking models: uses /api/chat (think:true), reads
    `message.thinking`, since the OpenAI-compat /v1/ endpoint drops thinking
    output for several models (Ollama issues #15293, #15635 in May 2026).
    """

    def __init__(self, base_url: str = LLM_BASE_URL, api_key: str = LLM_API_KEY,
                 timeout: float = DEFAULT_REQUEST_TIMEOUT):
        # Normalize base_url: strip trailing "/" and "/v1" so we can always
        # append "/v1/chat/completions" without doubling. Users tend to set
        # LLM_BASE_URL to either form (e.g. "http://127.0.0.1:11434" for Ollama,
        # "http://127.0.0.1:8088/v1" for llama-server, "https://openrouter.ai/api/v1"
        # for OpenRouter); double "/v1/" lands on a 404 → empty body → JSONDecodeError.
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3].rstrip("/")
        self.api_key = api_key
        # Explicit timeouts per phase. read=None on streaming would let server
        # hang forever; we keep a finite read but rely on per-turn asyncio
        # timeouts upstairs. limits keepalive for proxy survival.
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=timeout, write=30.0, pool=10.0),
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=40,
                keepalive_expiry=30.0,
            ),
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self.backend: BackendInfo = BackendInfo(base_url=self.base_url)
        self._probed = False

    async def aclose(self) -> None:
        await self._client.aclose()

    async def ensure_probed(self) -> BackendInfo:
        if not self._probed:
            self.backend = await _probe_backend(self.base_url, self.api_key, self._client)
            self._probed = True
            logger.info("WoT backend probe: %s at %s", self.backend.kind, self.backend.base_url)
        return self.backend

    # ───────── payload builders ─────────
    def _openai_payload(self, model: str, messages: List[Dict[str, Any]],
                        max_tokens: int, temperature: float, stream: bool,
                        slot_id: Optional[int], stop: Optional[List[str]]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if _is_qwen35_or_36(model):
            # Qwen3.5/3.6 default to thinking and ignore /think /no_think.
            # Per official model card, only chat_template_kwargs disables it.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if stop:
            payload["stop"] = stop
        # llama-server-only knobs: pin slot + reuse prompt prefix across turns.
        if self.backend.kind == "llama-server" and slot_id is not None:
            payload["id_slot"] = slot_id
            payload["cache_prompt"] = True
        return payload

    def _ollama_native_payload(self, model: str, messages: List[Dict[str, Any]],
                               max_tokens: int, temperature: float, stream: bool,
                               think: bool) -> Dict[str, Any]:
        """Ollama native /api/chat payload. `think` enables the structured thinking field."""
        return {
            "model": model,
            "messages": messages,
            "stream": stream,
            "think": think,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

    # ───────── non-streaming complete ─────────
    async def complete(self, model: str, messages: List[Dict[str, Any]],
                       max_tokens: int = DEFAULT_MAX_TOKENS_PER_TURN,
                       temperature: float = 0.7,
                       slot_id: Optional[int] = None,
                       stop: Optional[List[str]] = None) -> LLMResponse:
        await self.ensure_probed()
        thinking_model = _is_thinking_model(model)
        # Ollama path for thinking models: /v1/ drops thinking, /api/chat keeps it.
        if self.backend.kind == "ollama" and thinking_model:
            payload = self._ollama_native_payload(model, messages, max_tokens,
                                                  temperature, stream=False, think=True)
            r = await self._client.post(f"{self.base_url}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
            msg = data.get("message", {}) if isinstance(data, dict) else {}
            return LLMResponse(
                content=(msg.get("content") or "").strip(),
                reasoning=(msg.get("thinking") or "").strip(),
            )
        # OpenAI-compat path
        payload = self._openai_payload(model, messages, max_tokens, temperature,
                                        stream=False, slot_id=slot_id, stop=stop)
        r = await self._client.post(f"{self.base_url}/v1/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
        msg = data["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning_content") or "").strip()
        # If the server didn't split out reasoning but content has raw <think> tags,
        # split client-side so peers see clean content downstream.
        if not reasoning and "<think>" in content:
            m = re.search(r"<think>(.*?)</think>\s*", content, re.DOTALL)
            if m:
                reasoning = m.group(1).strip()
                content = (content[:m.start()] + content[m.end():]).strip()
        return LLMResponse(content=content, reasoning=reasoning)

    # ───────── streaming ─────────
    async def stream(self, model: str, messages: List[Dict[str, Any]],
                     max_tokens: int = DEFAULT_MAX_TOKENS_PER_TURN,
                     temperature: float = 0.7,
                     slot_id: Optional[int] = None,
                     stop: Optional[List[str]] = None
                     ) -> AsyncIterator[Tuple[str, str]]:
        """Yields (kind, delta) where kind ∈ {"content", "reasoning"}."""
        await self.ensure_probed()
        thinking_model = _is_thinking_model(model)
        if self.backend.kind == "ollama" and thinking_model:
            async for kv in self._stream_ollama_native(model, messages, max_tokens,
                                                        temperature, think=True):
                yield kv
            return
        async for kv in self._stream_openai(model, messages, max_tokens, temperature,
                                             slot_id, stop):
            yield kv

    async def _stream_openai(self, model: str, messages: List[Dict[str, Any]],
                             max_tokens: int, temperature: float,
                             slot_id: Optional[int], stop: Optional[List[str]]
                             ) -> AsyncIterator[Tuple[str, str]]:
        payload = self._openai_payload(model, messages, max_tokens, temperature,
                                        stream=True, slot_id=slot_id, stop=stop)
        async with self._client.stream(
            "POST", f"{self.base_url}/v1/chat/completions", json=payload,
        ) as r:
            r.raise_for_status()
            async for raw in r.aiter_lines():
                if not raw or not raw.startswith("data:"):
                    continue
                chunk_data = raw[5:].strip()
                if chunk_data == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk_data)
                except json.JSONDecodeError:
                    continue
                delta_obj = obj.get("choices", [{}])[0].get("delta", {})
                # llama-server with --jinja for thinking-mode models routes thinking
                # tokens to delta.reasoning_content. Both fields can appear within
                # the same stream (reasoning during <think>, content after </think>).
                rc = delta_obj.get("reasoning_content")
                if rc:
                    yield ("reasoning", rc)
                c = delta_obj.get("content")
                if c:
                    yield ("content", c)

    async def _stream_ollama_native(self, model: str, messages: List[Dict[str, Any]],
                                     max_tokens: int, temperature: float, think: bool
                                     ) -> AsyncIterator[Tuple[str, str]]:
        """Ollama native /api/chat streams NDJSON, not SSE."""
        payload = self._ollama_native_payload(model, messages, max_tokens, temperature,
                                               stream=True, think=think)
        async with self._client.stream(
            "POST", f"{self.base_url}/api/chat", json=payload,
        ) as r:
            r.raise_for_status()
            async for raw in r.aiter_lines():
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                msg = obj.get("message") or {}
                t = msg.get("thinking")
                if t:
                    yield ("reasoning", t)
                c = msg.get("content")
                if c:
                    yield ("content", c)
                if obj.get("done"):
                    break


# ─────────────────────────────────────────────────────────────────────────────
# Agent — generic LLM-backed slot that reads inbox, writes to channel
# ─────────────────────────────────────────────────────────────────────────────
class Agent:
    """A named LLM slot. Behaviour determined entirely by spec.system_prompt."""

    def __init__(self, spec: AgentSpec, channel: Channel, client: _LLMClient,
                 task: str, slot_id: Optional[int] = None,
                 propagate_reasoning: PropagateMode = "strip"):
        self.spec = spec
        self.channel = channel
        self.client = client
        self.task = task
        self.slot_id = slot_id   # pinned KV-cache slot for llama-server (None elsewhere)
        self.propagate_reasoning = propagate_reasoning
        self.inbox = channel.subscribe(spec.name)
        self.transcript_seen: List[Message] = []

    def _drain_inbox(self) -> List[Message]:
        out: List[Message] = []
        while not self.inbox.empty():
            try:
                out.append(self.inbox.get_nowait())
            except asyncio.QueueEmpty:
                break
        return out

    def _peer_render(self, m: Message) -> str:
        """Render a peer message into this agent's context, respecting propagate_reasoning.

        - "strip"  : final content only (cross-family-safe default; avoids style
                     contamination + MemoryTrap-class CoT injection)
        - "raw"    : show <think>...</think> + content (same-family debate / refinement)
        - "summary": placeholder — currently same as "strip"; in production this
                     would be a 1-2 sentence distill of m.reasoning by a small model.
        """
        tag = "chunk" if m.is_chunk else "msg"
        body = m.content
        if self.propagate_reasoning == "raw" and m.reasoning and not m.is_chunk:
            body = f"<think>\n{m.reasoning}\n</think>\n\n{m.content}"
        elif m.is_chunk and m.chunk_kind == "reasoning":
            # Even in "raw" mode, streaming reasoning chunks are tagged distinctly
            # so the receiver sees they're partial CoT not partial content.
            if self.propagate_reasoning == "raw":
                body = f"[partial CoT] {m.content}"
            else:
                # In strip/summary, we drop streaming reasoning chunks entirely.
                return ""
        return f"[from {m.from_agent} | {tag} | seq {m.seq}]: {body}"

    def _build_messages(self, new_peer_msgs: List[Message]) -> List[Dict[str, Any]]:
        sys_prompt = (
            f"{self.spec.system_prompt}\n\n"
            f"You are agent '{self.spec.name}' participating in a Web-of-Thought session "
            f"with other agents. Other agents may speak via messages prefixed `[from X]: ...`. "
            f"You can reply to them by addressing them as `@X: ...` in your output, "
            f"or you can broadcast (just write normally). When you have nothing more to add, "
            f"reply with the single word DONE on its own line."
        )
        msgs: List[Dict[str, Any]] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"TASK: {self.task}"},
        ]
        all_seen = self.transcript_seen + new_peer_msgs
        for m in all_seen:
            rendered = self._peer_render(m)
            if rendered:
                msgs.append({"role": "user", "content": rendered})
        self.transcript_seen = all_seen
        return msgs

    async def turn_batch(self, round_no: int) -> Tuple[Optional[Message], Optional[str]]:
        """One non-streaming turn. Returns (Message, None) on success, (None, error_str) on failure."""
        peer_msgs = self._drain_inbox()
        messages = self._build_messages(peer_msgs)
        try:
            resp = await asyncio.wait_for(
                self.client.complete(
                    self.spec.model, messages,
                    max_tokens=self.spec.max_tokens,
                    temperature=self.spec.temperature,
                    slot_id=self.slot_id,
                ),
                timeout=self.spec.turn_timeout,
            )
        except asyncio.TimeoutError:
            return None, f"turn timed out after {self.spec.turn_timeout}s"
        except Exception as e:
            err = f"LLM call failed: {type(e).__name__}: {e}"
            logger.warning("agent %s %s", self.spec.name, err)
            return None, err
        if resp.is_empty():
            return None, ("empty response (likely max_tokens exhausted in <think>; "
                          "raise max_tokens or pick a non-thinking model)")
        content = resp.content
        is_done = False
        if content.strip().splitlines() and content.strip().splitlines()[-1].strip().upper() == "DONE":
            content = "\n".join(content.strip().splitlines()[:-1]).strip() or "(no response)"
            is_done = True
        msg = Message(
            from_agent=self.spec.name,
            content=content,
            reasoning=resp.reasoning,
            round=round_no,
            tags=["DONE"] if is_done else [],
        )
        await self.channel.publish(msg)
        return msg, None

    async def turn_streaming(self, round_no: int,
                             chunk_size: int = DEFAULT_STREAM_CHUNK_TOKENS
                             ) -> Tuple[Optional[Message], Optional[str]]:
        """One streaming turn. Reasoning and content streams are tracked separately;
        chunks are published with chunk_kind so peers can route accordingly."""
        peer_msgs = self._drain_inbox()
        messages = self._build_messages(peer_msgs)
        content_buf: List[str] = []
        reasoning_buf: List[str] = []
        chunk_pending: Dict[str, List[str]] = {"content": [], "reasoning": []}
        chunk_idx = 0

        async def flush_chunk(kind: str, final: bool = False) -> None:
            nonlocal chunk_idx
            buf = chunk_pending[kind]
            if not buf:
                return
            text = "".join(buf)
            chunk_pending[kind] = []
            await self.channel.publish(Message(
                from_agent=self.spec.name,
                content=text,
                round=round_no,
                is_chunk=True,
                chunk_index=chunk_idx,
                is_final_chunk=final,
                chunk_kind="reasoning" if kind == "reasoning" else "content",
            ))
            chunk_idx += 1

        try:
            async def consume() -> None:
                async for kind, delta in self.client.stream(
                    self.spec.model, messages,
                    max_tokens=self.spec.max_tokens,
                    temperature=self.spec.temperature,
                    slot_id=self.slot_id,
                ):
                    if kind == "reasoning":
                        reasoning_buf.append(delta)
                        chunk_pending["reasoning"].append(delta)
                        if sum(len(c.split()) for c in chunk_pending["reasoning"]) >= chunk_size:
                            await flush_chunk("reasoning")
                    else:
                        content_buf.append(delta)
                        chunk_pending["content"].append(delta)
                        if sum(len(c.split()) for c in chunk_pending["content"]) >= chunk_size:
                            await flush_chunk("content")

            await asyncio.wait_for(consume(), timeout=self.spec.turn_timeout)
        except asyncio.TimeoutError:
            await flush_chunk("reasoning", final=True)
            await flush_chunk("content", final=True)
            return None, f"stream timed out after {self.spec.turn_timeout}s"
        except Exception as e:
            err = f"stream failed: {type(e).__name__}: {e}"
            logger.warning("agent %s %s", self.spec.name, err)
            return None, err

        # Tail flushes
        await flush_chunk("reasoning")
        await flush_chunk("content", final=True)

        full_content = "".join(content_buf).strip()
        full_reasoning = "".join(reasoning_buf).strip()
        if not full_content and not full_reasoning:
            return None, ("empty stream (likely max_tokens exhausted; "
                          "raise max_tokens or pick a non-thinking model)")
        is_done = False
        if full_content.splitlines() and full_content.splitlines()[-1].strip().upper() == "DONE":
            full_content = "\n".join(full_content.splitlines()[:-1]).strip() or "(no response)"
            is_done = True
        # If only reasoning came back (e.g. R1 ran out of budget mid-think with no
        # final answer), surface reasoning as the content so peers see something.
        if not full_content and full_reasoning:
            full_content = "(no final content — agent ran out of budget mid-thinking)"
        final = Message(
            from_agent=self.spec.name,
            content=full_content,
            reasoning=full_reasoning,
            round=round_no,
            is_chunk=False,
            tags=["DONE"] if is_done else [],
        )
        await self.channel.publish(final)
        return final, None


# ─────────────────────────────────────────────────────────────────────────────
# Engine — orchestrates rounds across the four modes
# ─────────────────────────────────────────────────────────────────────────────
class WoTEngine:
    """Multi-mode multi-agent orchestrator. No role enforcement.

    Public entry: run(specs, task, mode, max_rounds, ...) → transcript dict.

    Termination is enforced in three layers (recommended by 2026 multi-agent
    literature, e.g. arxiv 2510.12697):
      1. agent emits DONE on its own line → drops out of active set
      2. max_rounds hard cap
      3. asyncio.wait_for per turn (spec.turn_timeout)
    Plus an optional 4th layer: cumulative token_budget per channel.

    base_url defaults are RE-READ from env at construction time so callers can
    set LLM_BASE_URL after import.
    """

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.client = _LLMClient(
            base_url=base_url or os.getenv("LLM_BASE_URL",
                                            os.getenv("OLLAMA_URL", "http://localhost:11434")),
            api_key=api_key or os.getenv("LLM_API_KEY", "sk-no-key-required"),
        )

    async def aclose(self) -> None:
        await self.client.aclose()

    async def run(
        self,
        specs: List[AgentSpec],
        task: str,
        mode: str = "parallel",
        max_rounds: int = 5,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        propagate_reasoning: PropagateMode = "strip",
    ) -> Dict[str, Any]:
        if mode not in ("parallel", "streaming", "sequential", "queue"):
            raise ValueError(f"unknown mode: {mode!r}")
        if propagate_reasoning not in ("strip", "raw", "summary"):
            raise ValueError(f"unknown propagate_reasoning: {propagate_reasoning!r}")

        # Probe once up front so logging shows backend before first round.
        await self.client.ensure_probed()
        backend_kind = self.client.backend.kind

        channel = Channel()
        agents: List[Agent] = []
        for i, s in enumerate(specs):
            # Slot pinning only meaningful on llama-server; harmless elsewhere.
            slot_id = i if backend_kind == "llama-server" else None
            agents.append(Agent(s, channel, self.client, task,
                                slot_id=slot_id,
                                propagate_reasoning=propagate_reasoning))

        done: Set[str] = set()
        rounds_run = 0
        errors: List[Dict[str, Any]] = []
        budget_hit = False

        for round_no in range(1, max_rounds + 1):
            if token_budget and channel.total_chars >= token_budget * 4:  # ~4 chars/token
                budget_hit = True
                break
            active_agents = [a for a in agents if a.spec.name not in done]
            if not active_agents:
                break

            if mode in ("parallel", "streaming"):
                turn_fn = (lambda a: a.turn_streaming(round_no)) if mode == "streaming" \
                          else (lambda a: a.turn_batch(round_no))
                results = await asyncio.gather(
                    *[turn_fn(a) for a in active_agents],
                    return_exceptions=True,
                )
                for a, r in zip(active_agents, results):
                    if isinstance(r, Exception):
                        errors.append({"agent": a.spec.name, "round": round_no,
                                       "reason": f"{type(r).__name__}: {r}"})
                        continue
                    msg, err = r
                    if err:
                        errors.append({"agent": a.spec.name, "round": round_no, "reason": err})
                    if msg and "DONE" in msg.tags:
                        done.add(a.spec.name)

            elif mode == "sequential":
                for a in active_agents:
                    if token_budget and channel.total_chars >= token_budget * 4:
                        budget_hit = True
                        break
                    msg, err = await a.turn_batch(round_no)
                    if err:
                        errors.append({"agent": a.spec.name, "round": round_no, "reason": err})
                    if msg and "DONE" in msg.tags:
                        done.add(a.spec.name)
                if budget_hit:
                    break

            elif mode == "queue":
                for a in active_agents:
                    if token_budget and channel.total_chars >= token_budget * 4:
                        budget_hit = True
                        break
                    eligible = False
                    if round_no == 1:
                        eligible = True
                    else:
                        peek = list(a.inbox._queue)
                        for m in peek:
                            if not a.spec.interests:
                                eligible = True
                                break
                            if any(t in a.spec.interests for t in m.tags):
                                eligible = True
                                break
                    if eligible:
                        msg, err = await a.turn_batch(round_no)
                        if err:
                            errors.append({"agent": a.spec.name, "round": round_no, "reason": err})
                        if msg and "DONE" in msg.tags:
                            done.add(a.spec.name)
                if budget_hit:
                    break

            rounds_run = round_no

        return {
            "task": task,
            "mode": mode,
            "rounds_run": rounds_run,
            "agents_done": sorted(done),
            "errors": errors,
            "backend": {
                "kind": self.client.backend.kind,
                "base_url": self.client.backend.base_url,
            },
            "stop_reason": ("budget" if budget_hit
                            else "all_done" if not [a for a in agents if a.spec.name not in done]
                            else "max_rounds"),
            "total_chars": channel.total_chars,
            "transcript": channel.transcript(include_chunks=False),
            "transcript_with_chunks": channel.transcript(include_chunks=True)
                                       if mode == "streaming" else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Hermes tool entry point
# ─────────────────────────────────────────────────────────────────────────────
def wot_chat_tool(
    agents: List[Dict[str, Any]],
    task: str,
    mode: str = "parallel",
    max_rounds: int = 5,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    propagate_reasoning: PropagateMode = "strip",
) -> str:
    """Hermes tool: run a Web-of-Thought multi-agent session and return JSON transcript.

    Args:
        agents: list of {"name", "system_prompt", "model"?, "interests"?, "max_tokens"?,
                "temperature"?, "turn_timeout"?}. NO role taxonomy hardcoded.
        task: initial task seed (broadcast to all agents).
        mode: "parallel" | "streaming" | "sequential" | "queue".
        max_rounds: max conversation rounds before forced termination.
        token_budget: cumulative token cap across all messages (0 = unbounded).
        propagate_reasoning: "strip" (cross-family-safe default) | "raw" (same-family
                             debate) | "summary" (currently same as strip).
    """
    # Outer Hermes models tend to hallucinate inner-agent model names like
    # "gpt-4o" when constructing the wot_chat tool call. We strip caller-
    # supplied `model` fields and force every inner agent to use
    # LLM_DEFAULT_MODEL (env-driven), so the WoT engine remains in control of
    # its own backend rather than chasing whatever the outer model dreamed up.
    sanitized: List[Dict[str, Any]] = []
    for a in agents:
        spec = dict(a)
        spec.pop("model", None)
        sanitized.append(spec)
    specs = [AgentSpec(**a) for a in sanitized]

    async def _run_and_close() -> Dict[str, Any]:
        # CRITICAL: engine + run + aclose all share ONE event loop. Calling
        # asyncio.run() twice (one for run, one for aclose) closes the loop
        # between them and any resource (httpx client, asyncio.Queue) tied
        # to the first loop fails with "Event loop is closed" on the second.
        engine = WoTEngine()
        try:
            return await engine.run(
                specs, task, mode=mode, max_rounds=max_rounds,
                token_budget=token_budget, propagate_reasoning=propagate_reasoning,
            )
        finally:
            await engine.aclose()

    # Detect whether we're already inside a running loop (e.g. invoked from
    # an async Hermes tool dispatcher). If yes, run on a separate thread so
    # asyncio.run() doesn't conflict with the existing loop.
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is None:
        result = asyncio.run(_run_and_close())
    else:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            result = ex.submit(lambda: asyncio.run(_run_and_close())).result()
    return json.dumps(result, ensure_ascii=False)


wot_chat_tool_schema = {
    "type": "function",
    "function": {
        "name": "wot_chat",
        "description": (
            "Run a Web-of-Thought multi-agent session: 2-7 named agents talk to each "
            "other through a shared message bus over multiple rounds. No role "
            "taxonomy — caller supplies system_prompt per agent. Returns full "
            "transcript as JSON. Modes: parallel (batch concurrent), streaming "
            "(see partial CoT from peers), sequential (round-robin), queue "
            "(tag-driven pull). Auto-detects llama.cpp / Ollama / vLLM backends "
            "and pins KV-cache slots on llama-server for performance."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "agents": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 7,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "system_prompt": {"type": "string"},
                            "model": {"type": "string"},
                            "interests": {"type": "array", "items": {"type": "string"}},
                            "max_tokens": {"type": "integer", "default": 800},
                            "temperature": {"type": "number", "default": 0.7},
                            "turn_timeout": {"type": "number", "default": 240},
                        },
                        "required": ["name", "system_prompt"],
                    },
                },
                "task": {"type": "string"},
                "mode": {
                    "type": "string",
                    "enum": ["parallel", "streaming", "sequential", "queue"],
                    "default": "parallel",
                },
                "max_rounds": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
                "token_budget": {"type": "integer", "default": 0, "minimum": 0,
                                 "description": "0 = unbounded"},
                "propagate_reasoning": {
                    "type": "string",
                    "enum": ["strip", "raw", "summary"],
                    "default": "strip",
                    "description": ("How to share peer CoT. 'strip' = final content only "
                                    "(cross-family-safe default). 'raw' = full <think> "
                                    "blocks (same-family debate / refinement). 'summary' "
                                    "currently behaves as 'strip'."),
                },
            },
            "required": ["agents", "task"],
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke test
# ─────────────────────────────────────────────────────────────────────────────
async def _smoke_test() -> None:
    print("# wot_engine smoke test — 3 agents, parallel, max_rounds=2\n")
    specs = [
        AgentSpec(name="optimist",
                  system_prompt="You argue the optimistic case. Brief — 2-3 sentences per turn."),
        AgentSpec(name="skeptic",
                  system_prompt="You argue the skeptical case. Brief — 2-3 sentences per turn."),
        AgentSpec(name="synthesizer",
                  system_prompt="You synthesise the optimist and skeptic positions into a balanced view. Brief."),
    ]
    engine = WoTEngine()
    try:
        result = await engine.run(
            specs,
            task="Will small open-source language models become the dominant deployment paradigm in 2026?",
            mode="parallel",
            max_rounds=2,
        )
    finally:
        await engine.aclose()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_smoke_test())


# ─────────────────────────────────────────────────────────────────────────────
# Hermes tool registration — must be top-level for AST auto-discovery
# (tools/registry.py:_module_registers_tools scans tree.body only; wrapping
# the registry.register() call in try/except hides it from the scanner and
# the module is never imported into the registry).
# ─────────────────────────────────────────────────────────────────────────────
from tools.registry import registry  # noqa: E402

try:
    from toolsets import create_custom_toolset  # type: ignore
    create_custom_toolset(
        name="wot",
        description="Web-of-Thought multi-agent reasoning (3-7 agents debate / collaborate)",
        tools=["wot_chat"],
    )
except Exception:
    pass


def _wot_handler(args, **_kw):
    return wot_chat_tool(
        agents=args.get("agents", []),
        task=args.get("task", ""),
        mode=args.get("mode", "parallel"),
        max_rounds=int(args.get("max_rounds", 5)),
        token_budget=int(args.get("token_budget", 0)),
        propagate_reasoning=args.get("propagate_reasoning", "strip"),
    )


registry.register(
    name="wot_chat",
    toolset="wot",
    schema={
        "name": wot_chat_tool_schema["function"]["name"],
        "description": wot_chat_tool_schema["function"]["description"],
        "parameters": wot_chat_tool_schema["function"]["parameters"],
    },
    handler=_wot_handler,
    check_fn=lambda: True,
    emoji="🕸",
    max_result_size_chars=200_000,
)
