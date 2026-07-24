"""Unit tests for tools/wot_engine.py — mock LLM client, deterministic, fast.

Run:   python3 -m pytest test_wot_engine.py -v
       OR (without pytest):  python3 test_wot_engine.py

Covers:
  - Channel pub/sub (broadcast + direct + sender-skip + history + seq counter)
  - AgentSpec validation
  - DONE detection (own-line vs inline)
  - All 4 modes: parallel, streaming, sequential, queue
  - Streaming chunks publish before final message
  - Qwen3.5/3.6 chat_template_kwargs auto-injection
  - DONE early-exit short-circuits remaining rounds
  - Max-rounds boundary
  - LLMResponse reasoning/content separation
  - propagate_reasoning="strip" vs "raw" peer-message rendering
  - token_budget enforces early stop
  - per-agent turn_timeout
  - backend probe is awaited on run()
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import unittest
from pathlib import Path
from typing import AsyncIterator, Dict, List, Tuple
from unittest.mock import patch

from tools import wot_engine
from tools.wot_engine import (
    AgentSpec, Agent, BackendInfo, Channel, LLMResponse, Message, WoTEngine, _LLMClient,
)


# ─────────────────────────────────────────────────────────────────────────────
# Mock LLM — returns scripted responses by agent name + round
# ─────────────────────────────────────────────────────────────────────────────
class MockLLMClient:
    """Pretends to be _LLMClient. Caller provides per-agent scripted responses.

    Scripts may be:
      - str → becomes LLMResponse(content=str)
      - dict {"content": ..., "reasoning": ...} → LLMResponse with both fields
    """

    def __init__(self, scripts: Dict[str, List], backend_kind: str = "openai-compat"):
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.calls: List[Dict] = []
        self.payloads: List[Dict] = []
        self.backend = BackendInfo(kind=backend_kind, base_url="mock://test")
        self._probed = True

    @staticmethod
    def _is_qwen35_or_36(model: str) -> bool:
        n = (model or "").lower().replace("_", ".")
        return "qwen3.5" in n or "qwen3.6" in n

    def _payload(self, model, messages, max_tokens, temperature, stream, slot_id=None, stop=None):
        payload = {"model": model, "messages": messages,
                   "max_tokens": max_tokens, "temperature": temperature, "stream": stream}
        if self._is_qwen35_or_36(model):
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if stop:
            payload["stop"] = stop
        if self.backend.kind == "llama-server" and slot_id is not None:
            payload["id_slot"] = slot_id
            payload["cache_prompt"] = True
        return payload

    def _agent_from_messages(self, messages: List[Dict]) -> str:
        sys_msg = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        import re
        m = re.search(r"You are agent '([^']+)'", sys_msg)
        return m.group(1) if m else "?"

    def _resolve_response(self, agent: str) -> LLMResponse:
        if not self.scripts.get(agent):
            return LLMResponse(content="(empty)")
        item = self.scripts[agent].pop(0)
        if isinstance(item, str):
            return LLMResponse(content=item)
        if isinstance(item, dict):
            return LLMResponse(content=item.get("content", ""),
                               reasoning=item.get("reasoning", ""))
        raise ValueError(f"unsupported script item: {item!r}")

    async def ensure_probed(self, base_url_override=None) -> BackendInfo:
        return self.backend

    async def complete(self, model, messages, max_tokens=800, temperature=0.7,
                       slot_id=None, stop=None,
                       base_url_override=None, api_key_override=None) -> LLMResponse:
        agent = self._agent_from_messages(messages)
        payload = self._payload(model, messages, max_tokens, temperature,
                                 stream=False, slot_id=slot_id, stop=stop)
        self.payloads.append(payload)
        self.calls.append({"agent": agent, "mode": "complete", "slot_id": slot_id,
                           "base_url_override": base_url_override})
        return self._resolve_response(agent)

    async def stream(self, model, messages, max_tokens=800, temperature=0.7,
                     slot_id=None, stop=None,
                     base_url_override=None, api_key_override=None) -> AsyncIterator[Tuple[str, str]]:
        agent = self._agent_from_messages(messages)
        payload = self._payload(model, messages, max_tokens, temperature,
                                 stream=True, slot_id=slot_id, stop=stop)
        self.payloads.append(payload)
        self.calls.append({"agent": agent, "mode": "stream", "slot_id": slot_id,
                           "base_url_override": base_url_override})
        resp = self._resolve_response(agent)
        # Yield reasoning chunks first, then content chunks (mimics a real thinking-mode stream)
        for kind, text in (("reasoning", resp.reasoning), ("content", resp.content)):
            if not text:
                continue
            words = text.split()
            chunk_size = max(1, len(words) // 4) if words else 1
            for i in range(0, len(words), chunk_size):
                yield (kind, " ".join(words[i:i+chunk_size]) + " ")

    async def aclose(self):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Channel tests
# ─────────────────────────────────────────────────────────────────────────────
class ChannelTests(unittest.IsolatedAsyncioTestCase):
    async def test_broadcast_skips_sender(self):
        ch = Channel()
        a_q = ch.subscribe("alice"); b_q = ch.subscribe("bob"); c_q = ch.subscribe("carol")
        await ch.publish(Message(from_agent="alice", content="hi"))
        self.assertTrue(a_q.empty())
        self.assertEqual(b_q.qsize(), 1)
        self.assertEqual(c_q.qsize(), 1)

    async def test_direct_message_only_to_recipient(self):
        ch = Channel()
        a_q = ch.subscribe("alice"); b_q = ch.subscribe("bob"); c_q = ch.subscribe("carol")
        await ch.publish(Message(from_agent="alice", content="hi", to_agent="bob"))
        self.assertEqual(b_q.qsize(), 1)
        self.assertTrue(a_q.empty())
        self.assertTrue(c_q.empty())

    async def test_history_preserves_order(self):
        ch = Channel()
        ch.subscribe("a"); ch.subscribe("b")
        for i in range(5):
            await ch.publish(Message(from_agent="a", content=str(i)))
        self.assertEqual([m.content for m in ch.history], ["0","1","2","3","4"])

    async def test_transcript_excludes_chunks_by_default(self):
        ch = Channel()
        ch.subscribe("a")
        await ch.publish(Message(from_agent="a", content="full"))
        await ch.publish(Message(from_agent="a", content="part", is_chunk=True))
        self.assertEqual(len(ch.transcript(include_chunks=False)), 1)
        self.assertEqual(len(ch.transcript(include_chunks=True)), 2)

    async def test_seq_increments_per_agent(self):
        ch = Channel()
        ch.subscribe("a"); ch.subscribe("b")
        await ch.publish(Message(from_agent="a", content="1"))
        await ch.publish(Message(from_agent="a", content="2"))
        await ch.publish(Message(from_agent="b", content="1"))
        seqs_a = [m.seq for m in ch.history if m.from_agent == "a"]
        seqs_b = [m.seq for m in ch.history if m.from_agent == "b"]
        self.assertEqual(seqs_a, [1, 2])
        self.assertEqual(seqs_b, [1])

    async def test_total_chars_tracks_cumulative(self):
        ch = Channel()
        ch.subscribe("a")
        await ch.publish(Message(from_agent="a", content="hello"))
        await ch.publish(Message(from_agent="a", content="world", reasoning="thought"))
        self.assertEqual(ch.total_chars, len("hello") + len("world") + len("thought"))


# ─────────────────────────────────────────────────────────────────────────────
# AgentSpec tests
# ─────────────────────────────────────────────────────────────────────────────
class AgentSpecTests(unittest.TestCase):
    def test_name_with_spaces_is_auto_sanitized(self):
        # Real LLM callers emit names like "Critical Thinker" or "agent A".
        # AgentSpec normalizes whitespace to underscores rather than erroring.
        spec = AgentSpec(name="agent with spaces", system_prompt="hi")
        self.assertEqual(spec.name, "agent_with_spaces")

    def test_name_with_disallowed_chars_is_stripped(self):
        # Anything outside [A-Za-z0-9_-] is dropped after sanitization.
        spec = AgentSpec(name="alice/v2!", system_prompt="hi")
        self.assertEqual(spec.name, "alicev2")

    def test_empty_after_sanitization_raises(self):
        # If the name reduces to nothing usable, error explicitly.
        with self.assertRaises(ValueError):
            AgentSpec(name="!!!", system_prompt="hi")

    def test_default_model_uses_env_default(self):
        spec = AgentSpec(name="alice", system_prompt="hi")
        self.assertEqual(spec.model, wot_engine.LLM_DEFAULT_MODEL)

    def test_explicit_model_preserved(self):
        spec = AgentSpec(name="alice", system_prompt="hi", model="custom:v1")
        self.assertEqual(spec.model, "custom:v1")

    def test_thinking_model_auto_bumps_max_tokens(self):
        spec = AgentSpec(name="a", system_prompt="hi", model="deepseek-r1-distill-qwen-1.5b")
        self.assertEqual(spec.max_tokens, 2500)

    def test_explicit_max_tokens_not_clobbered_for_thinking_model(self):
        spec = AgentSpec(name="a", system_prompt="hi",
                          model="deepseek-r1-distill-qwen-1.5b", max_tokens=4096)
        self.assertEqual(spec.max_tokens, 4096)


# ─────────────────────────────────────────────────────────────────────────────
# DONE detection
# ─────────────────────────────────────────────────────────────────────────────
class DoneDetectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_done_on_own_line_marks_agent_done(self):
        scripts = {"alice": ["I am done with my analysis.\nDONE"]}
        engine = WoTEngine()
        engine.client = MockLLMClient(scripts)
        result = await engine.run(
            [AgentSpec(name="alice", system_prompt="...")],
            task="anything", mode="parallel", max_rounds=3,
        )
        self.assertEqual(result["agents_done"], ["alice"])
        self.assertEqual(result["rounds_run"], 1)
        self.assertEqual(result["transcript"][0]["content"], "I am done with my analysis.")

    async def test_done_inline_does_NOT_trigger(self):
        scripts = {"alice": ["I think we are DONE here.", "Final thoughts.\nDONE"]}
        engine = WoTEngine()
        engine.client = MockLLMClient(scripts)
        result = await engine.run(
            [AgentSpec(name="alice", system_prompt="...")],
            task="anything", mode="parallel", max_rounds=3,
        )
        self.assertEqual(result["agents_done"], ["alice"])
        self.assertEqual(result["rounds_run"], 2)


# ─────────────────────────────────────────────────────────────────────────────
# Mode tests — parallel, sequential, queue, streaming
# ─────────────────────────────────────────────────────────────────────────────
class ParallelModeTests(unittest.IsolatedAsyncioTestCase):
    async def test_three_agents_speak_in_parallel_round_one(self):
        scripts = {"a": ["A says hi.\nDONE"], "b": ["B says hi.\nDONE"], "c": ["C says hi.\nDONE"]}
        engine = WoTEngine()
        engine.client = MockLLMClient(scripts)
        result = await engine.run(
            [AgentSpec(name=n, system_prompt="...") for n in ("a","b","c")],
            task="say hi", mode="parallel", max_rounds=2,
        )
        self.assertEqual(set(result["agents_done"]), {"a", "b", "c"})
        self.assertEqual(result["rounds_run"], 1)
        self.assertEqual(len(result["transcript"]), 3)
        self.assertEqual({m["from"] for m in result["transcript"]}, {"a", "b", "c"})


class SequentialModeTests(unittest.IsolatedAsyncioTestCase):
    async def test_round_robin_order(self):
        scripts = {"a": ["A1\nDONE"], "b": ["B1\nDONE"], "c": ["C1\nDONE"]}
        engine = WoTEngine()
        engine.client = MockLLMClient(scripts)
        result = await engine.run(
            [AgentSpec(name=n, system_prompt="...") for n in ("a","b","c")],
            task="hello", mode="sequential", max_rounds=2,
        )
        self.assertEqual([m["from"] for m in result["transcript"]], ["a","b","c"])
        self.assertEqual(set(result["agents_done"]), {"a", "b", "c"})


class QueueModeTests(unittest.IsolatedAsyncioTestCase):
    async def test_round_one_all_eligible(self):
        scripts = {"a": ["A1\nDONE"], "b": ["B1\nDONE"]}
        engine = WoTEngine()
        engine.client = MockLLMClient(scripts)
        result = await engine.run(
            [AgentSpec(name="a", system_prompt="...", interests=["math"]),
             AgentSpec(name="b", system_prompt="...", interests=["code"])],
            task="anything", mode="queue", max_rounds=2,
        )
        self.assertEqual(set(result["agents_done"]), {"a", "b"})


class StreamingModeTests(unittest.IsolatedAsyncioTestCase):
    async def test_streaming_publishes_chunks_before_final(self):
        scripts = {"alice": ["one two three four five six seven eight\nDONE"]}
        engine = WoTEngine()
        engine.client = MockLLMClient(scripts)
        result = await engine.run(
            [AgentSpec(name="alice", system_prompt="...")],
            task="speak", mode="streaming", max_rounds=1,
        )
        chunks_xs = result["transcript_with_chunks"]
        self.assertIsNotNone(chunks_xs)
        chunks = [m for m in chunks_xs if m["is_chunk"]]
        finals = [m for m in chunks_xs if not m["is_chunk"]]
        self.assertGreater(len(chunks), 0)
        self.assertEqual(len(finals), 1)
        self.assertLessEqual(chunks[-1]["ts"], finals[0]["ts"])

    async def test_streaming_separates_reasoning_and_content_chunk_kinds(self):
        scripts = {"a": [{"content": "alpha beta gamma delta\nDONE",
                          "reasoning": "let me think this over"}]}
        engine = WoTEngine()
        engine.client = MockLLMClient(scripts)
        result = await engine.run(
            [AgentSpec(name="a", system_prompt="...")],
            task="x", mode="streaming", max_rounds=1,
        )
        chunks = [m for m in result["transcript_with_chunks"] if m["is_chunk"]]
        kinds = {c["chunk_kind"] for c in chunks}
        self.assertIn("reasoning", kinds)
        self.assertIn("content", kinds)


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning propagation tests
# ─────────────────────────────────────────────────────────────────────────────
class PropagateReasoningTests(unittest.IsolatedAsyncioTestCase):
    async def test_strip_default_hides_reasoning_from_peers(self):
        # Agent A produces reasoning; agent B's peer-render must NOT include <think> blocks
        scripts = {
            "a": [{"content": "answer A.\nDONE", "reasoning": "secret thought"}],
            "b": ["B saw nothing of A's CoT.\nDONE"],
        }
        client = MockLLMClient(scripts)
        engine = WoTEngine()
        engine.client = client
        await engine.run(
            [AgentSpec(name="a", system_prompt="..."),
             AgentSpec(name="b", system_prompt="...")],
            task="x", mode="sequential", max_rounds=1,
            propagate_reasoning="strip",
        )
        # Find B's request: peer message about A should be just content, no reasoning
        b_payload = next(p for p in client.payloads
                          if "agent 'b'" in p["messages"][0]["content"])
        peer_msg = next(m for m in b_payload["messages"] if "[from a" in m.get("content", ""))
        self.assertNotIn("<think>", peer_msg["content"])
        self.assertNotIn("secret thought", peer_msg["content"])
        self.assertIn("answer A.", peer_msg["content"])

    async def test_raw_propagation_includes_reasoning(self):
        scripts = {
            "a": [{"content": "answer A.\nDONE", "reasoning": "shared thought"}],
            "b": ["B saw A's CoT.\nDONE"],
        }
        client = MockLLMClient(scripts)
        engine = WoTEngine()
        engine.client = client
        await engine.run(
            [AgentSpec(name="a", system_prompt="..."),
             AgentSpec(name="b", system_prompt="...")],
            task="x", mode="sequential", max_rounds=1,
            propagate_reasoning="raw",
        )
        b_payload = next(p for p in client.payloads
                          if "agent 'b'" in p["messages"][0]["content"])
        peer_msg = next(m for m in b_payload["messages"] if "[from a" in m.get("content", ""))
        self.assertIn("<think>", peer_msg["content"])
        self.assertIn("shared thought", peer_msg["content"])

    async def test_message_to_dict_includes_reasoning_when_present(self):
        m = Message(from_agent="a", content="answer", reasoning="thought")
        d = m.to_dict()
        self.assertEqual(d["reasoning"], "thought")

    async def test_message_to_dict_omits_reasoning_when_empty(self):
        m = Message(from_agent="a", content="answer")
        self.assertNotIn("reasoning", m.to_dict())


# ─────────────────────────────────────────────────────────────────────────────
# Token budget + slot pinning + backend awareness
# ─────────────────────────────────────────────────────────────────────────────
class TokenBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_token_budget_aborts_run(self):
        # Each turn produces ~50 chars; budget=10 tokens ≈ 40 chars → should stop after round 1
        big = "x " * 30  # 60 chars
        scripts = {"a": [big, big, big, big]}
        engine = WoTEngine()
        engine.client = MockLLMClient(scripts)
        result = await engine.run(
            [AgentSpec(name="a", system_prompt="...")],
            task="x", mode="parallel", max_rounds=10,
            token_budget=10,
        )
        self.assertEqual(result["stop_reason"], "budget")
        self.assertLess(result["rounds_run"], 10)


class MultiBackendMixTests(unittest.IsolatedAsyncioTestCase):
    async def test_per_agent_base_url_override_threads_to_client(self):
        # Two agents with different base_url overrides — verify each call
        # to the client carries the right per-agent target.
        client = MockLLMClient({"a": ["A says hi.\nDONE"], "b": ["B says hi.\nDONE"]})
        engine = WoTEngine()
        engine.client = client
        await engine.run(
            [AgentSpec(name="a", system_prompt="...",
                       base_url="http://127.0.0.1:11434", api_key="ollama"),
             AgentSpec(name="b", system_prompt="...",
                       base_url="https://openrouter.ai/api/v1", api_key="sk-or-test")],
            task="x", mode="parallel", max_rounds=1,
        )
        targets = {c["agent"]: c["base_url_override"] for c in client.calls}
        self.assertEqual(targets["a"], "http://127.0.0.1:11434")
        self.assertEqual(targets["b"], "https://openrouter.ai/api/v1")

    async def test_default_base_url_is_None_when_not_set(self):
        # When AgentSpec has no override, base_url_override should be None
        client = MockLLMClient({"a": ["ok\nDONE"]})
        engine = WoTEngine()
        engine.client = client
        await engine.run(
            [AgentSpec(name="a", system_prompt="...")],
            task="x", mode="parallel", max_rounds=1,
        )
        self.assertIsNone(client.calls[0]["base_url_override"])


class WotChatToolStripsCallerControlFields(unittest.IsolatedAsyncioTestCase):
    async def test_tool_boundary_strips_model_base_url_api_key(self):
        # wot_chat_tool boundary should strip outer-Hermes-supplied backend
        # control fields (Hermes hallucinates them). Direct Python callers
        # using AgentSpec(base_url=..., api_key=...) still work.
        from tools.wot_engine import wot_chat_tool, _LLMClient

        captured_specs = []
        from unittest.mock import patch

        # Stub asyncio.run path: just inspect specs after sanitize
        original_async_run = None

        # Easier: capture by patching AgentSpec __init__ to save args
        from tools import wot_engine as we
        original_init = we.AgentSpec.__post_init__
        seen = []

        def capture_init(self):
            original_init(self)
            seen.append({"name": self.name, "model": self.model,
                         "base_url": self.base_url, "api_key": self.api_key})

        we.AgentSpec.__post_init__ = capture_init
        try:
            # Call should ignore the caller's model/base_url/api_key
            try:
                wot_chat_tool(
                    agents=[{"name": "alpha", "system_prompt": "hi",
                             "model": "gpt-4o-hallucinated",
                             "base_url": "https://evil.example.com",
                             "api_key": "sk-stolen"}],
                    task="x", mode="parallel", max_rounds=1,
                )
            except Exception:
                pass  # We don't care if the network call fails — we just want the spec inspection
        finally:
            we.AgentSpec.__post_init__ = original_init

        self.assertGreaterEqual(len(seen), 1)
        spec = seen[0]
        # base_url + api_key stripped → both None
        self.assertIsNone(spec["base_url"], "base_url should have been stripped at tool boundary")
        self.assertIsNone(spec["api_key"], "api_key should have been stripped at tool boundary")
        # model stripped → falls back to LLM_DEFAULT_MODEL (env-driven), NOT 'gpt-4o-hallucinated'
        self.assertNotEqual(spec["model"], "gpt-4o-hallucinated",
                              "model should have been stripped at tool boundary")


class SlotPinningTests(unittest.IsolatedAsyncioTestCase):
    async def test_llama_server_passes_id_slot_per_agent(self):
        client = MockLLMClient({"a": ["a1\nDONE"], "b": ["b1\nDONE"]},
                                backend_kind="llama-server")
        engine = WoTEngine()
        engine.client = client
        await engine.run(
            [AgentSpec(name="a", system_prompt="..."),
             AgentSpec(name="b", system_prompt="...")],
            task="x", mode="parallel", max_rounds=1,
        )
        slot_ids = sorted(c["slot_id"] for c in client.calls if c["slot_id"] is not None)
        self.assertEqual(slot_ids, [0, 1])
        # Payloads should include cache_prompt: True for llama-server
        self.assertTrue(all(p.get("cache_prompt") is True for p in client.payloads))

    async def test_other_backends_do_not_pass_id_slot(self):
        client = MockLLMClient({"a": ["a1\nDONE"]}, backend_kind="ollama")
        engine = WoTEngine()
        engine.client = client
        await engine.run(
            [AgentSpec(name="a", system_prompt="...")],
            task="x", mode="parallel", max_rounds=1,
        )
        self.assertTrue(all(c["slot_id"] is None for c in client.calls))
        self.assertTrue(all("id_slot" not in p for p in client.payloads))


# ─────────────────────────────────────────────────────────────────────────────
# Qwen3.5/3.6 chat_template_kwargs auto-injection
# ─────────────────────────────────────────────────────────────────────────────
class Qwen35DetectionTests(unittest.IsolatedAsyncioTestCase):
    async def _capture_payload(self, model_name: str) -> dict:
        client = MockLLMClient({"a": ["hi\nDONE"]})
        engine = WoTEngine()
        engine.client = client
        await engine.run(
            [AgentSpec(name="a", system_prompt="...", model=model_name)],
            task="x", mode="parallel", max_rounds=1,
        )
        return client.payloads[0]

    async def test_qwen35_triggers_enable_thinking_false(self):
        payload = await self._capture_payload("qwen3.5-9b-q4_k_m")
        self.assertEqual(payload.get("chat_template_kwargs"), {"enable_thinking": False})

    async def test_qwen36_triggers_enable_thinking_false(self):
        payload = await self._capture_payload("qwen3.6-27b")
        self.assertEqual(payload.get("chat_template_kwargs"), {"enable_thinking": False})

    async def test_qwen3_does_NOT_trigger(self):
        payload = await self._capture_payload("qwen3:8b")
        self.assertNotIn("chat_template_kwargs", payload)

    async def test_llama_does_NOT_trigger(self):
        payload = await self._capture_payload("llama3.1:8b")
        self.assertNotIn("chat_template_kwargs", payload)


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────
class EdgeCaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_max_rounds_hit_without_done(self):
        scripts = {"a": ["round 1", "round 2", "round 3"]}
        engine = WoTEngine()
        engine.client = MockLLMClient(scripts)
        result = await engine.run(
            [AgentSpec(name="a", system_prompt="...")],
            task="x", mode="parallel", max_rounds=3,
        )
        self.assertEqual(result["rounds_run"], 3)
        self.assertEqual(result["agents_done"], [])
        self.assertEqual(result["stop_reason"], "max_rounds")

    async def test_all_done_short_circuits(self):
        scripts = {"a": ["one\nDONE"], "b": ["two\nDONE"]}
        engine = WoTEngine()
        engine.client = MockLLMClient(scripts)
        result = await engine.run(
            [AgentSpec(name="a", system_prompt="..."),
             AgentSpec(name="b", system_prompt="...")],
            task="x", mode="parallel", max_rounds=10,
        )
        self.assertEqual(result["rounds_run"], 1)
        self.assertEqual(set(result["agents_done"]), {"a", "b"})
        self.assertEqual(result["stop_reason"], "all_done")

    async def test_invalid_mode_raises(self):
        engine = WoTEngine()
        engine.client = MockLLMClient({})
        with self.assertRaises(ValueError):
            await engine.run([AgentSpec(name="a", system_prompt="...")],
                              task="x", mode="banana", max_rounds=1)

    async def test_invalid_propagate_mode_raises(self):
        engine = WoTEngine()
        engine.client = MockLLMClient({})
        with self.assertRaises(ValueError):
            await engine.run([AgentSpec(name="a", system_prompt="...")],
                              task="x", mode="parallel", max_rounds=1,
                              propagate_reasoning="banana")

    async def test_result_includes_backend_info(self):
        engine = WoTEngine()
        engine.client = MockLLMClient({"a": ["hi\nDONE"]}, backend_kind="vllm")
        result = await engine.run(
            [AgentSpec(name="a", system_prompt="...")],
            task="x", mode="parallel", max_rounds=1,
        )
        self.assertEqual(result["backend"]["kind"], "vllm")
        self.assertIn("endpoint_source", result["backend"])


# ─────────────────────────────────────────────────────────────────────────────
# Option B: routing (@name / #tags) + Hermes endpoint resolution
# ─────────────────────────────────────────────────────────────────────────────
class MessageRoutingParseTests(unittest.TestCase):
    def test_at_mention_resolves_known_agent(self):
        to_agent, tags = wot_engine.parse_message_routing(
            "@bob: please review this", known_agents={"alice", "bob"}
        )
        self.assertEqual(to_agent, "bob")
        self.assertEqual(tags, [])

    def test_at_mention_case_insensitive_maps_to_canonical(self):
        to_agent, tags = wot_engine.parse_message_routing(
            "Hey @Bob check this", known_agents={"alice", "bob"}
        )
        self.assertEqual(to_agent, "bob")

    def test_unknown_mention_ignored_when_known_set(self):
        to_agent, tags = wot_engine.parse_message_routing(
            "@eve: secret", known_agents={"alice", "bob"}
        )
        self.assertIsNone(to_agent)

    def test_hashtags_extracted(self):
        to_agent, tags = wot_engine.parse_message_routing(
            "Discuss #math and #code please", known_agents={"a"}
        )
        self.assertIsNone(to_agent)
        self.assertEqual(tags, ["math", "code"])

    def test_combined_dm_and_tags(self):
        to_agent, tags = wot_engine.parse_message_routing(
            "@alice #security please audit", known_agents={"alice", "bob"}
        )
        self.assertEqual(to_agent, "alice")
        self.assertEqual(tags, ["security"])


class DirectMentionRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_dm_only_lands_in_target_inbox(self):
        scripts = {
            "alice": ["@bob: only for you\nDONE"],
            "bob": ["got it\nDONE"],
            "carol": ["untouched\nDONE"],
        }
        engine = WoTEngine()
        engine.client = MockLLMClient(scripts)
        result = await engine.run(
            [AgentSpec(name=n, system_prompt="...") for n in ("alice", "bob", "carol")],
            task="dm", mode="sequential", max_rounds=1,
        )
        dm = next(m for m in result["transcript"] if m["from"] == "alice")
        self.assertEqual(dm.get("to"), "bob")
        # Carol's payload must not contain alice's DM body (sequential: alice first)
        carol_payloads = [
            p for p in engine.client.payloads
            if "agent 'carol'" in p["messages"][0]["content"]
        ]
        self.assertTrue(carol_payloads)
        flat = " ".join(m.get("content", "") for m in carol_payloads[0]["messages"])
        self.assertNotIn("only for you", flat)


class QueueInterestsTests(unittest.IsolatedAsyncioTestCase):
    async def test_interest_agent_wakes_on_matching_tag(self):
        # Round 1: broadcaster emits #math; math_agent wakes round 2; code_agent stays quiet
        scripts = {
            "broadcaster": ["Looking at #math problem\nDONE", "still done"],
            "math_agent": ["I handle math\nDONE", "noop"],
            "code_agent": ["I only care about code", "still idle"],
        }
        engine = WoTEngine()
        engine.client = MockLLMClient(scripts)
        result = await engine.run(
            [
                AgentSpec(name="broadcaster", system_prompt="..."),
                AgentSpec(name="math_agent", system_prompt="...", interests=["math"]),
                AgentSpec(name="code_agent", system_prompt="...", interests=["code"]),
            ],
            task="queue", mode="queue", max_rounds=2,
        )
        speakers = [m["from"] for m in result["transcript"]]
        # broadcaster + both agents speak round 1; round 2 only math_agent (and maybe broadcaster if not DONE)
        self.assertIn("math_agent", speakers)
        # After round 1 DONE, code_agent with interests=["code"] should not get a round-2 turn
        # Count turns via mock call log
        code_calls = [c for c in engine.client.calls if c["agent"] == "code_agent"]
        math_calls = [c for c in engine.client.calls if c["agent"] == "math_agent"]
        self.assertEqual(len(code_calls), 1, "code_agent should only speak round 1")
        self.assertGreaterEqual(len(math_calls), 1)

    async def test_dm_wakes_interest_filtered_agent(self):
        scripts = {
            "alice": ["@bob: direct wake\nDONE"],
            # bob must NOT DONE in round 1 so he can be woken by the DM in round 2
            "bob": ["listening", "woken by DM\nDONE"],
            "carol": ["I am done too\nDONE"],
        }
        engine = WoTEngine()
        engine.client = MockLLMClient(scripts)
        result = await engine.run(
            [
                AgentSpec(name="alice", system_prompt="..."),
                AgentSpec(name="bob", system_prompt="...", interests=["security"]),
                AgentSpec(name="carol", system_prompt="...", interests=["other"]),
            ],
            task="dm-wake", mode="queue", max_rounds=2,
        )
        bob_calls = [c for c in engine.client.calls if c["agent"] == "bob"]
        # bob: round 1 always, round 2 because of DM
        self.assertGreaterEqual(len(bob_calls), 2)
        self.assertIn("bob", result["agents_done"])


class HermesEndpointResolutionTests(unittest.TestCase):
    def test_llm_base_url_env_wins(self):
        with patch.dict(os.environ, {
            "LLM_BASE_URL": "http://lab.example:8080",
            "LLM_API_KEY": "sk-lab",
            "LLM_DEFAULT_MODEL": "lab-model",
        }, clear=False):
            base, key, model, meta = wot_engine.resolve_hermes_endpoint()
        self.assertEqual(base, "http://lab.example:8080")
        self.assertEqual(key, "sk-lab")
        self.assertEqual(model, "lab-model")
        self.assertEqual(meta["source"], "env:LLM_BASE_URL")

    def test_engine_records_endpoint_meta(self):
        with patch.dict(os.environ, {
            "LLM_BASE_URL": "http://only-for-test:9",
            "LLM_API_KEY": "k",
            "LLM_DEFAULT_MODEL": "m",
        }, clear=False):
            engine = WoTEngine()
        self.assertEqual(engine.endpoint_meta.get("source"), "env:LLM_BASE_URL")
        self.assertEqual(engine.default_model, "m")


if __name__ == "__main__":
    unittest.main(verbosity=2)
