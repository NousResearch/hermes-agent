#!/usr/bin/env python3
"""Trajectory Compressor — post-process agent trajectories into a token budget.

Strategy: protect the head (system, human, first gpt, first tool) and the last N
turns; from the middle, summarize only as many turns as needed (never splitting a
<tool_call>/<tool_response> pair) and replace them with one human summary turn.

Usage:
    python trajectory_compressor.py --input=data/my_run            # directory
    python trajectory_compressor.py --input=data/trajectories.jsonl --sample_percent=15
    python trajectory_compressor.py --input=data/trajectories.jsonl --output=out.jsonl --target_max_tokens=16000
"""

import json
import os
import random
import shutil
import tempfile
import time
import yaml
import logging
import asyncio
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import asdict, dataclass, field
from datetime import datetime

from utils import base_url_host_matches, base_url_hostname
import fire
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console
from hermes_constants import OPENROUTER_BASE_URL, get_hermes_home
from agent.retry_utils import jittered_backoff
from hermes_cli.env_loader import load_hermes_dotenv

# Load .env from HERMES_HOME first, then project root as a dev fallback.
load_hermes_dotenv(hermes_home=get_hermes_home(), project_env=Path(__file__).parent / ".env")


def _response_finish_reason(response: Any) -> str:
    """Lowercased ``choices[0].finish_reason`` of a dict/object response, ``""`` if absent.

    Local copy of ``agent.context_compressor._response_finish_reason``: this
    standalone CLI deliberately avoids importing the heavy context compressor.
    """
    try:
        choices = (response.get("choices") if isinstance(response, dict) else getattr(response, "choices", None)) or []
        first = choices[0] if choices else None
        reason = first.get("finish_reason") if isinstance(first, dict) else getattr(first, "finish_reason", None)
        return str(reason).strip().lower() if reason else ""
    except Exception:
        return ""


def _effective_temperature_for_model(model: str, requested_temperature: Optional[float], base_url: Optional[str] = None) -> Optional[float]:
    """Apply fixed model temperature contracts to direct client calls.

    Returns ``None`` when the model manages temperature server-side (Kimi);
    callers must omit the ``temperature`` kwarg entirely in that case.
    Shared with ``mini_swe_runner`` (which passes ``requested_temperature=None``).
    """
    try:
        from agent.auxiliary_client import _fixed_temperature_for_model, OMIT_TEMPERATURE
    except Exception:
        return requested_temperature
    fixed_temperature = _fixed_temperature_for_model(model, base_url)
    if fixed_temperature is OMIT_TEMPERATURE:
        return None  # caller must omit temperature
    return requested_temperature if fixed_temperature is None else fixed_temperature


def _load_jsonl(path: Path, on_error: Optional[Callable[[int, json.JSONDecodeError], None]] = None, start: int = 0) -> List[Tuple[int, Any]]:
    """Return ``(line_num, entry)`` for each non-blank line; bad lines go to ``on_error``."""
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start):
            if not line.strip():
                continue
            try:
                entries.append((line_num, json.loads(line)))
            except json.JSONDecodeError as e:
                if on_error is not None:
                    on_error(line_num, e)
    return entries


def _write_jsonl(path: Path, entries) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


# YAML section -> keys; "yaml_key:attr" when the config attribute name differs.
_YAML_SECTIONS: Dict[str, Tuple[str, ...]] = {
    "tokenizer": ("name:tokenizer_name", "trust_remote_code"),
    "compression": ("target_max_tokens", "summary_target_tokens"),
    "protected_turns": ("first_system:protect_first_system", "first_human:protect_first_human",
                        "first_gpt:protect_first_gpt", "first_tool:protect_first_tool", "last_n_turns:protect_last_n_turns"),
    "summarization": ("model:summarization_model", "base_url", "api_key_env", "temperature", "max_retries", "retry_delay"),
    "output": ("add_summary_notice", "summary_notice_text", "output_suffix"),
    "processing": ("num_workers", "max_concurrent_requests", "skip_under_target", "save_over_limit"),
    "metrics": ("enabled:metrics_enabled", "per_trajectory:metrics_per_trajectory", "output_file:metrics_output_file"),
}


@dataclass
class CompressionConfig:
    """Configuration for trajectory compression (tokenizer / targets / protected turns / summarizer / output / processing / metrics)."""
    tokenizer_name: str = "moonshotai/Kimi-K2-Thinking"
    trust_remote_code: bool = True
    target_max_tokens: int = 15250
    summary_target_tokens: int = 750
    protect_first_system: bool = True
    protect_first_human: bool = True
    protect_first_gpt: bool = True
    protect_first_tool: bool = True
    protect_last_n_turns: int = 4
    summarization_model: str = "google/gemini-3-flash-preview"
    base_url: str = OPENROUTER_BASE_URL
    api_key_env: str = "OPENROUTER_API_KEY"
    temperature: float = 0.3
    max_retries: int = 3
    retry_delay: int = 2
    add_summary_notice: bool = True
    summary_notice_text: str = "\n\nSome of your previous tool responses may be summarized to preserve context."
    output_suffix: str = "_compressed"
    num_workers: int = 4
    max_concurrent_requests: int = 50  # Max concurrent API calls for summarization
    skip_under_target: bool = True
    save_over_limit: bool = True
    per_trajectory_timeout: int = 300  # seconds (default: 5 min)
    metrics_enabled: bool = True
    metrics_per_trajectory: bool = True
    metrics_output_file: str = "compression_metrics.json"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "CompressionConfig":
        """Load configuration from YAML file (missing keys keep the defaults)."""
        with open(yaml_path, 'r', encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        config = cls()
        for section, keys in _YAML_SECTIONS.items():
            for key in keys if section in data else ():
                yaml_key, _, attr = key.partition(":")
                attr = attr or yaml_key
                value = data[section].get(yaml_key, getattr(config, attr))
                if attr == "base_url":
                    value = value or config.base_url  # ``base_url: null`` keeps the default
                setattr(config, attr, value)
        return config


@dataclass
class TrajectoryMetrics:
    """Metrics for a single trajectory compression."""
    original_tokens: int = 0
    compressed_tokens: int = 0
    tokens_saved: int = 0
    compression_ratio: float = 1.0
    original_turns: int = 0
    compressed_turns: int = 0
    turns_removed: int = 0
    turns_compressed_start_idx: int = -1
    turns_compressed_end_idx: int = -1
    turns_in_compressed_region: int = 0
    was_compressed: bool = False
    still_over_limit: bool = False
    skipped_under_target: bool = False
    summarization_api_calls: int = 0
    summarization_errors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["compression_ratio"] = round(self.compression_ratio, 4)
        region = {"start_idx": d.pop("turns_compressed_start_idx"), "end_idx": d.pop("turns_compressed_end_idx"),
                  "turns_count": d.pop("turns_in_compressed_region")}
        items = list(d.items())
        items.insert(7, ("compression_region", region))  # after turns_removed: historical key order
        return dict(items)


def _mean(values, default):
    return sum(values) / len(values) if values else default


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all trajectories."""
    total_trajectories: int = 0
    trajectories_compressed: int = 0
    trajectories_skipped_under_target: int = 0
    trajectories_still_over_limit: int = 0
    trajectories_failed: int = 0
    total_tokens_before: int = 0
    total_tokens_after: int = 0
    total_tokens_saved: int = 0
    total_turns_before: int = 0
    total_turns_after: int = 0
    total_turns_removed: int = 0
    total_summarization_calls: int = 0
    total_summarization_errors: int = 0
    compression_ratios: List[float] = field(default_factory=list)
    tokens_saved_list: List[int] = field(default_factory=list)
    turns_removed_list: List[int] = field(default_factory=list)
    processing_start_time: str = ""
    processing_end_time: str = ""
    processing_duration_seconds: float = 0.0

    def add_trajectory_metrics(self, metrics: TrajectoryMetrics):
        """Add a trajectory's metrics to the aggregate."""
        self.total_trajectories += 1
        self.total_tokens_before += metrics.original_tokens
        self.total_tokens_after += metrics.compressed_tokens
        self.total_tokens_saved += metrics.tokens_saved
        self.total_turns_before += metrics.original_turns
        self.total_turns_after += metrics.compressed_turns
        self.total_turns_removed += metrics.turns_removed
        self.total_summarization_calls += metrics.summarization_api_calls
        self.total_summarization_errors += metrics.summarization_errors
        if metrics.was_compressed:
            self.trajectories_compressed += 1
            self.compression_ratios.append(metrics.compression_ratio)
            self.tokens_saved_list.append(metrics.tokens_saved)
            self.turns_removed_list.append(metrics.turns_removed)
        self.trajectories_skipped_under_target += bool(metrics.skipped_under_target)
        self.trajectories_still_over_limit += bool(metrics.still_over_limit)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {"total_trajectories": self.total_trajectories, "trajectories_compressed": self.trajectories_compressed,
                        "trajectories_skipped_under_target": self.trajectories_skipped_under_target,
                        "trajectories_still_over_limit": self.trajectories_still_over_limit, "trajectories_failed": self.trajectories_failed,
                        "compression_rate": round(self.trajectories_compressed / max(self.total_trajectories, 1), 4)},
            "tokens": {"total_before": self.total_tokens_before, "total_after": self.total_tokens_after, "total_saved": self.total_tokens_saved,
                       "overall_compression_ratio": round(self.total_tokens_after / max(self.total_tokens_before, 1), 4)},
            "turns": {"total_before": self.total_turns_before, "total_after": self.total_turns_after, "total_removed": self.total_turns_removed},
            "averages": {"avg_compression_ratio": round(_mean(self.compression_ratios, 1.0), 4),
                         "avg_tokens_saved_per_compressed": round(_mean(self.tokens_saved_list, 0), 1),
                         "avg_turns_removed_per_compressed": round(_mean(self.turns_removed_list, 0), 2)},
            "summarization": {"total_api_calls": self.total_summarization_calls, "total_errors": self.total_summarization_errors,
                              "success_rate": round(1 - (self.total_summarization_errors / max(self.total_summarization_calls, 1)), 4)},
            "processing": {"start_time": self.processing_start_time, "end_time": self.processing_end_time,
                           "duration_seconds": round(self.processing_duration_seconds, 2)},
        }


# Ordered (hostname, provider) table for _detect_provider (codex is matched separately).
_PROVIDER_HOSTS: Tuple[Tuple[str, str], ...] = (
    ("openrouter.ai", "openrouter"), ("nousresearch.com", "nous"), ("z.ai", "zai"), ("moonshot.ai", "kimi-coding"),
    ("moonshot.cn", "kimi-coding"), ("api.kimi.com", "kimi-coding"), ("arcee.ai", "arcee"), ("minimaxi.com", "minimax-cn"),
    ("minimax.io", "minimax"),
)

_SUMMARY_FALLBACK = "[CONTEXT SUMMARY]: [Summary generation failed - previous turns contained tool calls and responses that have been compressed to save context space.]"
_STATUS_FMT = "[dim]✅ {compressed} compressed | ⏭️ {skipped} skipped | ⏱️ {timeouts} timeout | 🔄 {api_calls} API calls | ⚡ {in_flight} in-flight[/dim]"


@dataclass
class _RunProgress:
    """Shared counters + rich progress handles for one directory run."""
    progress: Any
    main_task: Any
    status_task: Any
    lock: asyncio.Lock
    semaphore: asyncio.Semaphore
    compressed: int = 0
    skipped: int = 0
    api_calls: int = 0
    in_flight: int = 0
    timeouts: int = 0

    def finish(self, update_status: bool = True) -> None:
        """Retire one in-flight entry and advance the bar (caller holds ``lock``)."""
        self.in_flight -= 1
        self.progress.advance(self.main_task)
        if update_status:
            self.progress.update(self.status_task, description=_STATUS_FMT.format(
                compressed=self.compressed, skipped=self.skipped, timeouts=self.timeouts,
                api_calls=self.api_calls, in_flight=self.in_flight))


class TrajectoryCompressor:
    """Compresses agent trajectories to fit within a target token budget.

    Keeps protected head/tail turns, summarizes only as much of the middle as
    needed into one human summary turn, and keeps the remaining middle intact.
    """

    def __init__(self, config: CompressionConfig):
        self.config = config
        self.aggregate_metrics = AggregateMetrics()
        self._init_tokenizer()
        self._init_summarizer()
        self.logger = logging.getLogger(__name__)

    def _init_tokenizer(self):
        """Initialize HuggingFace tokenizer for token counting."""
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name, trust_remote_code=self.config.trust_remote_code)
            print(f"✅ Loaded tokenizer: {self.config.tokenizer_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer '{self.config.tokenizer_name}': {e}")

    def _init_summarizer(self):
        """Route summarization through call_llm for known providers, else a raw client."""
        provider = self._detect_provider()
        self._use_call_llm = bool(provider)
        if provider:
            self._llm_provider = provider
            from agent.auxiliary_client import resolve_provider_client
            client, _ = resolve_provider_client(provider, model=self.config.summarization_model)
            if client is None:
                raise RuntimeError(f"Provider '{provider}' is not configured. Check your API key or run: hermes setup")
            self.client = self.async_client = None  # Not used directly
        else:
            # Custom endpoint — use config's raw base_url + api_key_env
            api_key = os.getenv(self.config.api_key_env)
            if not api_key:
                raise RuntimeError(f"Missing API key. Set {self.config.api_key_env} environment variable.")
            from openai import OpenAI
            from agent.auxiliary_client import _to_openai_base_url
            self.client = OpenAI(api_key=api_key, base_url=_to_openai_base_url(self.config.base_url))
            # AsyncOpenAI is created lazily in _get_async_client() so it binds to the current event
            # loop — each process_directory() runs its own asyncio.run(); a shared client would hit
            # "Event loop is closed".
            self.async_client = None
            self._async_client_api_key = api_key
        print(f"✅ Initialized summarizer client: {self.config.summarization_model}")
        print(f"   Max concurrent requests: {self.config.max_concurrent_requests}")

    def _get_async_client(self):
        """Return a fresh AsyncOpenAI client bound to the running event loop."""
        from openai import AsyncOpenAI
        from agent.auxiliary_client import _to_openai_base_url
        self.async_client = AsyncOpenAI(api_key=self._async_client_api_key, base_url=_to_openai_base_url(self.config.base_url))
        return self.async_client

    def _detect_provider(self) -> str:
        """Provider name for the configured base_url, or ``""`` when unknown."""
        url = self.config.base_url or ""
        if base_url_hostname(url) == "chatgpt.com" and "/backend-api/codex" in url.lower():
            return "codex"
        return next((provider for host, provider in _PROVIDER_HOSTS if base_url_host_matches(url, host)), "")

    def count_tokens(self, text: str) -> int:
        """Token count via the configured tokenizer; falls back to len//4."""
        if not text:
            return 0
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            return len(text) // 4

    def count_trajectory_tokens(self, trajectory: List[Dict[str, str]]) -> int:
        return sum(self.count_turn_tokens(trajectory))

    def count_turn_tokens(self, trajectory: List[Dict[str, str]]) -> List[int]:
        return [self.count_tokens(turn.get("value", "")) for turn in trajectory]

    def _find_protected_indices(self, trajectory: List[Dict[str, str]]) -> Tuple[set, int, int]:
        """Return ``(protected_set, compressible_start, compressible_end)``."""
        n = len(trajectory)
        first_seen: Dict[str, int] = {}
        for i, turn in enumerate(trajectory):
            first_seen.setdefault(turn.get("from", ""), i)
        protected = {first_seen[role] for role in ("system", "human", "gpt", "tool")
                     if getattr(self.config, f"protect_first_{role}") and role in first_seen}
        protected.update(range(max(0, n - self.config.protect_last_n_turns), n))
        # Compressible region: after the last protected head turn, before the first tail turn.
        head_protected = [i for i in protected if i < n // 2]
        tail_protected = [i for i in protected if i >= n // 2]
        return protected, max(head_protected) + 1 if head_protected else 0, min(tail_protected) if tail_protected else n

    @staticmethod
    def _snap_boundary(trajectory: List[Dict[str, str]], idx: int, min_idx: int, max_idx: int) -> int:
        """Move a boundary onto the nearest turn boundary within ``[min_idx, max_idx]`` that does not
        split a gpt <tool_call>/tool <tool_response> pair.

        A ``tool`` turn always directly follows the ``gpt`` turn it answers, so a boundary landing *on*
        a tool turn cuts the pair; only the end of the trajectory or a non-``tool`` turn is clean.
        Forward is preferred (folds an orphaned ``tool`` turn into the region that holds its ``gpt``
        turn); backward only when nothing clean lies ahead.
        """
        def clean(i: int) -> bool:
            return i >= len(trajectory) or trajectory[i].get("from") != "tool"
        forward = idx
        while forward < max_idx and not clean(forward):
            forward += 1
        if clean(forward):
            return forward
        backward = idx
        while backward > min_idx and not clean(backward):
            backward -= 1
        return backward

    def _extract_turn_content_for_summary(self, trajectory: List[Dict[str, str]], start: int, end: int) -> str:
        """Format turns ``[start, end)`` for the summarization prompt (long values truncated)."""
        parts = []
        for i in range(start, end):
            turn = trajectory[i]
            value = turn.get("value", "")
            if len(value) > 3000:
                value = value[:1500] + "\n...[truncated]...\n" + value[-500:]
            parts.append(f"[Turn {i} - {turn.get('from', 'unknown').upper()}]:\n{value}")
        return "\n\n".join(parts)

    def _summary_prompt(self, content: str) -> str:
        return f"""Summarize the following agent conversation turns concisely. This summary will replace these turns in the conversation history.

Write the summary from a neutral perspective describing what the assistant did and learned. Include:
1. What actions the assistant took (tool calls, searches, file operations)
2. Key information or results obtained
3. Any important decisions or findings
4. Relevant data, file names, values, or outputs

Keep the summary factual and informative. Target approximately {self.config.summary_target_tokens} tokens.

---
TURNS TO SUMMARIZE:
{content}
---

Write only the summary, starting with "[CONTEXT SUMMARY]:" prefix."""

    def _summary_request(self, prompt: str) -> Tuple[Optional[float], Dict[str, Any]]:
        """Return ``(temperature, create-kwargs)``; temperature None means omit it."""
        cfg = self.config
        temperature = _effective_temperature_for_model(cfg.summarization_model, cfg.temperature, cfg.base_url)
        kwargs = {"model": cfg.summarization_model, "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": cfg.summary_target_tokens * 2}
        if not getattr(self, '_use_call_llm', False) and temperature is not None:
            kwargs["temperature"] = temperature
        return temperature, kwargs

    def _finish_summary(self, response: Any) -> str:
        """Extract the summary text with the ``[CONTEXT SUMMARY]:`` prefix exactly once; a ``length`` stop is a failure."""
        if _response_finish_reason(response) == "length":
            # Storing a truncated summary silently corrupts the trajectory's memory, so raise and
            # let the retry/backoff loop handle it.
            raise RuntimeError("trajectory summarization hit the output token cap (finish_reason=length); summary is incomplete")
        content = response.choices[0].message.content
        text = (content if isinstance(content, str) else str(content) if content else "").strip()
        if text.startswith("[CONTEXT SUMMARY]:"):
            return text
        return "[CONTEXT SUMMARY]:" if not text else f"[CONTEXT SUMMARY]: {text}"

    def _summary_attempt_failed(self, metrics: TrajectoryMetrics, attempt: int, exc: Exception) -> Optional[float]:
        """Record a failed attempt; return the backoff delay, or None on the last attempt."""
        metrics.summarization_errors += 1
        self.logger.warning("Summarization attempt %d failed: %s", attempt + 1, exc)
        if attempt < self.config.max_retries - 1:
            return jittered_backoff(attempt + 1, base_delay=self.config.retry_delay, max_delay=30.0)
        return None

    def _generate_summary(self, content: str, metrics: TrajectoryMetrics) -> str:
        """Summarize ``content`` with retries; returns a fallback summary after the last failure."""
        prompt = self._summary_prompt(content)
        for attempt in range(self.config.max_retries):
            try:
                metrics.summarization_api_calls += 1
                temperature, kwargs = self._summary_request(prompt)
                if getattr(self, '_use_call_llm', False):
                    from agent.auxiliary_client import call_llm
                    response = call_llm(provider=self._llm_provider, temperature=temperature, **kwargs)
                else:
                    response = self.client.chat.completions.create(**kwargs)
                return self._finish_summary(response)
            except Exception as e:
                delay = self._summary_attempt_failed(metrics, attempt, e)
                if delay is None:
                    return _SUMMARY_FALLBACK
                time.sleep(delay)

    async def _generate_summary_async(self, content: str, metrics: TrajectoryMetrics) -> str:
        """Async twin of ``_generate_summary``."""
        prompt = self._summary_prompt(content)
        for attempt in range(self.config.max_retries):
            try:
                metrics.summarization_api_calls += 1
                temperature, kwargs = self._summary_request(prompt)
                if getattr(self, '_use_call_llm', False):
                    from agent.auxiliary_client import async_call_llm
                    response = await async_call_llm(provider=self._llm_provider, temperature=temperature, **kwargs)
                else:
                    response = await self._get_async_client().chat.completions.create(**kwargs)
                return self._finish_summary(response)
            except Exception as e:
                delay = self._summary_attempt_failed(metrics, attempt, e)
                if delay is None:
                    return _SUMMARY_FALLBACK
                await asyncio.sleep(delay)

    def _plan_compression(self, trajectory: List[Dict[str, str]], metrics: TrajectoryMetrics) -> Optional[Tuple[int, int]]:
        """Choose the ``[start, until)`` region to summarize, or None if nothing can be.

        Fills the pre-compression metrics either way. Accumulates turns from the
        start of the compressible middle until the savings cover the overage plus
        the summary itself, then snaps both boundaries off ``tool`` turns.
        """
        cfg = self.config
        turn_tokens = self.count_turn_tokens(trajectory)
        total_tokens = sum(turn_tokens)
        metrics.original_turns = metrics.compressed_turns = len(trajectory)
        metrics.original_tokens = metrics.compressed_tokens = total_tokens
        if total_tokens <= cfg.target_max_tokens:
            metrics.skipped_under_target = True
            return None
        metrics.still_over_limit = True
        _, start, end = self._find_protected_indices(trajectory)
        # Never *start* on an orphaned <tool_response> whose <tool_call> is in the protected head.
        start = self._snap_boundary(trajectory, start, start, end)
        if start >= end:
            return None
        # Replacing N turns with one summary saves sum(N) - summary_target_tokens.
        target_tokens_to_compress = total_tokens - cfg.target_max_tokens + cfg.summary_target_tokens
        accumulated = 0
        until = start
        for i in range(start, end):
            accumulated += turn_tokens[i]
            until = i + 1
            if accumulated >= target_tokens_to_compress:
                break
        if accumulated < target_tokens_to_compress and until < end:
            until = end
        # The remainder is kept verbatim, so a tail boundary on a tool turn would orphan a marker.
        until = self._snap_boundary(trajectory, until, start, end)
        # A region no larger than the summary replacing it cannot shrink the trajectory.
        if until <= start or sum(turn_tokens[start:until]) <= cfg.summary_target_tokens:
            return None
        metrics.turns_compressed_start_idx, metrics.turns_compressed_end_idx = start, until
        metrics.turns_in_compressed_region = until - start
        return start, until

    def _assemble_compressed(self, trajectory: List[Dict[str, str]], start: int, until: int, summary: str,
                             metrics: TrajectoryMetrics) -> List[Dict[str, str]]:
        """Head (with summary notice on system) + summary human turn + verbatim tail; finalize metrics."""
        compressed = []
        for turn in trajectory[:start]:
            turn = turn.copy()
            if turn.get("from") == "system" and self.config.add_summary_notice:
                turn["value"] = turn["value"] + self.config.summary_notice_text
            compressed.append(turn)
        compressed.append({"from": "human", "value": summary})
        compressed.extend(turn.copy() for turn in trajectory[until:])
        metrics.compressed_turns = len(compressed)
        metrics.compressed_tokens = self.count_trajectory_tokens(compressed)
        metrics.turns_removed = metrics.original_turns - metrics.compressed_turns
        metrics.tokens_saved = metrics.original_tokens - metrics.compressed_tokens
        metrics.compression_ratio = metrics.compressed_tokens / max(metrics.original_tokens, 1)
        metrics.was_compressed = True
        metrics.still_over_limit = metrics.compressed_tokens > self.config.target_max_tokens
        return compressed

    def compress_trajectory(self, trajectory: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], TrajectoryMetrics]:
        """Compress one trajectory into the target budget; returns ``(trajectory, metrics)``."""
        metrics = TrajectoryMetrics()
        region = self._plan_compression(trajectory, metrics)
        if region is None:
            return trajectory, metrics
        summary = self._generate_summary(self._extract_turn_content_for_summary(trajectory, *region), metrics)
        return self._assemble_compressed(trajectory, *region, summary, metrics), metrics

    async def compress_trajectory_async(self, trajectory: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], TrajectoryMetrics]:
        """Async twin of ``compress_trajectory``."""
        metrics = TrajectoryMetrics()
        region = self._plan_compression(trajectory, metrics)
        if region is None:
            return trajectory, metrics
        summary = await self._generate_summary_async(self._extract_turn_content_for_summary(trajectory, *region), metrics)
        return self._assemble_compressed(trajectory, *region, summary, metrics), metrics

    async def process_entry_async(self, entry: Dict[str, Any]) -> Tuple[Dict[str, Any], TrajectoryMetrics]:
        """Compress one JSONL entry's ``conversations``; attach metrics when compressed."""
        if "conversations" not in entry:
            return entry, TrajectoryMetrics()
        compressed_trajectory, metrics = await self.compress_trajectory_async(entry["conversations"])
        result = dict(entry, conversations=compressed_trajectory)
        if self.config.metrics_per_trajectory and metrics.was_compressed:
            result["compression_metrics"] = metrics.to_dict()
        return result, metrics

    def process_directory(self, input_dir: Path, output_dir: Path):
        """Compress every ``*.jsonl`` in ``input_dir`` into ``output_dir`` (async, parallel API calls)."""
        asyncio.run(self._process_directory_async(input_dir, output_dir))

    async def _process_one(self, run: _RunProgress, file_path: Path, entry_idx: int, entry: Dict) -> Optional[Tuple[Dict[str, Any], TrajectoryMetrics]]:
        """Process one entry under the semaphore/timeout; None means dropped (timed out)."""
        async with run.semaphore:
            async with run.lock:
                run.in_flight += 1
            try:
                processed_entry, metrics = await asyncio.wait_for(self.process_entry_async(entry), timeout=self.config.per_trajectory_timeout)
                async with run.lock:
                    self.aggregate_metrics.add_trajectory_metrics(metrics)
                    if metrics.was_compressed:
                        run.compressed += 1
                        run.api_calls += metrics.summarization_api_calls
                    run.skipped += bool(metrics.skipped_under_target)
                    run.finish()
                return processed_entry, metrics
            except asyncio.TimeoutError:
                self.logger.warning("Timeout processing entry from %s:%s (>%ss)", file_path, entry_idx, self.config.per_trajectory_timeout)
                async with run.lock:
                    self.aggregate_metrics.trajectories_failed += 1
                    run.timeouts += 1
                    run.finish()
                return None
            except Exception as e:
                self.logger.error("Error processing entry from %s:%s: %s", file_path, entry_idx, e)
                async with run.lock:
                    self.aggregate_metrics.trajectories_failed += 1
                    run.finish(update_status=False)
                return entry, TrajectoryMetrics()  # keep the original on error

    async def _process_directory_async(self, input_dir: Path, output_dir: Path):
        console = Console()
        self.aggregate_metrics.processing_start_time = datetime.now().isoformat()
        start_time = time.time()
        jsonl_files = sorted(input_dir.glob("*.jsonl"))
        if not jsonl_files:
            self.logger.warning("No JSONL files found in %s", input_dir)
            return

        console.print("\n[dim]Loading all entries...[/dim]")
        all_entries = []  # List of (file_path, entry_idx, entry)
        for file_path in jsonl_files:
            def _warn(line_num, e, file_path=file_path):
                self.logger.warning("Skipping invalid JSON at %s:%s: %s", file_path, line_num, e)
            all_entries.extend((file_path, idx, entry) for idx, entry in _load_jsonl(file_path, _warn))
        total_entries = len(all_entries)

        console.print(f"\n{'='*60}")
        console.print(f"📂 Input: {input_dir}")
        console.print(f"📂 Output: {output_dir}")
        console.print(f"📄 Files to process: {len(jsonl_files)}")
        console.print(f"📊 Total trajectories: {total_entries:,}")
        console.print(f"🎯 Target max tokens: {self.config.target_max_tokens:,}")
        console.print(f"📝 Summary target tokens: {self.config.summary_target_tokens}")
        console.print(f"⚡ Max concurrent API calls: {self.config.max_concurrent_requests}")
        console.print(f"{'='*60}\n")

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(),
            TextColumn("•"), TimeElapsedColumn(), TextColumn("•"), TimeRemainingColumn(),
            console=console, refresh_per_second=10,  # Higher refresh for async
        ) as progress:
            run = _RunProgress(
                progress, progress.add_task(f"[cyan]Compressing {total_entries:,} trajectories", total=total_entries),
                progress.add_task("[dim]Starting...[/dim]", total=None),
                asyncio.Lock(), asyncio.Semaphore(self.config.max_concurrent_requests),
            )
            outcomes = await asyncio.gather(*(self._process_one(run, *item) for item in all_entries))
            progress.remove_task(run.status_task)

        # Write results preserving original order; timed-out entries are dropped.
        console.print("\n[dim]Writing output files...[/dim]")
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {f: [] for f in jsonl_files}
        for (file_path, _, _), outcome in zip(all_entries, outcomes):
            if outcome is not None:
                results[file_path].append(outcome[0])
        for file_path in jsonl_files:
            _write_jsonl(output_dir / file_path.name, results[file_path])

        self.aggregate_metrics.processing_end_time = datetime.now().isoformat()
        self.aggregate_metrics.processing_duration_seconds = time.time() - start_time
        self._print_summary()
        if self.config.metrics_enabled:
            metrics_path = output_dir / self.config.metrics_output_file
            with open(metrics_path, 'w', encoding="utf-8") as f:
                json.dump(self.aggregate_metrics.to_dict(), f, indent=2)
            console.print(f"\n💾 Metrics saved to {metrics_path}")

    def _print_summary(self):
        """Print comprehensive compression summary statistics."""
        m = self.aggregate_metrics.to_dict()
        s, t, u, a, z, p = m['summary'], m['tokens'], m['turns'], m['averages'], m['summarization'], m['processing']
        total, compressed = s['total_trajectories'], s['trajectories_compressed']
        pct = lambda n: (n / max(total, 1)) * 100  # noqa: E731
        duration = p['duration_seconds']
        time_str = f"{duration/60:.1f} minutes" if duration > 60 else f"{duration:.1f} seconds"

        sections = [
            ("📁 TRAJECTORIES", 54, [
                f"║{'':4}Total Processed:        {total:>10,}{' '*32}║",
                f"║{'':4}├─ Compressed:          {compressed:>10,}  ({pct(compressed):>5.1f}%){' '*18}║",
                f"║{'':4}├─ Skipped (under limit):{s['trajectories_skipped_under_target']:>9,}  ({pct(s['trajectories_skipped_under_target']):>5.1f}%){' '*18}║",
                f"║{'':4}├─ Still over limit:    {s['trajectories_still_over_limit']:>10,}  ({pct(s['trajectories_still_over_limit']):>5.1f}%){' '*18}║",
                f"║{'':4}└─ Failed:              {s['trajectories_failed']:>10,}{' '*32}║",
            ]),
            ("🔢 TOKENS", 60, [
                f"║{'':4}Before Compression:     {t['total_before']:>15,} tokens{' '*21}║",
                f"║{'':4}After Compression:      {t['total_after']:>15,} tokens{' '*21}║",
                f"║{'':4}Total Saved:            {t['total_saved']:>15,} tokens{' '*21}║",
                f"║{'':4}Overall Compression:    {t['overall_compression_ratio']:>14.1%}{' '*28}║",
            ] + ([f"║{'':4}Space Savings:          {(t['total_saved'] / t['total_before']) * 100:>14.1f}%{' '*28}║"] if t['total_before'] > 0 else [])),
            ("💬 CONVERSATION TURNS", 48, [
                f"║{'':4}Before Compression:     {u['total_before']:>15,} turns{' '*22}║",
                f"║{'':4}After Compression:      {u['total_after']:>15,} turns{' '*22}║",
                f"║{'':4}Total Removed:          {u['total_removed']:>15,} turns{' '*22}║",
            ]),
            ("📈 AVERAGES (Compressed Trajectories Only)", 27, [
                f"║{'':4}Avg Compression Ratio:  {a['avg_compression_ratio']:>14.1%}{' '*28}║",
                f"║{'':4}Avg Tokens Saved:       {a['avg_tokens_saved_per_compressed']:>14,.0f}{' '*28}║",
                f"║{'':4}Avg Turns Removed:      {a['avg_turns_removed_per_compressed']:>14.1f}{' '*28}║",
            ] if compressed > 0 else [f"║{'':4}No trajectories were compressed{' '*38}║"]),
            ("🤖 SUMMARIZATION API", 49, [
                f"║{'':4}API Calls Made:         {z['total_api_calls']:>15,}{' '*27}║",
                f"║{'':4}Errors:                 {z['total_errors']:>15,}{' '*27}║",
                f"║{'':4}Success Rate:           {z['success_rate']:>14.1%}{' '*28}║",
            ]),
            ("⏱️  PROCESSING TIME", 51, [
                f"║{'':4}Duration:               {time_str:>20}{' '*22}║",
                f"║{'':4}Throughput:             {total / max(duration, 0.001):>15.1f} traj/sec{' '*18}║",
                f"║{'':4}Started:                {p['start_time'][:19]:>20}{' '*22}║",
                f"║{'':4}Finished:               {p['end_time'][:19]:>20}{' '*22}║",
            ]),
        ]
        print("\n")
        print(f"╔{'═'*70}╗")
        print(f"║{'TRAJECTORY COMPRESSION REPORT':^70}║")
        for title, pad, rows in sections:
            print(f"╠{'═'*70}╣")
            print(f"║{'':2}{title}{' '*pad}║")
            print(f"║{'─'*70}║")
            for row in rows:
                print(row)
        print(f"╚{'═'*70}╝")

        ratios = self.aggregate_metrics.compression_ratios
        if ratios:
            saved = self.aggregate_metrics.tokens_saved_list
            print("\n📊 Distribution Summary:")
            print(f"   Compression ratios: min={min(ratios):.2%}, max={max(ratios):.2%}, median={sorted(ratios)[len(ratios)//2]:.2%}")
            print(f"   Tokens saved:       min={min(saved):,}, max={max(saved):,}, median={sorted(saved)[len(saved)//2]:,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_cli_config(config: str, target_max_tokens: Optional[int], tokenizer: Optional[str]) -> CompressionConfig:
    """Load the YAML config (defaults if missing) and apply CLI overrides."""
    if Path(config).exists():
        print(f"📋 Loading config from {config}")
        compression_config = CompressionConfig.from_yaml(config)
    else:
        print(f"⚠️  Config not found at {config}, using defaults")
        compression_config = CompressionConfig()
    if target_max_tokens:
        compression_config.target_max_tokens = target_max_tokens
    if tokenizer:
        compression_config.tokenizer_name = tokenizer
    return compression_config


def _print_dry_run(icon: str, target: Any, output_path: Path) -> None:
    print("\n🔍 DRY RUN MODE - analyzing without writing")
    print(f"{icon} Would process: {target}")
    print(f"{icon} Would output to: {output_path}")


def _sample(entries: list, sample_percent: float) -> list:
    return random.sample(entries, min(max(1, int(len(entries) * sample_percent / 100)), len(entries)))


def _run_file_mode(input_path: Path, output: Optional[str], compression_config: CompressionConfig, sample_percent: Optional[float], seed: int, dry_run: bool) -> None:
    """Single-file input: (sample,) compress via a temp directory, merge into one output file."""
    print("📄 Input mode: Single JSONL file")
    output_path = Path(output) if output else input_path.parent / (input_path.stem + compression_config.output_suffix + ".jsonl")
    entries = [entry for _, entry in _load_jsonl(input_path, lambda n, e: print(f"⚠️  Skipping invalid JSON at line {n}: {e}"), start=1)]
    total_entries = len(entries)
    print(f"   Loaded {total_entries:,} trajectories from {input_path.name}")
    if sample_percent is not None:
        random.seed(seed)
        entries = random.sample(entries, max(1, int(total_entries * sample_percent / 100)))
        print(f"   Sampled {len(entries):,} trajectories ({sample_percent}% of {total_entries:,})")
    if dry_run:
        _print_dry_run("📄", f"{len(entries):,} trajectories", output_path)
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_input_dir, temp_output_dir = Path(temp_dir) / "input", Path(temp_dir) / "output"
        temp_input_dir.mkdir()
        _write_jsonl(temp_input_dir / "trajectories.jsonl", entries)
        TrajectoryCompressor(compression_config).process_directory(temp_input_dir, temp_output_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as out_f:
            for jsonl_file in sorted(temp_output_dir.glob("*.jsonl")):
                with open(jsonl_file, 'r', encoding='utf-8') as in_f:
                    shutil.copyfileobj(in_f, out_f)
        metrics_file = temp_output_dir / compression_config.metrics_output_file
        if metrics_file.exists():
            metrics_output = output_path.parent / (output_path.stem + "_metrics.json")
            shutil.copy(metrics_file, metrics_output)
            print(f"💾 Metrics saved to {metrics_output}")
    print("\n✅ Compression complete!")
    print(f"📄 Output: {output_path}")


def _run_dir_mode(input_path: Path, output: Optional[str], compression_config: CompressionConfig, sample_percent: Optional[float], seed: int, dry_run: bool) -> None:
    """Directory input: compress in place, or per-file sample into a temp dir first."""
    print("📁 Input mode: Directory of JSONL files")
    output_path = Path(output) if output else input_path.parent / (input_path.name + compression_config.output_suffix)
    if sample_percent is None:
        if dry_run:
            _print_dry_run("📁", input_path, output_path)
            return
        TrajectoryCompressor(compression_config).process_directory(input_path, output_path)
    else:
        print(f"\n⚠️  Sampling from directory: will sample {sample_percent}% from each file")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input_dir = Path(temp_dir) / "input"
            temp_input_dir.mkdir()
            random.seed(seed)
            total_original = total_sampled = 0
            for jsonl_file in sorted(input_path.glob("*.jsonl")):
                entries = [entry for _, entry in _load_jsonl(jsonl_file)]
                sampled_entries = _sample(entries, sample_percent)
                total_original += len(entries)
                total_sampled += len(sampled_entries)
                _write_jsonl(temp_input_dir / jsonl_file.name, sampled_entries)
            print(f"   Sampled {total_sampled:,} from {total_original:,} total trajectories")
            if dry_run:
                _print_dry_run("📁", temp_input_dir, output_path)
                return
            TrajectoryCompressor(compression_config).process_directory(temp_input_dir, output_path)
    print("\n✅ Compression complete!")


def main(input: str, output: str = None, config: str = "configs/trajectory_compression.yaml", target_max_tokens: int = None,
         tokenizer: str = None, sample_percent: float = None, seed: int = 42, dry_run: bool = False):
    """
    Compress agent trajectories to fit within a target token budget.
    
    Supports both single JSONL files and directories containing multiple JSONL files.
    Optionally sample a percentage of trajectories before compression.
    
    Args:
        input: Path to JSONL file or directory containing JSONL files
        output: Output path (file for file input, directory for dir input)
                Default: adds "_compressed" suffix to input name
        config: Path to YAML configuration file
        target_max_tokens: Override target token count from config
        tokenizer: Override tokenizer name from config
        sample_percent: Sample this percentage of trajectories (1-100) before compression
        seed: Random seed for sampling reproducibility (default: 42)
        dry_run: Analyze without compressing (just show what would happen)
    """
    print("🗜️  Trajectory Compressor")
    print("=" * 60)
    compression_config = _load_cli_config(config, target_max_tokens, tokenizer)
    if sample_percent is not None:
        if sample_percent <= 0 or sample_percent > 100:
            print(f"❌ sample_percent must be between 1 and 100, got {sample_percent}")
            return
        print(f"🎲 Will sample {sample_percent}% of trajectories (seed={seed})")
    input_path = Path(input)
    if not input_path.exists():
        print(f"❌ Input not found: {input}")
        return
    run_mode = _run_file_mode if input_path.is_file() else _run_dir_mode
    run_mode(input_path, output, compression_config, sample_percent, seed, dry_run)


if __name__ == "__main__":
    fire.Fire(main)
