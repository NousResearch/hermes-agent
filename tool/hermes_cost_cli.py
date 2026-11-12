#!/usr/bin/env python3
"""hermes-cost-meter — a live session-cost meter for the Hermes CLI, as a wrapper CLI.

No core edits. Hermes documents five protected extension hooks on ``HermesCLI``
(``developer-guide/extending-the-cli``); this tool uses the two that matter here:

* ``_get_extra_tui_widgets()`` — injects a one-line prompt_toolkit widget that renders
  between the transcript and the status bar.
* ``process_command()`` — adds ``/cost`` for a full breakdown on demand.

Why not ``display.show_cost``: that config key is documented but not implemented in the
CLI (upstream issue #93951; PRs #75487/#99649 are open and unmerged), so nothing renders.

Cost is computed from the session's own token counters times the model's rate card —
i.e. the same ``agent.usage_pricing`` helpers ``/usage`` and state.db use — so the number
tracks the provider's meter without touching the turn loop. OpenRouter ``@preset/<slug>``
names carry no price row, so they are resolved to the routed model through
``GET /api/v1/presets/<slug>`` and cached on disk for a week.

Display-only: nothing here mutates context, toolsets or the system prompt, so prompt
caching is untouched.
"""

from __future__ import annotations

import json
import os
import time
from decimal import Decimal

from prompt_toolkit.layout import FormattedTextControl, Window

from cli import HermesCLI

CACHE_FILE = os.path.join(os.path.expanduser("~/.hermes"), "cache", "cost_meter_presets.json")
PRESET_TTL_S = 7 * 24 * 3600


# ── preset resolution (OpenRouter presets have no /models price row) ──────────────

def _preset_slug(model: str) -> str | None:
    if "@preset/" not in (model or ""):
        return None
    return (model.split("@preset/", 1)[1].strip().strip("/")) or None


def _load_preset_cache() -> dict:
    try:
        with open(CACHE_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_preset_cache(data: dict) -> None:
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        tmp = CACHE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1, sort_keys=True)
        os.replace(tmp, CACHE_FILE)
    except Exception:
        pass


def _resolve_preset(slug: str) -> str | None:
    """Route model behind ``@preset/<slug>``, cached on disk. Never raises."""
    cache = _load_preset_cache()
    hit = cache.get(slug) or {}
    if isinstance(hit.get("model"), str) and hit["model"] and time.time() - float(hit.get("at", 0)) < PRESET_TTL_S:
        return hit["model"]
    key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not key:
        return None
    model = None
    try:
        import urllib.request

        req = urllib.request.Request(
            f"https://openrouter.ai/api/v1/presets/{slug}",
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=4) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        model = ((payload.get("data") or {}).get("designated_version") or {}).get("config", {}).get("model")
    except Exception:
        model = None
    if isinstance(model, str) and model:
        cache[slug] = {"model": model, "at": time.time()}
        _save_preset_cache(cache)
        return model
    return None


# ── pricing ──────────────────────────────────────────────────────────────────────

def session_cost(agent) -> tuple[float | None, str]:
    """(usd, note). ``(None, note)`` when this route has no price row — never a fake 0."""
    if agent is None:
        return None, ""
    # A provider that reports its own dollar cost wins over any rate card.
    reported = getattr(agent, "session_estimated_cost_usd", None)
    status = str(getattr(agent, "session_cost_status", "") or "")
    if isinstance(reported, (int, float)) and reported > 0 and status in {"actual", "estimated"}:
        return float(reported), status

    model = getattr(agent, "model", "") or ""
    slug = _preset_slug(model)
    priced_model = (_resolve_preset(slug) if slug else None) or model
    try:
        from agent.usage_pricing import CanonicalUsage, estimate_usage_cost

        usage = CanonicalUsage(
            input_tokens=int(getattr(agent, "session_input_tokens", 0) or 0),
            output_tokens=int(getattr(agent, "session_output_tokens", 0) or 0),
            cache_read_tokens=int(getattr(agent, "session_cache_read_tokens", 0) or 0),
            cache_write_tokens=int(getattr(agent, "session_cache_write_tokens", 0) or 0),
        )
        result = estimate_usage_cost(
            priced_model, usage, provider=getattr(agent, "provider", None),
            base_url=getattr(agent, "base_url", None), api_key=getattr(agent, "api_key", "") or "",
        )
    except Exception:
        return None, ""
    if result.amount_usd is None:
        return None, (slug and f"no pricing for {priced_model}") or "no pricing for this route"
    return float(result.amount_usd), result.status


def _fmt(usd: float | None) -> str:
    if usd is None:
        return "~$?"
    if usd == 0:
        return "$0"
    if usd < 0.01:
        return f"~${usd:.4f}"
    if usd < 100:
        return f"~${usd:.2f}"
    return f"~${usd:.0f}"


# ── the wrapper ──────────────────────────────────────────────────────────────────

class CostMeterCLI(HermesCLI):
    """HermesCLI + a live spend line, using only the documented extension hooks."""

    def _get_extra_tui_widgets(self) -> list:
        widgets = list(super()._get_extra_tui_widgets() or [])
        widgets.append(
            Window(
                FormattedTextControl(self._cost_fragments),
                height=lambda: 1 if self._cost_ready() else 0,
                style="class:status-bar-dim",
                wrap_lines=False,
            )
        )
        return widgets

    def _cost_ready(self) -> bool:
        """Nothing to meter before the first priced call — the row stays collapsed."""
        agent = getattr(self, "agent", None)
        return bool(agent is not None and (getattr(agent, "session_api_calls", 0) or 0) > 0)

    def _cost_fragments(self) -> list[tuple[str, str]]:
        """One dim line: session spend, this turn's spend, and a hint that /cost exists."""
        if not self._cost_ready():
            return []
        usd, note = session_cost(getattr(self, "agent", None))
        turn = usd - self._turn_base if (usd is not None and self._turn_base is not None) else None
        if usd is None:
            return [("class:status-bar-strong", " cost: "), ("class:status-bar-dim", note or "unavailable")]
        parts = [("class:status-bar-strong", f" {_fmt(usd)}")]
        if turn is not None and turn > 0:
            parts.append(("class:status-bar-dim", f" · turn {_fmt(turn)}"))
        parts.append(("class:status-bar-dim", "  · /cost"))
        return parts

    def process_command(self, command: str, *, redispatch: bool = False) -> bool:
        if command.strip().lower().startswith("/cost"):
            self._print_cost()
            return True
        return super().process_command(command, redispatch=redispatch)

    # -- per-turn baseline, refreshed whenever a turn starts ------------------------
    _turn_base: float | None = None

    def _turn_summary_begin(self) -> None:
        super()._turn_summary_begin()
        try:
            usd, _ = session_cost(getattr(self, "agent", None))
            self._turn_base = usd
        except Exception:
            self._turn_base = None

    def _print_cost(self):
        agent = getattr(self, "agent", None)
        usd, note = session_cost(agent)
        rows = [
            ("model", getattr(agent, "model", "-")),
            ("session cost", _fmt(usd) + (f"  ({note})" if note else "")),
            ("input tokens", f"{int(getattr(agent, 'session_input_tokens', 0) or 0):,}"),
            ("output tokens", f"{int(getattr(agent, 'session_output_tokens', 0) or 0):,}"),
            ("cache read", f"{int(getattr(agent, 'session_cache_read_tokens', 0) or 0):,}"),
            ("API calls", f"{int(getattr(agent, 'session_api_calls', 0) or 0):,}"),
        ]
        print("\n  Session cost (estimate, main-agent usage only)")
        print("  " + "─" * 44)
        for label, value in rows:
            print(f"  {label:<16} {value}")
        print("  " + "─" * 44 + "\n")


def main() -> int:
    cli = CostMeterCLI()
    cli.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
