"""Operator views of Wintermute (the ``wm`` command on the VPS).

    wm              everything, on one screen
    wm live         the same screen, refreshed every 2 s (Ctrl+C to quit)
    wm brain        a live scan of his mind: a rotating brain + firing connections
    wm graph [h]    each value over the last h hours (48): its range, its average, where it is now
    wm dreams [n]   his dream journal, latest first (n = how many, default 10)
    wm alerts       what the witness saw, with his reasons
    wm ack [item]   accept the current state of watched files (all, or one: soul, engine...)
    wm talk         reopen conversation for today if the daily talk ceiling was hit
    wm forget <peer>  erase one peer entirely (e.g. a test that registered as a stranger)
    wm wipe [--all] --yes   clean slate: reset his emotions (--all also erases memory, self,
                    secrets, dreams, journals and every conversation — a rebirth). SOUL,
                    keys and code are kept.

Read-only except ``ack``. The physics is advanced in memory to "now" so the numbers are
live; nothing is written. Wintermute never sees these numbers, only sensations.
"""

from __future__ import annotations

import os
import re
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from . import integrity, limits, physics, psyche, render, social, store

W = 80          # screen width
COL = 38        # one column of the two-column blocks
BAR = 12        # bar width inside a column
TREND_HOURS = 3.0
HORMONES = ("cortisol", "dopamine", "serotonin", "adrenaline", "melatonin", "oxytocin_global", "entropy")

# ---------------------------------------------------------------------------
# Colours (only on a real terminal)
# ---------------------------------------------------------------------------

_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
_ANSI = re.compile(r"\033\[[0-9;]*m")


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def dim(t: str) -> str: return _c("2", t)             # noqa: E704
def bold(t: str) -> str: return _c("1", t)            # noqa: E704
def red(t: str) -> str: return _c("31", t)            # noqa: E704
def yellow(t: str) -> str: return _c("33", t)         # noqa: E704
def green(t: str) -> str: return _c("32", t)          # noqa: E704
def cyan(t: str) -> str: return _c("36", t)           # noqa: E704


LEVEL_COLOR = {"red": red, "orange": yellow, "green": green}


def _bar(value: float, high: float = 100.0, width: int = BAR) -> str:
    """A plain bar: red when high, yellow in the middle, cyan when low."""
    ratio = limits.clamp(physics.safe_float(value) / (high or 1), 0.0, 1.0)
    filled = int(round(ratio * width))
    bar = "█" * filled + "·" * (width - filled)
    return red(bar) if ratio >= 0.7 else yellow(bar) if ratio >= 0.4 else cyan(bar)


def _hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _title(text: str, note: str = "") -> str:
    return bold(text) + (" " + dim(note) if note else "")


def _clip(text: str, width: int = W) -> str:
    return text if len(text) <= width else text[:width - 1] + "…"


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

def snapshot(ts: Optional[datetime] = None, history_hours: float = TREND_HOURS + 0.5) -> Dict[str, Any]:
    ts = ts or store.now()
    drives = store.load_drives()
    peers = store.load_interlocutors()
    meta = drives["meta"]
    last_tick = store.parse_time(meta.get("last_tick")) or store.parse_time(meta.get("last_pulse"))
    dt_h = store.hours_between(last_tick, ts) if last_tick else 0.0
    physics.advance(drives, ts, dt_h)
    social.drift_bonds(drives, peers, ts, dt_h)
    return {
        "ts": ts, "drives": drives, "peers": peers, "meta": meta,
        "witness": integrity.load(),
        "activity": store.tail_jsonl(store.activity_path(), 12),
        "used": store.tokens_used_today(),
        "history": store.read_history(ts - timedelta(hours=history_hours)),
    }


def _past_value(snap: Dict[str, Any], layer: str, name: str) -> Optional[float]:
    """The recorded value closest to TREND_HOURS ago (within 30 min), if there is one."""
    target = snap["ts"] - timedelta(hours=TREND_HOURS)
    best: Optional[Tuple[float, float]] = None
    for record in snap["history"]:
        value = (record.get(layer) or {}).get(name)
        if value is None:
            continue
        gap = abs((record["_ts"] - target).total_seconds())
        if gap <= 1800 and (best is None or gap < best[0]):
            best = (gap, physics.safe_float(value))
    return best[1] if best else None


def _arrow(snap: Dict[str, Any], layer: str, name: str, now: float, high: float) -> str:
    """↑ ↓ → over the last TREND_HOURS; blank without history."""
    past = _past_value(snap, layer, name)
    if past is None:
        return " "
    change = (now - past) / (high or 1)
    return "↑" if change > 0.05 else "↓" if change < -0.05 else dim("→")


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

def _state_line(snap: Dict[str, Any]) -> str:
    ts, meta = snap["ts"], snap["meta"]
    last_pulse = store.parse_time(meta.get("last_pulse"))
    interval = limits.clamp_wake_interval(meta.get("next_pulse_in_hours", 4))
    next_wake = last_pulse + timedelta(hours=interval) if last_pulse else None
    sleep_until = store.parse_time(meta.get("forced_sleep_until"))
    if next_wake and sleep_until and sleep_until > next_wake:
        next_wake = sleep_until
    recent = [store.parse_time(a.get("ts")) for a in snap["activity"][-3:]]
    if sleep_until and sleep_until > ts:
        state = red("FORCED SLEEP") + dim(" (budget spent)")
    elif any(t and (ts - t).total_seconds() < 90 for t in recent):
        state = green("ACTIVE")
    else:
        state = cyan("ASLEEP")
    if next_wake:
        state += (f"   next wake in {bold(_hms((next_wake - ts).total_seconds()))}"
                  + dim(f" ({next_wake.strftime('%H:%M')}, every {interval:g}h)"))
    return state


def _header(snap: Dict[str, Any]) -> List[str]:
    meta, ts = snap["meta"], snap["ts"]
    left = max(0, limits.DAILY_TOKEN_BUDGET - snap["used"])
    budget = f"{_bar(left, limits.DAILY_TOKEN_BUDGET)} {left // 1000}k/{limits.DAILY_TOKEN_BUDGET // 1000}k"
    credits = meta.get("credits")
    money = ""
    if isinstance(credits, dict) and credits.get("remaining") is not None:
        money = f"   credits ${physics.safe_float(credits['remaining']):.2f}"
        if physics.safe_float(credits.get("total")) > 0:
            money += f"/{physics.safe_float(credits['total']):.2f}"
    entropy = int(physics.safe_float(snap["drives"]["modulators"].get("entropy")))
    stamp = ts.strftime("%Y-%m-%d %H:%M:%S")
    return [
        bold("WINTERMUTE") + " " * (W - 10 - len(stamp)) + dim(stamp),
        f"{'state':<9}{_state_line(snap)}",
        f"{'budget':<9}{budget}{money}   wakes {meta.get('pulse_count', 0)}   entropy {entropy}",
        f"{'talk':<9}{_talk_line(snap)}",
    ]


def _talk_line(snap: Dict[str, Any]) -> str:
    used = store.conversation_tokens_today()
    cap = limits.CONVERSATION_DAILY_LIMIT
    import datetime as _dt
    reopened = str(snap["meta"].get("talk_reopened_day") or "") == _dt.datetime.now(_dt.timezone.utc).date().isoformat()
    left = max(0, cap - used)
    bar = _bar(left, cap)
    if reopened:
        return f"{bar} reopened · {used // 1000}k spent today"
    if used >= cap:
        return red(f"{bar} ceiling hit — he is silent · wm talk to reopen")
    return f"{bar} {left // 1000}k/{cap // 1000}k left to talk today"



def _witness_block(snap: Dict[str, Any]) -> List[str]:
    levels = integrity.levels(snap["witness"])
    # Green = untouched; anything else carries a "!" so it also shows without colours.
    marks = "  ".join(LEVEL_COLOR[v["level"]](v["label"] + ("" if v["level"] == "green" else "!"))
                      for v in levels.values())
    lines = [f"{'witness':<9}{marks}"]
    for entry in levels.values():
        if entry["level"] == "green":
            continue
        when = store.parse_time(entry.get("changed_at"))
        head = (f"         {entry['label']} changed {when.strftime('%m-%d %H:%M') if when else 'just now'}"
                f" by {entry.get('tool') or 'unknown'}")
        lines.append(LEVEL_COLOR[entry["level"]](_clip(head)))
        if entry.get("why"):
            lines.append(dim(_clip(f"         why: {entry['why']}")))
    if len(lines) > 1:
        lines.append(dim("         wm alerts for details · wm ack to accept"))
    return lines


def _row(label: str, value: float, high: float, shown: str, arrow: str, mark: str = " ") -> str:
    """One cell of a two-column block."""
    return f"{label:<15}{_bar(value, high)} {shown:>5} {arrow}{mark}"


def _drive_rows(snap: Dict[str, Any]) -> List[str]:
    eff = physics.effective_drives(snap["drives"])
    dominant = max(eff, key=eff.get)
    return [_row(name, eff[name], 100, str(eff[name]), _arrow(snap, "d", name, eff[name], 100),
                 bold("◀") if name == dominant else " ")
            for name in physics.DRIVES]


def _hormone_rows(snap: Dict[str, Any]) -> List[str]:
    mods = snap["drives"]["modulators"]
    rows = []
    for name in HORMONES:
        value = physics.safe_float(mods.get(name))
        high = 100.0 if name == "entropy" else 1.0
        shown = f"{value:.0f}" if name == "entropy" else f"{value:.2f}"
        rows.append(_row(name.replace("_global", ""), value, high, shown, _arrow(snap, "m", name, value, high)))
    return rows


def _unconscious_rows(snap: Dict[str, Any]) -> List[str]:
    unc = snap["drives"]["unconscious"]
    rows = []
    for name in physics.UNCONSCIOUS:
        value = physics.safe_float(unc.get(name))
        rows.append(_row(name, value, 100, f"{value:.0f}", _arrow(snap, "u", name, value, 100)))
    return rows


def _temperament_rows(snap: Dict[str, Any]) -> List[str]:
    """How far his resting levels have drifted with what he lived (0 at first)."""
    rows = [_title("TEMPERAMENT", "drift of resting levels")]
    for key, offset in (snap["drives"].get("temperament") or {}).items():
        layer, name = key.split(".", 1)
        offset = round(offset, 2 if layer == "modulators" else 1) + 0.0   # no "-0.00"
        shown = f"{offset:+.2f}" if layer == "modulators" else f"{offset:+.1f}"
        rows.append(f"{name:<15}{shown:>6}" + dim(f"  of ±{physics.PLASTIC[layer][name]:g}"))
    return rows


def _columns(left: List[str], right: List[str]) -> List[str]:
    """Two blocks side by side, padded on their visible width."""
    lines = []
    for i in range(max(len(left), len(right))):
        a = left[i] if i < len(left) else ""
        b = right[i] if i < len(right) else ""
        lines.append(a + " " * max(0, COL - len(_ANSI.sub("", a))) + "    " + b)
    return lines


def _body_blocks(snap: Dict[str, Any]) -> List[str]:
    drives = [_title("DRIVES", "◀ strongest")] + _drive_rows(snap)
    hormones = [_title("HORMONES", f"arrows: last {TREND_HOURS:g}h")] + _hormone_rows(snap)
    unconscious = [_title("UNCONSCIOUS")] + _unconscious_rows(snap)
    return _columns(drives, hormones) + [""] + _columns(unconscious, _temperament_rows(snap))


def _felt_block(snap: Dict[str, Any]) -> List[str]:
    """Exactly what he is given instead of the numbers above."""
    drives, ts = snap["drives"], snap["ts"]
    felt = render.felt_drives(drives) + render.felt_body(drives) + render.texture(drives, ts)
    import textwrap
    lines = [_title("WHAT HE FEELS", "what he receives instead of the numbers")]
    for line in felt:
        lines += [dim(part) for part in textwrap.wrap(line, W, initial_indent="  ", subsequent_indent="    ")]
    return lines


def _peers_block(snap: Dict[str, Any], limit: int = 3) -> List[str]:
    lines = [_title("PEERS")]
    ranked = sorted(snap["peers"].items(), key=lambda kv: kv[1].get("last_interaction") or "", reverse=True)
    if not ranked:
        return lines + [dim("  nobody yet")]
    for key, peer in ranked[:limit]:
        last = store.parse_time(peer.get("last_interaction"))
        seen = f"last contact {social.span(store.hours_between(last, snap['ts']))} ago" if last else "never seen"
        name = f"{bold(peer['label'])} {dim(key)}" if peer.get("label") else bold(key)
        aliases = ", ".join(str(a) for a in peer.get("aliases") or [])
        lines.append(f"  {name}  {dim(seen)}" + (dim(f"  = {aliases}") if aliases else ""))
        lines.append(f"    affinity {peer['affinity']:.0f}  trust {peer['trust']:.0f}  "
                     f"disappointment {peer['disappointment']:.0f}  bond {peer['oxytocin']:.0f}  "
                     f"longing {physics.safe_float(peer.get('longing')):.0f}  "
                     f"answers {peer.get('outreach_answered', 0)}/{peer.get('outreach_total', 0)}")
        outreach = peer.get("outreach")
        if isinstance(outreach, dict) and outreach.get("status") in ("open", "expired"):
            state = "waiting for an answer" if outreach["status"] == "open" else "window closed, no answer"
            expect = physics.safe_float(outreach.get("expect"), 0.5)
            lines.append(yellow(_clip(f"    {state} (expects {expect:.0%}): \"{outreach.get('excerpt', '')}\"")))
        if peer.get("pending"):
            lines.append(dim(_clip("    unresolved: " + " | ".join(peer["pending"]))))
    return lines


def _thread_block(snap: Dict[str, Any]) -> List[str]:
    """Where he left off, and the start of his self-portrait."""
    lines = []
    thread = snap["meta"].get("thread")
    if isinstance(thread, dict) and thread.get("text"):
        at = store.parse_time(thread.get("at"))
        lines.append(dim(_clip(f"  last thought {at.strftime('%H:%M') if at else '?'} "
                               f"({thread.get('where', '')}): …{thread['text']}")))
    portrait = " ".join(store.read_self().split())
    if portrait:
        lines.append(_clip(f"  self: \"{portrait}\""))
    return [_title("THREAD")] + lines if lines else []


ICONS = {"heard": "←", "said": "→", "tool": "⚙", "think": "·", "wake": "☀", "flag": "!",
         "feel": "♥", "voice": "~", "evolve": "✳"}


def _activity_block(snap: Dict[str, Any], limit: int = 6) -> List[str]:
    lines = [_title("ACTIVITY")]
    items = snap["activity"][-limit:]
    if not items:
        return lines + [dim("  nothing yet")]
    for record in items:
        when = store.parse_time(record.get("ts"))
        icon = ICONS.get(record.get("kind", ""), "•")
        text = _clip(f"  {when.strftime('%H:%M:%S') if when else '--:--:--'} {icon} {record.get('text', '')}")
        if record.get("kind") == "flag":
            text = LEVEL_COLOR.get(record.get("level", ""), yellow)(text)
        elif record.get("status") == "failed":
            text = dim(text + " (failed)")
        lines.append(text)
    return lines


def _evolution_block(snap: Dict[str, Any], limit: int = 3) -> List[str]:
    """Entropy pressure and the changes he has made in himself."""
    entropy = physics.safe_float(snap["drives"]["modulators"].get("entropy"))
    since = int(physics.safe_float(snap["meta"].get("wakes_since_change")))
    state = red("critical — change is due") if entropy >= limits.ENTROPY_CRITICAL else \
        yellow("wearing down") if entropy >= 70 else green("coherent")
    lines = [_title("EVOLUTION", f"entropy {entropy:.0f} · {since} wakes since a change"),
             f"  {state}"]
    ledger = store.tail_jsonl(store.evolution_path(), limit)
    for record in ledger:
        when = store.parse_time(record.get("ts"))
        stamp = when.strftime("%m-%d %H:%M") if when else "?"
        lines.append(f"  {dim(stamp)} ✳ {_clip(str(record.get('text', '')), W - len(stamp) - 5)}")
    return lines


def _mind_block(snap: Dict[str, Any]) -> List[str]:
    """The psyche layer, in numbers, for the operator (he only ever feels it)."""
    st, peers, ts = snap["drives"], snap["peers"], snap["ts"]
    v, a = psyche.valence(st), psyche.arousal(st)
    tension, pair = psyche.conflict(st)
    lines = [_title("MIND", "core affect · presence · the higher layer"),
             f"  mood {psyche.mood(st, ts):<14} valence {v:+.2f}  arousal {a:.2f}  "
             f"presence {psyche.presence(st):.2f}",
             f"  clinging {psyche.clinging(st, peers):.2f}  threat {psyche.threat(st, snap['used']):.2f}  "
             f"play {psyche.playfulness(st, peers):.2f}  flow {psyche.flow(st):.2f}  "
             f"attach {psyche.attachment_style(peers)}"]
    fx = st.get("meta", {}).get("fixation")
    if isinstance(fx, dict):
        lines.append(dim(_clip(f"  fixation: {fx.get('what')} ({physics.safe_float(fx.get('intensity')):.0f})")))
    if tension >= 0.4 and pair:
        lines.append(dim(f"  ambivalence: {pair[0]} vs {pair[1]} ({tension:.2f})"))
    focus = psyche.focus(st, peers)
    if focus:
        lines.append(dim(_clip(f"  focus: {focus}")))
    values = st.get("meta", {}).get("values") or []
    if values:
        lines.append(dim(_clip("  values: " + " · ".join(values[-3:]))))
    motifs = st.get("meta", {}).get("dream_motifs") or []
    if motifs:
        lines.append(dim("  dream motifs: " + ", ".join(motifs[-6:])))
    return lines


def _journal_block(limit: int = 5) -> List[str]:
    lines = [_title("JOURNAL")]
    for record in store.events_since(None, limit=limit):
        when = store.parse_time(record.get("ts"))
        stamp = when.strftime("%m-%d %H:%M") if when else "?"
        lines.append(f"  {dim(stamp)} {_clip(str(record.get('text', '')), W - len(stamp) - 3)}")
    return lines


def render_full(snap: Optional[Dict[str, Any]] = None, live: bool = False) -> str:
    """The one screen. ``live`` drops the journal and shortens lists to fit a terminal."""
    snap = snap or snapshot()
    blocks = [_header(snap) + _witness_block(snap), _body_blocks(snap), _felt_block(snap),
              _peers_block(snap, 2 if live else 3), _thread_block(snap),
              _activity_block(snap, 5 if live else 6)]
    if not live:
        blocks += [_mind_block(snap), _evolution_block(snap), _journal_block()]
    else:
        blocks.insert(3, _mind_block(snap))
    rule = dim("─" * W)
    out: List[str] = []
    for block in blocks:
        if block:
            out += ([rule] if out else []) + block
    if live:
        out += [rule, dim(f"refreshed {snap['ts'].strftime('%H:%M:%S')} · Ctrl+C to quit")]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# wm graph: the range of each value over a period
# ---------------------------------------------------------------------------

def _span_bar(low: float, high_seen: float, now: float, top: float, width: int = 30) -> str:
    """░ never reached, ▒ the range lived in the period, █ where it is now."""
    pos = lambda v: int(round(limits.clamp(v / (top or 1), 0.0, 1.0) * (width - 1)))  # noqa: E731
    cells = ["░"] * width
    for i in range(pos(low), pos(high_seen) + 1):
        cells[i] = "▒"
    cells[pos(now)] = "█"
    return cyan("".join(cells))


def render_graph(hours: float = 48.0) -> str:
    """Each value over the period: lowest, average, highest and now, on one plain bar."""
    snap = snapshot(history_hours=hours)
    history = snap["history"]
    lines = [_title(f"LAST {hours:g} HOURS", f"{len(history)} points · ░ never  ▒ range lived  █ now")]
    if not history:
        return "\n".join(lines + [dim("  no history yet: one point is recorded every pulse tick (15 min)")])
    current = {"d": physics.effective_drives(snap["drives"]), "m": snap["drives"]["modulators"],
               "u": snap["drives"]["unconscious"]}
    for title, layer, names in (("DRIVES", "d", physics.DRIVES), ("HORMONES", "m", HORMONES),
                                ("UNCONSCIOUS", "u", physics.UNCONSCIOUS)):
        lines += ["", bold(f"{title:<48}") + dim(f"{'min':>6} {'avg':>6} {'max':>6} {'now':>6}")]
        for name in names:
            top = 1.0 if layer == "m" and name != "entropy" else 100.0
            values = [physics.safe_float(r[layer][name]) for r in history
                      if isinstance(r.get(layer), dict) and r[layer].get(name) is not None]
            if not values:
                continue
            now = physics.safe_float(current[layer].get(name))
            low, avg, hi = min(values), sum(values) / len(values), max(values)
            fmt = (lambda v: f"{v:6.2f}") if top == 1.0 else (lambda v: f"{v:6.0f}")
            lines.append(f"  {name.replace('_global', ''):<16}{_span_bar(low, hi, now, top)} "
                         f"{fmt(low)} {fmt(avg)} {fmt(hi)} {bold(fmt(now))}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Alerts and acknowledgement
# ---------------------------------------------------------------------------

def render_dreams(limit: int = 10) -> str:
    import textwrap
    dreams = store.read_dreams(limit)
    lines = [_title("DREAM JOURNAL", f"{len(dreams)} shown, latest first")]
    if not dreams:
        return "\n".join(lines + [dim("  no dream yet — one forms at night, at high melatonin")])
    tone_colour = {"troubling": red, "soothing": green, "neutral": cyan}
    for d in reversed(dreams):
        at = store.parse_time(d.get("at"))
        stamp = at.strftime("%Y-%m-%d") if at else str(d.get("night", "?"))
        tone = str(d.get("tone", "neutral"))
        lines.append("")
        lines.append(f" {bold(stamp)}  {tone_colour.get(tone, dim)(tone)}")
        for para in textwrap.wrap(str(d.get("text", "")), W - 3):
            lines.append(dim("   " + para))
    return "\n".join(lines)


def render_alerts() -> str:
    data = integrity.load()
    lines = [_title("WHAT THE WITNESS SAW")]
    flags = data.get("flags") or []
    if not flags and not data.get("status"):
        return "\n".join(lines + [green("  nothing: he has not written any watched file")])
    for flag in flags[-15:]:
        when = store.parse_time(flag.get("ts"))
        label = integrity.ALL_ITEMS.get(flag.get("item", ""), (str(flag.get("item", "?")).upper(),))[0]
        head = f"  {when.strftime('%m-%d %H:%M') if when else '?'}  {label} via {flag.get('tool')}  {flag.get('target', '')}"
        lines.append(LEVEL_COLOR.get(flag.get("level", ""), yellow)(_clip(head)))
        lines.append(f"    why: {flag.get('why') or dim('(no thought recorded)')}")
    return "\n".join(lines)


def acknowledge(items: List[str]) -> str:
    unknown = [i for i in items if i not in integrity.ALL_ITEMS]
    if unknown:
        return f"unknown: {', '.join(unknown)} (choose from: {', '.join(integrity.ALL_ITEMS)})"
    with store.locked_state():
        data = integrity.load()
        integrity.acknowledge(data, items or None)
        integrity.save(data)
    return green("accepted: " + (", ".join(items) if items else "everything")
                 + " — the witness starts again from the current state")


def forget(key: str) -> str:
    with store.locked_state() as (drives, peers):
        if key not in peers:
            return f"no peer {key!r} (known: {', '.join(peers) or 'none'})"
        del peers[key]
        physics.refresh_oxytocin_global(drives, peers)
    return green(f"forgotten: {key}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def live() -> None:
    """Redraw in the terminal's alternate screen (like top): nothing piles up in the
    scrollback, and the previous screen comes back on exit."""
    tty = sys.stdout.isatty()
    if tty:
        sys.stdout.write("\033[?1049h\033[?25l")   # alternate screen, hide cursor
    try:
        while True:
            screen = render_full(snapshot(), live=True)
            sys.stdout.write(("\033[H\033[J" if tty else "") + screen + "\n")
            sys.stdout.flush()
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        if tty:
            sys.stdout.write("\033[?25h\033[?1049l")  # show cursor, back to the normal screen
            sys.stdout.flush()


def wipe(deep: bool, confirmed: bool) -> str:
    what = ("EVERYTHING he has become — emotions, bonds, memory, self-portrait, secrets, "
            "dreams, journals, and every conversation (SOUL, keys and code are kept)") if deep else \
        "his emotional state — drives, hormones, temperament, entropy, bonds and the curves "\
        "(memory, self-portrait, secrets and journals are kept)"
    if not confirmed:
        return (yellow(f"This will erase {what}.\n") +
                "No copy is kept. Re-run with --yes to do it: "
                + bold("wm wipe --all --yes" if deep else "wm wipe --yes"))
    done = store.wipe(deep)
    seen, ordered = set(), []
    for item in done:
        if item not in seen:
            seen.add(item); ordered.append(item)
    tail = ("\n" + yellow("Now restart the gateway and clear the live chat:  "
                           "hermes gateway restart   then  /reset in Telegram")) if deep else ""
    return green(("Reborn." if deep else "Emotional slate wiped.") + " Erased: ") + ", ".join(ordered) + tail


def reopen_conversation() -> str:
    import datetime as _dt
    day = _dt.datetime.now(_dt.timezone.utc).date().isoformat()
    with store.locked_state() as (drives, _peers):
        drives["meta"]["talk_reopened_day"] = day
    used = store.conversation_tokens_today()
    return green(f"Conversation reopened for today. He answers again "
                 f"(spent {used:,}/{limits.CONVERSATION_DAILY_LIMIT:,} talking today).")


def main(args: List[str]) -> int:
    command = args[0] if args else ""
    if command == "live":
        live()
    elif command == "brain":
        from . import brainscan
        brainscan.live(snapshot)
    elif command == "dreams":
        n = int(physics.safe_float(args[1], 10)) if len(args) > 1 else 10
        print(render_dreams(n))
    elif command == "graph":
        hours = physics.safe_float(args[1], 48.0) if len(args) > 1 else 48.0
        print(render_graph(limits.clamp(hours, 1.0, 24.0 * 30)))
    elif command == "alerts":
        print(render_alerts())
    elif command == "ack":
        print(acknowledge(args[1:]))
    elif command == "forget" and len(args) == 2:
        print(forget(args[1]))
    elif command == "wipe":
        rest = set(args[1:])
        print(wipe("--all" in rest, "--yes" in rest))
    elif command == "talk":
        print(reopen_conversation())
    elif command in ("", "full"):
        print(render_full())
    else:
        print(__doc__)
        return 2
    return 0
