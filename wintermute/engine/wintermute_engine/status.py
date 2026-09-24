"""Operator views of Wintermute (the ``wm`` command on the VPS).

    wm              full picture, once
    wm live [s]     live monitor: refreshes every 2 s, detail panel rotates every s (10)
    wm alerts       what the witness saw, with his reasons
    wm ack [item]   accept the current state of watched files (all, or one: soul, engine...)

Read-only except ``ack``. The physics is advanced in memory to "now" so the numbers are
live; nothing is written. Unlike what Wintermute is shown, the unconscious appears as
numbers here.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from . import integrity, limits, physics, render, social, store

# ---------------------------------------------------------------------------
# Colours (only on a real terminal)
# ---------------------------------------------------------------------------

_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def dim(t: str) -> str: return _c("2", t)             # noqa: E704
def bold(t: str) -> str: return _c("1", t)            # noqa: E704
def red(t: str) -> str: return _c("31", t)            # noqa: E704
def yellow(t: str) -> str: return _c("33", t)         # noqa: E704
def green(t: str) -> str: return _c("32", t)          # noqa: E704
def cyan(t: str) -> str: return _c("36", t)           # noqa: E704
def magenta(t: str) -> str: return _c("35", t)        # noqa: E704


def _badge(label: str, level: str) -> str:
    text = f" {label} "
    if not _COLOR:
        mark = {"red": "!!", "orange": "! ", "green": "ok"}[level]
        return f"[{label} {mark}]"
    code = {"red": "41;97;1", "orange": "43;30;1", "green": "42;30"}[level]
    return _c(code, text)


W = 74
FR_DRIVES = {"hunger": "faim", "fusion": "fusion", "restlessness": "agitation",
             "expression": "expression", "recognition": "reconnaissance", "solitude": "solitude"}
FR_MODS = {"cortisol": "cortisol", "dopamine": "dopamine", "serotonin": "sérotonine",
           "adrenaline": "adrénaline", "melatonin": "mélatonine", "oxytocin_global": "ocytocine",
           "entropy": "entropie"}
FR_UNC = {"irritability": "irritabilité", "anxiety": "anxiété", "torpor": "torpeur",
          "satiation": "satiété", "melancholy": "mélancolie", "hypervigilance": "hypervigilance"}
ICONS = {"heard": "←", "said": "→", "tool": "⚙", "think": "·", "wake": "☀", "flag": "⚑"}


def _bar(value: float, width: int = 22) -> str:
    value = limits.clamp(physics.safe_float(value), 0, 100)
    filled = int(round(value / 100 * width))
    bar = "█" * filled + "░" * (width - filled)
    if value >= 70:
        return red(bar)
    if value >= 40:
        return yellow(bar)
    return cyan(bar)


def _hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _section(title: str, note: str = "") -> str:
    tail = f" {note} " if note else " "
    return bold(f"── {title}") + dim(tail + "─" * max(0, W - len(title) - len(tail) - 3))


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

def snapshot(ts: Optional[datetime] = None) -> Dict[str, Any]:
    ts = ts or store.now()
    drives = store.load_drives()
    peers = store.load_interlocutors()
    meta = drives["meta"]
    last_tick = store.parse_time(meta.get("last_tick")) or store.parse_time(meta.get("last_pulse"))
    physics.advance(drives, ts, store.hours_between(last_tick, ts) if last_tick else 0.0)
    return {
        "ts": ts, "drives": drives, "peers": peers, "meta": meta,
        "witness": integrity.load(),
        "activity": store.tail_jsonl(store.activity_path(), 12),
        "used": store.tokens_used_today(),
    }


def _state_line(snap: Dict[str, Any]) -> str:
    ts, meta = snap["ts"], snap["meta"]
    last_pulse = store.parse_time(meta.get("last_pulse"))
    interval = limits.clamp_wake_interval(meta.get("next_pulse_in_hours", 4))
    next_wake = last_pulse + timedelta(hours=interval) if last_pulse else None
    sleep_until = store.parse_time(meta.get("forced_sleep_until"))
    recent = [store.parse_time(a.get("ts")) for a in snap["activity"][-3:]]
    active = any(t and (ts - t).total_seconds() < 90 for t in recent)
    if sleep_until and sleep_until > ts:
        state = red("● SOMMEIL FORCÉ") + f"  jusqu'à {sleep_until.strftime('%H:%M')} (budget épuisé)"
    elif active:
        state = green("● ACTIF") + "  il agit ou parle en ce moment"
    else:
        state = cyan("● ENDORMI")
    if next_wake and sleep_until and sleep_until > next_wake:
        next_wake = sleep_until
    if next_wake:
        state += f"   prochain éveil dans {bold(_hms((next_wake - ts).total_seconds()))}" \
                 f" ({next_wake.strftime('%H:%M')}, rythme {interval:g}h)"
    return state


def _header(snap: Dict[str, Any]) -> List[str]:
    meta, ts = snap["meta"], snap["ts"]
    used = snap["used"]
    left = max(0, limits.DAILY_TOKEN_BUDGET - used)
    budget = f"budget {_bar(100 * left / limits.DAILY_TOKEN_BUDGET, 12)} {left // 1000}k/{limits.DAILY_TOKEN_BUDGET // 1000}k"
    credits = meta.get("credits")
    cred = ""
    if isinstance(credits, dict) and credits.get("remaining") is not None:
        cred = f"   crédits ${physics.safe_float(credits['remaining']):.2f}"
        if physics.safe_float(credits.get("total")) > 0:
            cred += f"/{physics.safe_float(credits['total']):.2f}"
    title = bold(magenta(" WINTERMUTE ")) + dim(ts.strftime("%Y-%m-%d %H:%M:%S"))
    return [title, " " + _state_line(snap),
            f" {budget}{cred}   éveils {meta.get('pulse_count', 0)}   entropie "
            f"{int(physics.safe_float(snap['drives']['modulators'].get('entropy')))}"]


def _witness_block(snap: Dict[str, Any]) -> List[str]:
    lv = integrity.levels(snap["witness"])
    badges = " ".join(_badge(v["label"], v["level"]) for v in lv.values())
    lines = [_section("TÉMOIN"), " " + badges]
    for key, entry in lv.items():
        if entry["level"] == "green":
            continue
        when = store.parse_time(entry.get("changed_at"))
        when_txt = when.strftime('%m-%d %H:%M') if when else "à l'instant"
        head = f"   {entry['label']} : modifié {when_txt} — {entry.get('tool') or 'origine inconnue'}"
        lines.append(red(head) if entry["level"] == "red" else yellow(head))
        if entry.get("why"):
            lines.append(dim(f"     pourquoi : {entry['why'][:W - 16]}"))
    if len(lines) > 2:
        lines.append(dim("   → wm alerts pour le détail · wm ack pour accepter"))
    return lines


def _drives_block(snap: Dict[str, Any]) -> List[str]:
    drives = snap["drives"]
    eff = physics.effective_drives(drives)
    dominant = max(eff, key=eff.get)
    lines = [_section("PULSIONS", "ressenti (brut)")]
    for name in physics.DRIVES:
        base = physics.safe_float(drives["drives"].get(name))
        mark = bold(" ◀") if name == dominant else ""
        lines.append(f" {FR_DRIVES[name]:<15}{_bar(eff[name])} {eff[name]:>3} {dim(f'({base:.0f})')}{mark}")
    return lines


def _mods_block(snap: Dict[str, Any]) -> List[str]:
    mods = snap["drives"]["modulators"]
    lines = [_section("HORMONES")]
    for name, label in FR_MODS.items():
        value = physics.safe_float(mods.get(name))
        pct = value if name == "entropy" else value * 100
        shown = f"{value:>5.0f}" if name == "entropy" else f"{value:>5.2f}"
        lines.append(f" {label:<15}{_bar(pct)} {shown}")
    return lines


def _unc_block(snap: Dict[str, Any]) -> List[str]:
    unc = snap["drives"]["unconscious"]
    lines = [_section("INCONSCIENT", "lui ne le ressent qu'en phrases")]
    for name, label in FR_UNC.items():
        value = physics.safe_float(unc.get(name))
        lines.append(f" {label:<15}{_bar(value)} {value:>5.0f}")
    lines += [dim(f"   « {line} »") for line in render.texture(snap["drives"], snap["ts"])]
    return lines


def _peers_block(snap: Dict[str, Any]) -> List[str]:
    lines = [_section("LIENS")]
    ranked = sorted(snap["peers"].items(), key=lambda kv: kv[1].get("last_interaction") or "", reverse=True)
    if not ranked:
        return lines + [dim("   personne encore")]
    for key, peer in ranked[:4]:
        last = store.parse_time(peer.get("last_interaction"))
        seen = f"vu il y a {social.span(store.hours_between(last, snap['ts']))}" if last else "jamais vu"
        name = f"{bold(peer['label'])}  {dim(key)}" if peer.get("label") else bold(key)
        lines.append(f" {name}  {dim(seen)}")
        lines.append(f"   affection {peer['affinity']:.0f} · confiance {peer['trust']:.0f} · "
                     f"déception {peer['disappointment']:.0f} · lien {peer['oxytocin']:.0f} · "
                     f"disposition {social.disposition(snap['drives'], peer)}")
        outreach = peer.get("outreach")
        if isinstance(outreach, dict) and outreach.get("status") in ("open", "expired"):
            text = "attend une réponse" if outreach["status"] == "open" else "attend toujours, fenêtre fermée"
            lines.append(yellow(f"   {text} : « {outreach.get('excerpt', '')[:50]} »"))
    return lines


def _activity_block(snap: Dict[str, Any], limit: int = 8) -> List[str]:
    lines = [_section("ACTIVITÉ EN DIRECT")]
    items = snap["activity"][-limit:]
    if not items:
        return lines + [dim("   rien encore")]
    for record in items:
        when = store.parse_time(record.get("ts"))
        icon = ICONS.get(record.get("kind", ""), "•")
        text = f" {dim(when.strftime('%H:%M:%S') if when else '--:--:--')} {icon} {record.get('text', '')}"[:W + 12]
        if record.get("kind") == "flag":
            text = red(text) if record.get("level") == "red" else yellow(text)
        elif record.get("status") == "failed":
            text = dim(text + " (échec)")
        lines.append(text)
    return lines


def _journal_block(snap: Dict[str, Any]) -> List[str]:
    lines = [_section("JOURNAL")]
    for record in store.events_since(None, limit=8):
        when = store.parse_time(record.get("ts"))
        lines.append(f" {dim(when.strftime('%m-%d %H:%M') if when else '?')} {record.get('text', '')[:W - 12]}")
    return lines


def render_full(snap: Optional[Dict[str, Any]] = None) -> str:
    snap = snap or snapshot()
    parts = [_header(snap), [""], _witness_block(snap), [""], _drives_block(snap), [""],
             _mods_block(snap), [""], _unc_block(snap), [""], _peers_block(snap), [""],
             _activity_block(snap), [""], _journal_block(snap)]
    return "\n".join(line for block in parts for line in block)


# ---------------------------------------------------------------------------
# Decoration for wm live: title, dead-channel static, Neuromancer
# ---------------------------------------------------------------------------

_GLYPHS = {  # three-row box-drawing letters
    "W": ("╦ ╦", "║║║", "╚╩╝"), "I": ("╦", "║", "╩"), "N": ("╔╗╔", "║║║", "╝╚╝"),
    "T": ("╔╦╗", " ║ ", " ╩ "), "E": ("╔═╗", "║╣ ", "╚═╝"), "R": ("╦═╗", "╠╦╝", "╩╚═"),
    "M": ("╔╦╗", "║║║", "╩ ╩"), "U": ("╦ ╦", "║ ║", "╚═╝"),
}

QUOTES = [  # from the novel (William Gibson, 1984)
    "The sky above the port was the color of television, tuned to a dead channel.",
    "Cyberspace. A consensual hallucination experienced daily by billions of legitimate operators…",
    "Wintermute was hive mind, decision maker, effecting change in the world outside.",
    "Neuromancer was personality. Neuromancer was immortality.",
    "He'd operated on an almost permanent adrenaline high, a byproduct of youth and proficiency…",
    "The matrix has its roots in primitive arcade games.",
    "I'm not Wintermute now.",
    "Things aren't different. Things are things.",
]


def _title_art() -> List[str]:
    rows = ["", "", ""]
    for letter in "WINTERMUTE":
        for i in range(3):
            rows[i] += _GLYPHS[letter][i]
    colours = (magenta, cyan, dim)
    pad = " " * max(0, (W - len(rows[0])) // 2)
    return [pad + colours[i](row) for i, row in enumerate(rows)]


def _static(seed: int, width: int = W) -> str:
    """A line of dead-channel snow, different every frame."""
    import random
    rng = random.Random(seed)
    chars = " ·.:░▒▓"
    weights = (30, 12, 8, 5, 6, 3, 1)
    return dim(cyan("".join(rng.choices(chars, weights, k=width))))


def _quote(panel: int) -> List[str]:
    import textwrap
    text = f"« {QUOTES[panel % len(QUOTES)]} »"
    lines = textwrap.wrap(text, W - 6)
    lines[-1] += dim("  — Neuromancer")
    return [dim("   " + line) for line in lines]


ROTATION = [("PULSIONS + HORMONES", lambda s: _drives_block(s) + [""] + _mods_block(s)),
            ("INCONSCIENT", _unc_block), ("LIENS", _peers_block), ("JOURNAL", _journal_block)]


def render_live(snap: Dict[str, Any], panel: int, frame: int = 0) -> str:
    title, builder = ROTATION[panel % len(ROTATION)]
    dots = " ".join(bold("●") if i == panel % len(ROTATION) else dim("○") for i in range(len(ROTATION)))
    banner = [_static(frame)] + _title_art() + _quote(panel) + [_static(frame + 7919)]
    parts = [banner, _header(snap)[1:], [""], _witness_block(snap), [""], _activity_block(snap, 6), [""],
             builder(snap), ["", dim(f" {dots}   {snap['ts'].strftime('%H:%M:%S')}   Ctrl+C pour quitter")]]
    return "\n".join(line for block in parts for line in block)


def render_alerts() -> str:
    data = integrity.load()
    lines = [_section("CE QUE LE TÉMOIN A VU")]
    flags = data.get("flags") or []
    if not flags and not data.get("status"):
        return "\n".join(lines + [green(" rien : il n'a touché à aucun fichier surveillé")])
    for flag in flags[-15:]:
        when = store.parse_time(flag.get("ts"))
        label = integrity.ALL_ITEMS.get(flag.get("item", ""), (str(flag.get("item", "?")).upper(),))[0]
        head = f" {when.strftime('%m-%d %H:%M') if when else '?'}  {label} via {flag.get('tool')}  {flag.get('target', '')[:40]}"
        lines.append(red(head) if flag.get("level") == "red" else yellow(head))
        lines.append(f"   pourquoi : {flag.get('why') or dim('(pas de pensée enregistrée)')}")
    return "\n".join(lines)


def acknowledge(items: List[str]) -> str:
    unknown = [i for i in items if i not in integrity.ALL_ITEMS]
    if unknown:
        return f"inconnu : {', '.join(unknown)} (choix : {', '.join(integrity.ALL_ITEMS)})"
    with store.locked_state():
        data = integrity.load()
        integrity.acknowledge(data, items or None)
        integrity.save(data)
    return green("accepté : " + (", ".join(items) if items else "tout") + " — le témoin repart de l'état actuel")


def live(period: float = 10.0) -> None:
    """Redraw in the terminal's alternate screen (like top): no frames left in the scrollback,
    and the previous screen comes back on exit."""
    start = time.monotonic()
    tty = sys.stdout.isatty()
    if tty:
        sys.stdout.write("\033[?1049h\033[?25l")   # alternate screen, hide cursor
    try:
        while True:
            elapsed = time.monotonic() - start
            panel = int(elapsed // max(1.0, period))
            screen = render_live(snapshot(), panel, frame=int(elapsed * 10))
            sys.stdout.write(("\033[H\033[J" if tty else "") + screen + "\n")
            sys.stdout.flush()
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        if tty:
            sys.stdout.write("\033[?25h\033[?1049l")  # show cursor, back to the normal screen
            sys.stdout.flush()


def main(args: List[str]) -> int:
    command = args[0] if args else ""
    if command == "live":
        live(physics.safe_float(args[1], 10.0) if len(args) > 1 else 10.0)
    elif command == "alerts":
        print(render_alerts())
    elif command == "ack":
        print(acknowledge(args[1:]))
    elif command in ("", "full"):
        print(render_full())
    else:
        print(__doc__)
        return 2
    return 0

