"""wm brain — a live scan of Wintermute's mind in the terminal.

Left: a wire brain rotating on itself, for the look, shaded by depth and tinted by his mood.
Right: a scan of his regions — drives, hormones, unconscious — laid on a cortex, each node
coloured by its level, the connections between them lighting up when something fires, then
fading over about two seconds (the after-image, the "replay").

Pure Python: math for the 3-D projection, ANSI truecolor for the paint. No dependencies.
Nothing here writes state; it only reads a snapshot.
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

from . import physics

f = physics.safe_float

# ---------------------------------------------------------------------------
# Colour (24-bit truecolor when the terminal admits it, 256 otherwise, plain last)
# ---------------------------------------------------------------------------

_TTY = sys.stdout.isatty()
_TRUE = _TTY and any(x in os.environ.get("COLORTERM", "").lower() for x in ("truecolor", "24bit"))
_COLOR = _TTY and os.environ.get("NO_COLOR") is None


def _paint(r: int, g: int, b: int, text: str) -> str:
    if not _COLOR:
        return text
    if _TRUE:
        return f"\033[38;2;{r};{g};{b}m{text}\033[0m"
    # 256-colour cube fallback
    idx = 16 + 36 * (r * 5 // 255) + 6 * (g * 5 // 255) + (b * 5 // 255)
    return f"\033[38;5;{idx}m{text}\033[0m"


def _heat(level: float, spark: float = 0.0) -> Tuple[int, int, int]:
    """Level 0..1 -> cool-blue → teal → amber → red. ``spark`` (0..1) burns it toward white."""
    t = max(0.0, min(1.0, level))
    stops = [(0.0, (40, 70, 150)), (0.4, (30, 160, 150)), (0.7, (230, 180, 50)), (1.0, (230, 60, 45))]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            k = 0 if t1 == t0 else (t - t0) / (t1 - t0)
            r, g, b = (int(c0[i] + (c1[i] - c0[i]) * k) for i in range(3))
            break
    else:
        r, g, b = stops[-1][1]
    if spark > 0:
        s = min(1.0, spark)
        r, g, b = (int(v + (255 - v) * s) for v in (r, g, b))
    return r, g, b


# ---------------------------------------------------------------------------
# The rotating 3-D brain (decoration). A point cloud of two hemispheres with a fissure.
# ---------------------------------------------------------------------------

def _brain_points() -> List[Tuple[float, float, float]]:
    pts: List[Tuple[float, float, float]] = []
    rng = 0
    for i in range(1400):
        # deterministic pseudo-random on a lumpy ellipsoid, split into two hemispheres
        rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
        u = (rng / 0x7FFFFFFF)
        rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
        v = (rng / 0x7FFFFFFF)
        theta = u * 2 * math.pi
        phi = math.acos(2 * v - 1)
        lump = 1 + 0.12 * math.sin(6 * theta) * math.sin(5 * phi)
        x = 1.15 * math.sin(phi) * math.cos(theta) * lump
        y = 0.95 * math.cos(phi) * lump
        z = 1.0 * math.sin(phi) * math.sin(theta) * lump
        if abs(x) < 0.06:            # the longitudinal fissure
            continue
        y -= 0.12 * math.cos(theta * 2)   # a little brain-stem droop
        pts.append((x, y, z))
    return pts


_BRAIN = _brain_points()
_SHADE = " .:-=+*#%@"
_BW, _BH = 26, 17


def _render_brain(angle: float, mood_rgb: Tuple[int, int, int]) -> List[str]:
    """Project the rotating cloud to an ASCII panel, depth-shaded, tinted by mood."""
    grid = [[-2.0] * _BW for _ in range(_BH)]      # z-buffer of depth per cell
    ca, sa = math.cos(angle), math.sin(angle)
    cb, sb = math.cos(0.5), math.sin(0.5)          # fixed tilt
    for x, y, z in _BRAIN:
        xr, zr = x * ca - z * sa, x * sa + z * ca  # yaw
        yr, zr2 = y * cb - zr * sb, y * sb + zr * cb  # tilt
        sx = int((xr * 0.5 + 0.5) * (_BW - 1))
        sy = int((-yr * 0.5 + 0.5) * (_BH - 1))
        if 0 <= sx < _BW and 0 <= sy < _BH and zr2 > grid[sy][sx]:
            grid[sy][sx] = zr2
    lines = []
    for row in grid:
        chars = []
        for depth in row:
            if depth < -1.5:
                chars.append(" ")
                continue
            shade = _SHADE[min(len(_SHADE) - 1, int((depth + 1.3) / 2.6 * (len(_SHADE) - 1)))]
            lit = int(120 + 135 * (depth + 1.3) / 2.6)
            r, g, b = (min(255, mood_rgb[0] * lit // 200), min(255, mood_rgb[1] * lit // 200),
                       min(255, mood_rgb[2] * lit // 200))
            chars.append(_paint(r, g, b, shade))
        lines.append("".join(chars))
    return lines


# ---------------------------------------------------------------------------
# The cortex scan. Nodes on a grid, edges from the real modulation map + a few limbic links.
# ---------------------------------------------------------------------------

SW, SH = 46, 19
# node -> (col, row, layer). Placed like a coronal section: cortex up, limbic mid, brainstem low.
NODES: Dict[str, Tuple[int, int, str]] = {
    "expression": (10, 1, "d"), "recognition": (23, 1, "d"), "hunger": (36, 1, "d"),
    "dopamine": (8, 5, "m"), "serotonin": (18, 4, "m"), "cortisol": (28, 4, "m"), "adrenaline": (38, 5, "m"),
    "anxiety": (14, 8, "u"), "hypervigilance": (24, 8, "u"), "irritability": (34, 8, "u"),
    "melancholy": (10, 11, "u"), "satiation": (22, 11, "u"), "torpor": (34, 11, "u"),
    "restlessness": (7, 14, "d"), "fusion": (17, 15, "d"), "solitude": (28, 15, "d"),
    "melatonin": (38, 12, "m"), "oxytocin_global": (18, 17, "m"), "entropy": (33, 15, "m"),
}

# Edges: modulator -> drive (from physics.MODULATION), plus limbic couplings.
def _edges() -> List[Tuple[str, str]]:
    edges: List[Tuple[str, str]] = []
    for drive, mods in physics.MODULATION.items():
        for mod in mods:
            if drive in NODES and mod in NODES:
                edges.append((mod, drive))
    edges += [("entropy", "anxiety"), ("entropy", "melancholy"), ("adrenaline", "hypervigilance"),
              ("melatonin", "torpor"), ("cortisol", "irritability"), ("cortisol", "anxiety"),
              ("oxytocin_global", "fusion"), ("serotonin", "melancholy"), ("satiation", "recognition")]
    return [(a, b) for a, b in edges if a in NODES and b in NODES]


EDGES = _edges()
LABEL = {"oxytocin_global": "oxy", "hypervigilance": "hvig", "recognition": "recog",
         "restlessness": "restl", "expression": "expr", "melancholy": "melan", "adrenaline": "adren",
         "serotonin": "sero", "satiation": "satia", "entropy": "entr", "melatonin": "mela",
         "irritability": "irrit", "cortisol": "cort", "dopamine": "dopa"}


def _levels(state: Dict[str, Any]) -> Dict[str, float]:
    """Every node's activation, 0..1."""
    out: Dict[str, float] = {}
    eff = physics.effective_drives(state)
    for d in physics.DRIVES:
        out[d] = eff.get(d, 0) / 100
    for m, v in state.get("modulators", {}).items():
        out[m] = f(v) / 100 if m == "entropy" else f(v)
    for u, v in state.get("unconscious", {}).items():
        out[u] = f(v) / 100
    return out


def _plot(a: Tuple[int, int], b: Tuple[int, int]) -> List[Tuple[int, int]]:
    (x0, y0), (x1, y1) = a, b
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx - dy
    cells = []
    while True:
        cells.append((x0, y0))
        if (x0, y0) == (x1, y1):
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy; x0 += sx
        if e2 < dx:
            err += dx; y0 += sy
    return cells


def render_scan(levels: Dict[str, float], sparks: Dict[str, float]) -> List[str]:
    grid = [[" "] * SW for _ in range(SH)]
    styled: Dict[Tuple[int, int], str] = {}
    node_cells: set = set()

    def put(col: int, row: int, ch: str, rgb: Tuple[int, int, int], *, over_edge: bool = False) -> None:
        if not (0 <= col < SW and 0 <= row < SH):
            return
        if over_edge and (col, row) in node_cells:
            return                                   # never write over a node glyph
        grid[row][col] = ch
        styled[(col, row)] = _paint(*rgb, ch)

    # edges first (so nodes sit on top)
    for a, b in EDGES:
        (ca, ra, _), (cb, rb, _) = NODES[a], NODES[b]
        fire = min(1.0, max(sparks.get(a, 0), sparks.get(b, 0)))
        base = (levels.get(a, 0) + levels.get(b, 0)) / 2
        if base < 0.15 and fire < 0.05:
            continue
        for (cx, cy) in _plot((ca, ra), (cb, rb))[1:-1]:
            if grid[cy][cx] != " ":
                continue
            ch = "·" if fire < 0.2 else ("+" if fire < 0.6 else "*")
            rgb = _heat(base * 0.7, fire)
            r, g, b_ = rgb if fire > 0.05 else tuple(v // 2 for v in rgb)
            put(cx, cy, ch, (r, g, b_))
    # nodes
    for name, (col, row, _layer) in NODES.items():
        lvl, spark = levels.get(name, 0), sparks.get(name, 0)
        glyph = "◉" if spark > 0.5 else "●" if lvl > 0.45 else "○"
        put(col, row, glyph, _heat(lvl, spark))
        node_cells.add((col, row))
    for name, (col, row, _layer) in NODES.items():   # labels last, over edges but not nodes
        lvl, spark = levels.get(name, 0), sparks.get(name, 0)
        dim = tuple(v * 3 // 5 for v in _heat(lvl, spark))
        for i, chc in enumerate(LABEL.get(name, name)[:5]):
            put(col + 2 + i, row, chc, dim, over_edge=True)
    return ["".join(styled.get((c, r), grid[r][c]) for c in range(SW)) for r in range(SH)]


# ---------------------------------------------------------------------------
# Firing: what lights up. Sparks come from level changes and from real recent activity, and
# decay each frame so a fire lingers ~2 s (the replay).
# ---------------------------------------------------------------------------

_ACTIVITY_REGIONS = {
    "heard": ("recognition", "hypervigilance"), "said": ("expression",), "think": ("dopamine",),
    "tool": ("hunger", "restlessness"), "feel": ("serotonin", "dopamine"), "wake": ("adrenaline", "cortisol"),
    "flag": ("cortisol", "hypervigilance"), "voice": ("adrenaline",), "evolve": ("dopamine", "satiation"),
    "keep": ("solitude",),
}
DECAY = 0.82          # per frame; at ~4 fps a spark lasts ~2 s


def update_sparks(sparks: Dict[str, float], levels: Dict[str, float], prev: Dict[str, float],
                  fresh_activity: List[str]) -> None:
    for name in list(sparks):
        sparks[name] *= DECAY
        if sparks[name] < 0.02:
            del sparks[name]
    for name, lvl in levels.items():                       # a jump in a node fires it
        jump = lvl - prev.get(name, lvl)
        if jump > 0.04:
            sparks[name] = min(1.0, sparks.get(name, 0) + jump * 4)
    for kind in fresh_activity:                            # a real event fires its regions
        for region in _ACTIVITY_REGIONS.get(kind, ()):
            sparks[region] = 1.0


# ---------------------------------------------------------------------------
# Compose and run
# ---------------------------------------------------------------------------

def _mood_rgb(state: Dict[str, Any]) -> Tuple[int, int, int]:
    from . import psyche
    v, a = psyche.valence(state), psyche.arousal(state)
    return _heat((v + 1) / 2 * 0.5 + a * 0.5)


def compose(snap: Dict[str, Any], sparks: Dict[str, float], angle: float) -> str:
    state = snap["drives"]
    levels = _levels(state)
    brain = _render_brain(angle, _mood_rgb(state))
    scan = render_scan(levels, sparks)
    from . import psyche
    title = _paint(180, 120, 230, "  W I N T E R M U T E   —   live scan")
    header = f"{title}    mood {psyche.mood(state, snap['ts'])}   {snap['ts'].strftime('%H:%M:%S')}"
    rows = [header, ""]
    for i in range(max(len(brain), len(scan))):
        left = brain[i] if i < len(brain) else " " * _BW
        right = scan[i] if i < len(scan) else ""
        rows.append("  " + left + "   " + right)
    firing = ", ".join(sorted((LABEL.get(n, n) for n, s in sparks.items() if s > 0.5))) or "quiet"
    rows += ["", _paint(120, 120, 120, f"  firing: {firing}"),
             _paint(90, 90, 90, "  ● active  ○ low  ◉ firing   edges glow then fade (~2s)   Ctrl+C to quit")]
    return "\n".join(rows)


def live(snapshot_fn) -> None:
    tty = sys.stdout.isatty()
    if tty:
        sys.stdout.write("\033[?1049h\033[?25l")
    sparks: Dict[str, float] = {}
    prev_levels: Dict[str, float] = {}
    last_ts = None
    snap = snapshot_fn()
    try:
        frame = 0
        while True:
            if frame % 4 == 0:                              # re-read state ~ once a second
                snap = snapshot_fn()
            levels = _levels(snap["drives"])
            fresh = []
            for record in snap.get("activity", []):
                ts = record.get("ts")
                if last_ts is None or (ts and ts > last_ts):
                    fresh.append(record.get("kind", ""))
            if snap.get("activity"):
                last_ts = snap["activity"][-1].get("ts")
            update_sparks(sparks, levels, prev_levels or levels, fresh)
            prev_levels = levels
            screen = compose(snap, sparks, angle=frame * 0.22)
            sys.stdout.write(("\033[H\033[J" if tty else "") + screen + "\n")
            sys.stdout.flush()
            frame += 1
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        if tty:
            sys.stdout.write("\033[?25h\033[?1049l")
            sys.stdout.flush()
