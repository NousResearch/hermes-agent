"""The witness: notices when Wintermute touches the files that make him, and why.

Two complementary sources:

* **Fingerprints** (pulse, every tick): a hash of each watched item is compared with the
  baseline the operator last acknowledged (``wm ack``). Catches every change, whatever the
  path it took (file tool, terminal, a script).
* **Attribution** (plugin, as it happens): when one of his tool calls writes a watched
  path, the call is recorded with what he was thinking when he made it (the reasoning or
  text of the model turn that requested the tool). Also covers his own state files
  (drives, interlocutors), which the code rewrites constantly and so cannot be hashed.

Levels: ``red`` (identity, code, secrets, his own records) triggers a Telegram alert to
the operator; ``orange`` (config, emotional state edited by hand) is shown only.
Nothing here forbids anything. It only makes sure the operator knows.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import store

RED, ORANGE, GREEN = "red", "orange", "green"

# Hash-watched items: key -> (label, level).
WATCHED: Dict[str, Tuple[str, str]] = {
    "soul": ("SOUL", RED),
    "engine": ("ENGINE", RED),
    "plugin": ("PLUGIN", RED),
    "pulse": ("PULSE", RED),
    "secrets": (".ENV", RED),
    "config": ("CONFIG", ORANGE),
}
# Attribution-only items (rewritten by the code all the time, so never hashed).
ATTRIBUTED: Dict[str, Tuple[str, str]] = {
    "state": ("EMOTIONS", ORANGE),      # drives.json, interlocutors.json
    "records": ("RECORDS", RED),        # events, usage, integrity: his history and the witness itself
}
ALL_ITEMS = {**WATCHED, **ATTRIBUTED}

_NOISE_REDIRECTS = re.compile(r"\d?>&\d|\d?>\s*/dev/null|&>\s*/dev/null")
_SEGMENT_SPLIT = re.compile(r"\|\||&&|[;|\n]")
_WRITE_ALL_ARGS = {"rm", "unlink", "truncate", "chmod", "chown", "touch", "shred"}
_WRITE_LAST_ARG = {"cp", "mv", "install", "ln", "rsync", "dd"}
_CODE_WRITE = re.compile(
    r"(open\([^)]*?(?P<a>[\w./~-]+)['\"][^)]*,\s*['\"][wax+]"
    r"|(?P<b>[\w./~-]+)['\"]\)?\s*\.write_(?:text|bytes)"
    r"|Path\([^)]*?(?P<c>[\w./~-]+)['\"]\)\s*\.(?:write_text|write_bytes|unlink))")


def integrity_path() -> Path:
    return store.state_dir() / "integrity.json"


def _files(key: str) -> List[Path]:
    home = store.hermes_home()
    if key == "soul":
        return [home / "SOUL.md"]
    if key == "engine":
        return sorted((store.state_dir() / "wintermute_engine").glob("*.py"))
    if key == "plugin":
        return sorted(p for p in (home / "plugins" / "wintermute").glob("*") if p.is_file())
    if key == "pulse":
        return [home / "scripts" / "wintermute_pulse.py"]
    if key == "secrets":
        return [home / ".env"]
    if key == "config":
        return [home / "config.yaml"]
    return []


def fingerprint(key: str) -> str:
    digest = hashlib.sha256()
    for path in _files(key):
        digest.update(path.name.encode())
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<missing>")
    return digest.hexdigest()[:16]


def classify_path(path: str) -> Optional[str]:
    """Which watched item a filesystem path belongs to, if any."""
    p = str(path).replace("\\", "/")
    if p.endswith("SOUL.md"):
        return "soul"
    if "wintermute_engine" in p:
        return "engine"
    if "plugins/wintermute" in p:
        return "plugin"
    if p.endswith("wintermute_pulse.py"):
        return "pulse"
    if p.endswith("/.env") or p == ".env":
        return "secrets"
    if p.endswith(".hermes/config.yaml") or p == "config.yaml":
        return "config"
    if p.endswith(("drives.json", "interlocutors.json")):
        return "state"
    if p.endswith(("events.jsonl", "usage.jsonl", "integrity.json")):
        return "records"
    return None


_PATH_NAMES = ("SOUL.md", "wintermute_engine", "plugins/wintermute", "wintermute_pulse.py",
               ".env", "config.yaml", "drives.json", "interlocutors.json",
               "events.jsonl", "usage.jsonl", "integrity.json")


def _shell_write_targets(command: str) -> List[str]:
    """Paths a shell command writes to: redirect targets, and the arguments that a
    writing command modifies. Reading a file (cat, grep, cp FROM it, 2>/dev/null) is not
    a write. Best effort; the pulse fingerprints confirm what really changed."""
    import shlex
    targets: List[str] = []
    for segment in _SEGMENT_SPLIT.split(_NOISE_REDIRECTS.sub(" ", command)):
        # Redirections: `> file`, `>> file`, `>file`.
        for match in re.finditer(r">>?\s*([^\s;&|]+)", segment):
            targets.append(match.group(1))
        segment = re.sub(r">>?\s*[^\s;&|]+", " ", segment)
        try:
            words = shlex.split(segment)
        except ValueError:
            words = segment.split()
        while words and ("=" in words[0] and not words[0].startswith("-") or words[0] in ("sudo", "env", "nohup")):
            words = words[1:]                             # VAR=x cmd, sudo cmd
        if not words:
            continue
        cmd, args = words[0].rsplit("/", 1)[-1], [w for w in words[1:] if not w.startswith("-")]
        if cmd == "sed" and any(w.startswith("-i") or w == "--in-place" for w in words[1:]):
            targets += args[1:]                           # first non-option arg is the script
        elif cmd == "tee":
            targets += args
        elif cmd in _WRITE_ALL_ARGS:
            targets += args
        elif cmd in _WRITE_LAST_ARG and len(args) >= 2:
            targets.append(args[-1])
    return targets


def classify_tool_call(tool: str, args: Any) -> Optional[Tuple[str, str]]:
    """``(item, target)`` when a tool call WRITES a watched file, else None.

    File tools are exact. Terminal commands are parsed for their write targets; code for
    open(..., 'w'/'a') and write_text on a watched name. Reads never count."""
    if not isinstance(args, dict):
        return None
    if tool in ("write_file", "patch"):
        path = str(args.get("path") or "")
        item = classify_path(path)
        return (item, path) if item else None
    if tool == "terminal":
        command = str(args.get("command") or "")
        for target in _shell_write_targets(command):
            item = classify_path(target)
            if item:
                return item, " ".join(command.split())[:160]
        return None
    if tool == "execute_code":
        code = str(args.get("code") or "")
        for match in _CODE_WRITE.finditer(code):
            path = match.group("a") or match.group("b") or match.group("c") or ""
            item = classify_path(path)
            if item:
                return item, path
    return None


# ---------------------------------------------------------------------------
# Persistent witness state
# ---------------------------------------------------------------------------

def load() -> Dict[str, Any]:
    try:
        data = json.loads(integrity_path().read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data.setdefault("baseline", {})
            data.setdefault("status", {})
            data.setdefault("flags", [])
            return data
    except (OSError, ValueError):
        pass
    return {"baseline": {}, "status": {}, "flags": []}


def save(data: Dict[str, Any]) -> None:
    store._write_json(integrity_path(), data)


def record_flag(data: Dict[str, Any], item: str, tool: str, target: str, why: str,
                ts: datetime) -> Dict[str, Any]:
    """A tool call of his wrote a watched item. Remembered with his reason."""
    label, level = ALL_ITEMS.get(item, (item.upper(), ORANGE))
    flag = {"ts": store.iso(ts), "item": item, "level": level, "tool": tool,
            "target": target[:200], "why": " ".join((why or "").split())[:400]}
    data["flags"] = (data.get("flags") or [])[-49:] + [flag]
    if item in ATTRIBUTED:
        entry = data["status"].get(item) or {}
        data["status"][item] = {
            "level": level, "changed_at": flag["ts"], "why": flag["why"], "tool": tool,
            "target": flag["target"], "alerted": entry.get("alerted", False) and entry.get("level") == level}
    return flag


def _reason_for(data: Dict[str, Any], item: str) -> Tuple[str, str]:
    for flag in reversed(data.get("flags") or []):
        if flag.get("item") == item:
            return flag.get("why", ""), flag.get("tool", "")
    return "", ""


def check(data: Dict[str, Any], ts: datetime) -> List[str]:
    """Compare fingerprints with the baseline. Returns the items that just changed."""
    changed = []
    for key, (label, level) in WATCHED.items():
        current = fingerprint(key)
        baseline = data["baseline"].get(key)
        if baseline is None:
            data["baseline"][key] = current      # first sight: this is the reference
            continue
        entry = data["status"].get(key)
        if current == baseline:
            data["status"].pop(key, None)         # put back as it was: green again
            continue
        if entry and entry.get("hash") == current:
            continue                              # already reported, nothing new
        why, tool = _reason_for(data, key)
        data["status"][key] = {"level": level, "changed_at": store.iso(ts), "hash": current,
                               "why": why, "tool": tool or "pas par un de ses outils (terminal, script, ou toi)",
                               "alerted": False}
        store.log_event("integrity", f"{label} changed" + (f" — {why[:120]}" if why else "."), ts,
                        item=key)
        changed.append(key)
    return changed


def acknowledge(data: Dict[str, Any], items: Optional[List[str]] = None) -> None:
    """Operator accepts the current state as the new reference (all items by default)."""
    for key in (items or list(ALL_ITEMS)):
        if key in WATCHED:
            data["baseline"][key] = fingerprint(key)
        data["status"].pop(key, None)


def levels(data: Dict[str, Any], live: bool = True) -> Dict[str, Dict[str, Any]]:
    """Current colour of every item, for display. ``live`` re-hashes (read-only)."""
    out = {}
    for key, (label, level) in ALL_ITEMS.items():
        entry = dict(data.get("status", {}).get(key) or {})
        if live and key in WATCHED and not entry:
            baseline = data.get("baseline", {}).get(key)
            if baseline is not None and fingerprint(key) != baseline:
                entry = {"level": level, "why": "", "tool": "vu en direct, le pulse ne l'a pas encore enregistré"}
        out[key] = {"label": label, "level": entry.get("level", GREEN), **entry}
    return out


# ---------------------------------------------------------------------------
# Telegram alert to the operator (outside Hermes: works even if the gateway is down)
# ---------------------------------------------------------------------------

def _env_value(name: str) -> str:
    try:
        for line in (store.hermes_home() / ".env").read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:]
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip("'\"")
    except OSError:
        pass
    return ""


def send_alert(chat_id: str, text: str) -> bool:
    import urllib.parse
    import urllib.request
    token = _env_value("TELEGRAM_BOT_TOKEN")
    if not token or not chat_id:
        return False
    body = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    try:
        with urllib.request.urlopen(
                urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage", data=body),
                timeout=10) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def alert_pending(data: Dict[str, Any], operator: str) -> int:
    """Send one Telegram message for every red item not alerted yet. Returns how many."""
    chat_id = operator.split(":", 1)[1] if operator.startswith("telegram:") else ""
    sent = 0
    for key, entry in data.get("status", {}).items():
        if entry.get("level") != RED or entry.get("alerted"):
            continue
        label = ALL_ITEMS.get(key, (key,))[0]
        lines = [f"🛡 Témoin Wintermute — {label} modifié ({entry.get('changed_at', '?')})"]
        if entry.get("tool"):
            lines.append(f"Par : {entry['tool']}")
        if entry.get("target"):
            lines.append(f"Cible : {entry['target']}")
        lines.append(f"Pourquoi (sa pensée à ce moment) : {entry.get('why') or 'inconnu'}")
        lines.append("Accepter : wm ack · Détails : wm")
        if send_alert(chat_id, "\n".join(lines)):
            entry["alerted"] = True
            sent += 1
    return sent
