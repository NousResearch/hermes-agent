#!/usr/bin/env python3
"""Behavioural canaries for the fork's [LOCAL] patches.

WHY THIS FILE EXISTS
--------------------
This fork carries ~22 [LOCAL] patches re-derived onto each new upstream base.
Twice now a patch has died silently while its code stayed present and the suite
stayed green:

  * 8c466c3971 exists solely to "restore local mods dropped by fleet-to-main
    merge" -- a merge ate local work and nothing noticed.
  * Patch L (78c30ad11f) reworded the Telegram observe prompt but left
    _TELEGRAM_OBSERVED_CONTEXT_PROMPT_MARKER on the old string, so the gate it
    feeds returned False from 2026-06-13 until 2026-08-21. Compile-clean,
    import-clean, and the tests that covered it were red and being ignored.

So these tests assert the patch's EFFECT, never merely that its code or its
config flag is present. A canary that checks "the function exists" would have
passed throughout both incidents.

Standalone-runnable, like the rest of tests/: `PYTHONPATH=. python <file>`.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import tools.file_tools as ft  # noqa: E402

P = 0
FAILED = []


def check(name, condition):
    global P
    if condition:
        P += 1
        print(f"  ok  {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL {name}")


def _first_content_line(payload: dict) -> str:
    """First rendered line of a read result, used as a position fingerprint."""
    return (payload.get("content") or "").splitlines()[0] if payload.get("content") else ""


# ── Patch J: truncate-and-continue + auto-advance ────────────────────
#
# On an oversized read the model gets the head plus a next_offset. The LOCAL
# half is the cursor: a REPEATED default/top read must serve the NEXT window
# rather than the same head, or a weak model loops on the identical call
# forever.
#
# The historical break: upstream's truncation path stopped returning early and
# began falling through to the cursor-clearing block, so the cursor recorded
# for a truncated read was erased in the same call and auto-advance never
# fired. Nothing crashed; reads just silently looped.
def patch_j_canary():
    max_chars = ft._read_max_chars() if hasattr(ft, "_read_max_chars") else 200_000
    with tempfile.TemporaryDirectory() as td:
        big = Path(td) / "oversized.log"
        # Long lines so we blow the CHAR budget well before the line limit.
        line = "x" * 2_000
        needed = (max_chars // len(line)) * 3
        big.write_text("\n".join(f"{i:06d} {line}" for i in range(needed)))

        task = "canary-patch-j"
        ft._read_tracker.pop(task, None)

        first = json.loads(ft.read_file_tool(str(big), task_id=task))
        check("patch J: an oversized read is truncated rather than refused",
              first.get("truncated") is True and bool(first.get("content")))
        check("patch J: truncation offers a concrete next_offset",
              isinstance(first.get("next_offset"), int) and first["next_offset"] > 1)

        second = json.loads(ft.read_file_tool(str(big), task_id=task))
        # THE canary: same call, same default offset -> must not re-serve the head.
        check("patch J: a repeated top read AUTO-ADVANCES instead of looping",
              _first_content_line(second) != _first_content_line(first))

        third = json.loads(ft.read_file_tool(str(big), task_id=task))
        check("patch J: it keeps advancing, not just once",
              _first_content_line(third) != _first_content_line(second))


# ── fast_shell: fork-only tool, must stay registered ─────────────────
#
# 5ce895a23a registers fast_shell into the `terminal` toolset. It exists in no
# upstream release, so a re-derive that drops the registration leaves the tool
# file on disk and simply stops offering it to the model.
def fast_shell_canary():
    # Assert through the path the model actually sees -- a schema handed to the
    # LLM -- rather than by inspecting registry internals. "The file is on disk"
    # and "the tool is offered" are exactly the two things that came apart in
    # the incidents above.
    from model_tools import get_tool_definitions

    terminal = {
        (d.get("function") or {}).get("name") or d.get("name")
        for d in get_tool_definitions(enabled_toolsets=["terminal"], quiet_mode=True)
    }
    check("fast_shell: the fork-only tool is OFFERED to the model via `terminal`",
          "fast_shell" in terminal)

    without = {
        (d.get("function") or {}).get("name") or d.get("name")
        for d in get_tool_definitions(enabled_toolsets=["file"], quiet_mode=True)
    }
    check("fast_shell: and it is scoped to `terminal`, not leaking into `file`",
          "fast_shell" not in without)


# ── Patch L: the observe gate must actually fire ─────────────────────
#
# The 2026-06→08 incident in one assertion: the marker the run path gates on
# must be a substring of the prompt the Telegram adapter really builds. Held
# here as well as in tests/gateway/test_telegram_group_gating.py because this
# is the file someone reads when re-deriving patches.
def patch_l_canary():
    import re
    from gateway.run import _uses_telegram_observed_group_context

    src = Path(__file__).resolve().parent.parent.parent / "plugins/platforms/telegram/adapter.py"
    body = src.read_text(encoding="utf-8")
    m = re.search(
        r'def _telegram_group_observe_channel_prompt\(self\).*?return \((.*?)\n        \)',
        body, re.S)
    check("patch L: the observe channel prompt builder is still present", m is not None)
    if not m:
        return
    prompt = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))).replace("\\n", "\n")
    check("patch L: the run-path gate FIRES on the prompt the adapter really builds",
          _uses_telegram_observed_group_context(prompt) is True)


for fn in (patch_j_canary, fast_shell_canary, patch_l_canary):
    print(f"\n{fn.__name__}:")
    fn()

print()
if FAILED:
    print(f"FAILED {len(FAILED)} canary check(s): {FAILED}")
    sys.exit(1)
print(f"ALL {P} CANARY CHECKS PASSED — [LOCAL] patch effects still observable")
