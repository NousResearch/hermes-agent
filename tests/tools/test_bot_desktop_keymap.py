"""The Latin keymap step of launcher.sh (é, ñ, ß, €… placed on spare media keys at screen start).

The awk program is extracted from the launcher itself and run on a trimmed ``xkbcomp -xkb`` dump, so
the test exercises the shipped code, not a copy.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[2] / "tools" / "bot_desktop" / "launcher.sh"

DUMP = """xkb_keymap {
xkb_keycodes "evdev+aliases(qwerty)" {
    <AC01> = 38;
};
xkb_symbols "pc+us+inet(evdev)" {
    name[Group1]="English (US)";
    key <AC01> {         [               a,               A ] };
    key <SPCE> {         [           space ] };
%s
    key <I999> {         [   XF86AudioPlay,       Next ] };
};
};
""" % "\n".join(f"    key <I{100 + i}> {{         [     XF86Launch{i:X} ] }};" for i in range(60))


def _run(dump: str) -> str:
    text = LAUNCHER.read_text()
    table = re.search(r"^HERMES_BD_LATIN_KEYSYMS='.*?'$", text, re.S | re.M).group(0)
    func = re.search(r"# BEGIN latin-keymap.*?# END latin-keymap", text, re.S).group(0)
    script = f"{table}\n{func}\nbd_latin_keymap\n"
    return subprocess.run(["bash", "-c", script], input=dump, capture_output=True, text=True, check=True).stdout


def _pairs() -> list[str]:
    table = re.search(r"^HERMES_BD_LATIN_KEYSYMS='(.*?)'$", LAUNCHER.read_text(), re.S | re.M).group(1)
    return table.split()


@pytest.mark.skipif(shutil.which("awk") is None, reason="awk not installed")
def test_latin_characters_land_on_media_only_keys():
    out = _run(DUMP)
    for pair in _pairs():
        assert "[ " + pair.replace(":", ", ") + " ]" in out, pair
    assert "key <I100> { [ agrave, Agrave ] };" in out
    # Ordinary keys and a key mixing a media keysym with a real one are left alone.
    assert "[               a,               A ]" in out
    assert "[           space ]" in out
    assert "XF86AudioPlay,       Next" in out
    assert out.count("key <") == DUMP.count("key <")
    # Every placed character is a distinct keysym: no key is assigned twice.
    placed = re.findall(r"key <\w+> \{ \[ ([^]]+) \] \};", out)
    assert len(placed) == len(_pairs()) == len(set(placed))


@pytest.mark.skipif(shutil.which("awk") is None, reason="awk not installed")
def test_short_map_places_what_fits_and_keeps_the_dump_valid():
    few = DUMP.replace("\n".join(f"    key <I{100 + i}> {{         [     XF86Launch{i:X} ] }};" for i in range(60)),
                       "    key <I100> {         [     XF86Launch0 ] };")
    out = _run(few)
    assert "key <I100> { [ agrave, Agrave ] };" in out
    assert out.count("key <") == few.count("key <")
    assert out.rstrip().endswith("};")
