"""Screenshot parsing preserves legacy paths without repeated suffix scans."""

import subprocess
import sys

import pytest

from tools import browser_use_cli as bu_cli


@pytest.mark.parametrize("stdout, expected", [
    ("saved /tmp/first.png then '/tmp/last.webp'", "/tmp/last.webp"),
    (r"saved C:\screens\first.PNG and D:\screens\last.JPEG", r"D:\screens\last.JPEG"),
    ("prefix=(/tmp/first.png/nonexistent/last.webp", "/tmp/first.png"),
    ("saved /tmp/first.png then /nonexistent/missing.jpg", "/tmp/first.png"),
    ("relative.png and no screenshot", None),
    ("", None),
    ("/" + "a/" * 400 + "long.png", "/" + "a/" * 400 + "long.png"),
    ("/" + "a/" * 1100 + "long.png", "/" + "a/" * 1100 + "long.png"),
])
def test_screenshot_discovery_preserves_path_order_and_matching(monkeypatch, stdout, expected):
    existing = {"/tmp/first.png", "/tmp/last.webp", r"C:\screens\first.PNG", r"D:\screens\last.JPEG", expected}
    monkeypatch.setattr(bu_cli.os.path, "isfile", lambda path: path in existing)
    monkeypatch.setattr(bu_cli.os.path, "getmtime", lambda path: 100)
    assert bu_cli._find_screenshot(stdout, since=100) == expected


@pytest.mark.parametrize("letter", ["\u0130", "\u0131", "\u017f", "\u212a"])
@pytest.mark.parametrize("separator", ["/", "\\"])
def test_unicode_casefold_drive_prefix_preserves_legacy_path(monkeypatch, letter, separator):
    # Python's legacy IGNORECASE character class also accepts these Unicode letters.
    path = letter + ":" + separator + "screens" + separator + "shot.png"
    assert bu_cli._IMAGE_PATH_RE.findall(path) == [path]
    monkeypatch.setattr(bu_cli.os.path, "isfile", lambda candidate: candidate == path)
    monkeypatch.setattr(bu_cli.os.path, "getmtime", lambda candidate: 100)
    assert bu_cli._find_screenshot(path, since=100) == path


@pytest.mark.parametrize("size", [120_000, 1_000_000])
def test_slash_heavy_output_does_not_block_screenshot_discovery(size):
    # A GIL-holding regex can block thread-based timeout enforcement.
    code = (
        "import time\n"
        "from tools.browser_use_cli import _find_screenshot\n"
        f"assert _find_screenshot('/' * {size}, time.time()) is None\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=10)
