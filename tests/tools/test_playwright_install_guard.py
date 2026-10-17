"""The playwright guard: browser-driver binary downloads (playwright/puppeteer
install) require approval — they write multi-hundred-MB browser builds into
the user cache and are almost always agent improvisation, not operator intent."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tools.approval_detection import detect_dangerous_command


def test_playwright_install_variants_are_flagged():
    for cmd in (
        "playwright install chromium",
        "python3 -m playwright install chromium",
        "npx playwright install chromium",
        "python3 -m playwright install chromium --with-deps",
        "~/venv/bin/playwright install",
        "cd /app && playwright install chromium",
    ):
        flagged, _, desc = detect_dangerous_command(cmd)
        assert flagged, f"not flagged: {cmd}"
        assert "browser-driver" in (desc or "")


def test_innocent_neighbours_are_not_flagged():
    for cmd in (
        "pip install requests",
        "ls ~/.cache/ms-playwright",
        "docker run camofox",
        "git commit -m 'npm uninstall docs'",
    ):
        flagged, _, _ = detect_dangerous_command(cmd)
        assert not flagged, f"false positive: {cmd}"
