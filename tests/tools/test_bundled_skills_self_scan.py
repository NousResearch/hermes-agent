"""Bundled skills must stay clean under Hermes' own scanners.

Stock skills ship in every install, so a SKILL.md whose curl example matches
``exfil_curl`` is a self-inflicted outage: the shared threat-pattern library
flags it at context scope (#98474) and Skills Guard refuses the very same
file on a Hub install (#91569). These anchors fail closed — re-introducing
a credential-interpolating one-liner into the bundled file turns them red.
"""

from pathlib import Path

from tools.skills_guard import scan_skill, should_allow_install
from tools.threat_patterns import scan_for_threats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUNDLED_SKILL = PROJECT_ROOT / "skills" / "web" / "blocked-page-recovery"


def test_bundled_blocked_page_recovery_passes_context_scan():
    """The bundled SKILL.md must not trip the shared context-scope scanner.

    The Jina Reader example used to interpolate ``$JINA_API_KEY`` on the same
    line as ``curl``, matching ``exfil_curl`` (tools/threat_patterns.py) on
    every fresh install (#98474). Moving the header into its own variable
    line clears the pattern while keeping the command functional.
    """
    assert BUNDLED_SKILL.is_dir(), f"bundled skill moved or missing: {BUNDLED_SKILL}"
    text = (BUNDLED_SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert scan_for_threats(text, scope="context") == []


def test_bundled_blocked_page_recovery_installable_from_hub():
    """Skills Guard must allow reinstalling the bundled skill from the Hub.

    The same ``exfil_curl`` line made the stock skill verdict DANGEROUS
    under ``scan_skill`` (#91569) — the product rejected its own bundled
    content. Only informational findings (e.g. ``os.environ.get`` key reads
    in the script) may remain.
    """
    assert BUNDLED_SKILL.is_dir(), f"bundled skill moved or missing: {BUNDLED_SKILL}"
    result = scan_skill(BUNDLED_SKILL, source="community")
    allowed, reason = should_allow_install(result)
    assert allowed, f"bundled skill must reinstall cleanly: {reason}"
