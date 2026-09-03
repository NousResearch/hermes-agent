"""The bundled GitHub skill must stay clean under Hermes' own context scanner.

``software-development/github`` (#98539) shipped six reference files whose
curl examples interpolated ``$GITHUB_TOKEN`` on the same line as ``curl`` —
the ``exfil_curl`` shape in tools/threat_patterns.py. The prompt builder
drops every matching reference from context with only a ``logger.warning``,
so a fresh install advertises references the agent can never read (#102473).
These anchors fail closed — re-introducing a credential-interpolating
one-liner into any shipped markdown file of the skill turns them red.
"""

from pathlib import Path

from tools.threat_patterns import scan_for_threats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUNDLED_SKILL = PROJECT_ROOT / "skills" / "software-development" / "github"


def test_bundled_github_skill_references_pass_context_scan():
    """Every shipped markdown file must scan clean at context scope.

    Six references (auth, ci-troubleshooting, code-review,
    github-api-cheatsheet, pr-workflow, repo-management) used to trip
    ``exfil_curl`` on every fresh v0.21.0 install (#102473). Moving the
    Authorization header into its own plainly-named variable line — the
    convention #98489 established for blocked-page-recovery — clears the
    pattern while keeping the commands functional.
    """
    files = sorted(BUNDLED_SKILL.rglob("*.md"))
    assert files, "bundled github skill must still ship markdown files"
    dirty = {
        str(f.relative_to(BUNDLED_SKILL)): scan_for_threats(
            f.read_text(encoding="utf-8"), scope="context"
        )
        for f in files
    }
    dirty = {name: hits for name, hits in dirty.items() if hits}
    assert dirty == {}, f"bundled skill files tripped the context scanner: {dirty}"
