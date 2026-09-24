"""Repo-wide bundled-skills self-scan CI gate (issue #111334, class 2 acceptance).

"Bundled skills/ Hub-scan: 0 self-blocks; CI gate prevents regression."

Every bundled skill under ``skills/<group>/<skill>/`` is scanned with the same
community-source policy the Hub applies at install time (scan_skill uses
rglob, so nested scripts/references are covered recursively from each
skill's top-level entry). Any skill that would self-block (verdict ==
"dangerous") must appear in the KNOWN_BLOCKED allowlist below, and only with
the critical pattern-ids its entry explicitly permits. Each allowlisted entry
is a debt item with a named fix PR that must eventually burn down to zero.

Design notes:
- Pattern-level subset semantics (actual critical ids ⊆ allowed ids per
  entry): an allowlisted skill that gets fixed and turns CLEAN — or sheds one
  of its patterns — must NOT fail this test, keeping the gate merge-order-safe
  against the outstanding fix PRs landing in any order. But an allowlisted
  skill silently GAINING a new critical pattern id DOES fail: keys-only
  allowlists would let a listed skill acquire new criticals unnoticed.
- The vacuum guard is load-bearing: a walk that silently finds zero skills
  (layout change, path typo) would make the subset assertion pass on an empty
  set. It must fail loudly instead. (Competitor PR #98489's test has exactly
  that vacuous-pass bug; we do not repeat it.)
"""

from pathlib import Path

from tools.skills_guard import scan_skill

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_ROOT = REPO_ROOT / "skills"

# Skills that currently scan "dangerous" under the community policy and are
# therefore known self-blocks. Each entry maps skill -> (allowed critical
# pattern-ids, prose citation of the fix PRs). An allowlisted skill may only
# carry the listed pattern ids; a new pattern id on a listed skill fails the
# gate just like a new dangerous skill. When a fix merges, delete the entry.
# This list is a burn-down, not a home.
KNOWN_BLOCKED: dict[str, tuple[frozenset, str]] = {
    # Fix vehicles: #98489, #98490 — both currently hardcode the pre-move
    # skills/research/ path and need a retarget to skills/web/ before merge.
    "web/blocked-page-recovery": (
        frozenset({"env_exfil_curl"}),
        "critical env_exfil_curl; fix vehicles #98489/#98490 (retarget "
        "needed: they still point at the pre-move skills/research/ path; "
        "#111334)",
    ),
    # Fix vehicles: #102476, #105220.
    "software-development/github": (
        frozenset({"curl_pipe_python", "env_exfil_curl"}),
        "critical curl_pipe_python + env_exfil_curl; fix vehicles "
        "#102476/#105220 (#111334)",
    ),
    # scripts/export-frames.js legitimately passes Chrome's
    # --disable-setuid-sandbox to headless Puppeteer; skills_guard's
    # setuid_setgid pattern false-positives on the flag name. Deliberately
    # kept as a whole literal (string-splitting to hide it would be
    # concealment). Fix vehicle: #121700 (related: #85975).
    "creative/p5js": (
        frozenset({"setuid_setgid"}),
        "critical setuid_setgid (Chrome sandbox-flag false positive); fix "
        "vehicle #121700, related #85975 (#111334)",
    ),
}

# Lower bound on how many skills the walk must find (58 on current main). A
# layout change that makes the walk scan almost nothing would otherwise let
# the subset assertion pass vacuously. If the bundled corpus legitimately
# shrinks (e.g. skills moved to optional-skills), update this floor in the
# same PR.
MIN_SCANNED_SKILLS = 50


def _iter_bundled_skills() -> list[Path]:
    """Two-level walk: ``skills/<group>/<skill>/`` directories. Skips
    DESCRIPTION.md and any non-directory entries. A group-level directory
    that itself holds a SKILL.md is scanned as a skill, so the walk cannot
    silently miss a skill that sits one level up from the canonical layout."""
    assert SKILLS_ROOT.is_dir(), f"skills root missing: {SKILLS_ROOT}"
    skills: list[Path] = []
    for group in sorted(p for p in SKILLS_ROOT.iterdir() if p.is_dir()):
        if (group / "SKILL.md").is_file():
            skills.append(group)
            continue
        for skill in sorted(p for p in group.iterdir() if p.is_dir()):
            skills.append(skill)
    return skills


def _scan_corpus() -> tuple[dict[str, list[str]], int]:
    """Scan every bundled skill at community trust (what the Hub applies on
    install). Returns ``(dangerous_map, scanned_count)`` where dangerous_map
    maps ``<group>/<skill>`` -> sorted critical pattern ids."""
    dangerous: dict[str, list[str]] = {}
    skills = _iter_bundled_skills()
    for skill in skills:
        result = scan_skill(skill, source="community")
        if result.verdict == "dangerous":
            key = (
                skill.name
                if skill.parent == SKILLS_ROOT
                else f"{skill.parent.name}/{skill.name}"
            )
            dangerous[key] = sorted(
                {f.pattern_id for f in result.findings if f.severity == "critical"}
            )
    return dangerous, len(skills)


def test_corpus_walk_finds_plausible_number_of_skills():
    """Vacuum guard: the walk must actually see the corpus, or the subset
    assertion below proves nothing (PR #98489's vacuous-pass bug)."""
    _, count = _scan_corpus()
    assert count >= MIN_SCANNED_SKILLS, (
        f"Only scanned {count} bundled skills (< {MIN_SCANNED_SKILLS}). "
        "The skills/ layout likely changed and this gate would pass "
        "vacuously — update the walk, do not lower this floor blindly. If "
        "the bundled corpus legitimately shrank (e.g. skills moved to "
        "optional-skills), update this floor in the same PR."
    )


def _check_allowlist(dangerous: dict, allowlist: dict) -> list[str]:
    """Return human-readable regressions for a dangerous-map vs the allowlist.

    Two rules, both subset-shaped (burn-down semantics):
      (a) every dangerous skill must have an allowlist entry;
      (b) each allowlisted skill's actual critical pattern-id set must be a
          SUBSET of the entry's allowed ids — an allowlisted skill silently
          gaining a NEW critical pattern is a regression even if its name was
          already listed.
    A skill shrinking its pattern set or turning clean never fails."""
    regressions = []
    for name in sorted(set(dangerous) - set(allowlist)):
        regressions.append(
            f"{name} criticals={sorted(dangerous[name])} has no allowlist entry"
        )
    for name in sorted(set(dangerous) & set(allowlist)):
        unexpected = sorted(set(dangerous[name]) - set(allowlist[name]))
        if unexpected:
            regressions.append(
                f"{name} gained unexpected critical pattern(s) {unexpected}; "
                f"allowlist permits only {sorted(allowlist[name])}"
            )
    return regressions


def test_bundled_skills_corpus_gate():
    """0 unexpected self-blocks: dangerous skills ⊆ KNOWN_BLOCKED and their
    critical pattern ids ⊆ each entry's allowed ids.

    Subset semantics are deliberate: an allowlisted skill that gets fixed and
    turns CLEAN (or sheds one of its patterns) does NOT fail the test
    (merge-order-safe against the cited fix PRs); only a NEW dangerous skill
    or a NEW critical pattern on a listed skill does."""
    dangerous, count = _scan_corpus()
    allowed = {name: ids for name, (ids, _cite) in KNOWN_BLOCKED.items()}
    regressions = _check_allowlist(dangerous, allowed)
    assert not regressions, (
        f"{len(regressions)} bundled-skill regression(s) against the "
        f"corpus allowlist ({count} skills scanned): "
        + "; ".join(regressions)
        + ". Reword the trigger content, or land a fix PR, before adding an "
        "allowlist entry (entries must cite their fix PRs — #111334)."
    )


def test_allowlist_subset_rule_rejects_new_pattern_id():
    """RED-derived: an allowlisted skill gaining a NEW critical pattern id
    must fail the subset check even though its name is already listed
    (keys-only allowlists let a listed skill acquire new criticals
    silently)."""
    allowlist = {"web/demo": frozenset({"env_exfil_curl"})}
    # Unexpected pattern id on a listed skill -> regression naming the id.
    regs = _check_allowlist({"web/demo": ["env_exfil_curl", "rm_rf_root"]}, allowlist)
    assert len(regs) == 1 and "rm_rf_root" in regs[0] and "web/demo" in regs[0]
    # Unlisted dangerous skill -> regression (rule (a)).
    assert len(_check_allowlist({"other/skill": ["rm_rf_root"]}, allowlist)) == 1
    # Exact match, subset (pattern cleared), and turning clean -> pass.
    assert not _check_allowlist({"web/demo": ["env_exfil_curl"]}, allowlist)
    assert not _check_allowlist({"web/demo": []}, allowlist)
    assert not _check_allowlist({}, allowlist)
