"""Tests for tools.skills_duplicate_audit — read-only near-duplicate scanner."""

import hashlib

from tools.skills_duplicate_audit import (
    format_duplicate_report,
    scan_duplicates,
    summarize,
)

# Long enough to clear the minimum-body guard, distinct enough that two skills
# using it are genuinely the same skill twice.
VOICE_BODY = (
    "## When to Use\n"
    "Clone a speaker's voice from a short reference sample for dubbing work.\n"
    "## Reference Audio\n"
    "Supply at least ten seconds of clean speech with no background music.\n"
    "## Procedure\n"
    "Upload the reference clip, then synthesize the target text with the model.\n"
)


def _write_skill(root, name, description, body, frontmatter=True):
    d = root / name
    d.mkdir(parents=True)
    if frontmatter:
        content = f"---\nname: {name}\ndescription: {description}\n---\n{body}"
    else:
        content = body
    (d / "SKILL.md").write_text(content, encoding="utf-8")
    return d


def _pairs(candidates):
    return {(c.name_a, c.name_b) for c in candidates}


def test_exact_duplicate_under_different_names(tmp_path):
    """The issue's real case: same content, name varied just enough to slip past
    the same-name overwrite guard."""
    _write_skill(tmp_path, "ai-voice-cloning", "Clone a voice", VOICE_BODY)
    _write_skill(tmp_path, "cosyvoice2-voice-cloning", "Clone a voice", VOICE_BODY)

    candidates = scan_duplicates([tmp_path])

    assert len(candidates) == 1
    assert candidates[0].confidence == "high"
    assert "identical normalized body hash" in candidates[0].signals


def test_similar_names_different_behavior_is_not_reported(tmp_path):
    """Sibling skills share a name prefix and the mandated section scaffold but do
    opposite things. Reporting these is how the audit loses a user's trust."""
    _write_skill(
        tmp_path, "aws-s3-upload", "Upload files to an S3 bucket",
        "## When to Use\nPushing build artifacts to object storage for release.\n"
        "## Procedure\nRun the sync command against the destination prefix.\n",
    )
    _write_skill(
        tmp_path, "aws-s3-delete", "Delete objects from an S3 bucket",
        "## When to Use\nRemoving objects and every historical version of them.\n"
        "## Procedure\nRestore from a versioned snapshot if this was a mistake.\n",
    )

    assert scan_duplicates([tmp_path]) == []


def test_matching_descriptions_with_different_bodies_is_medium(tmp_path):
    """Same advertised purpose, different actual content — worth a look, not a
    merge."""
    _write_skill(
        tmp_path, "git-commit-helper", "Write a conventional commit message",
        "## Conventional Commits\nPrefix the subject with the change type.\n"
        "## Scope Selection\nName the package the change actually touches.\n",
    )
    _write_skill(
        tmp_path, "commit-message-generator", "Write a conventional commit message",
        "## Draft From Diff\nSummarize the staged hunks into one subject line.\n"
        "## Length Budget\nKeep the subject under seventy-two characters total.\n",
    )

    candidates = scan_duplicates([tmp_path])

    assert len(candidates) == 1
    assert candidates[0].confidence == "medium"
    assert any("similar descriptions" in s for s in candidates[0].signals)


def test_malformed_and_missing_frontmatter_do_not_crash(tmp_path):
    """A skill with no frontmatter falls back to its directory name; broken YAML
    must not take the whole scan down with it."""
    _write_skill(tmp_path, "no-frontmatter", "", VOICE_BODY, frontmatter=False)
    d = tmp_path / "broken-yaml"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: [unclosed\ndescription: \"also broken\n---\n" + VOICE_BODY,
        encoding="utf-8",
    )

    candidates = scan_duplicates([tmp_path])

    names = {c.name_a for c in candidates} | {c.name_b for c in candidates}
    assert "no-frontmatter" in names
    assert all(isinstance(c.confidence, str) for c in candidates)


def test_ordering_is_deterministic(tmp_path):
    """Output gets diffed between runs, so identical input must produce an
    identical report — high confidence first, then alphabetical."""
    _write_skill(tmp_path, "zebra-voice", "Clone a voice", VOICE_BODY)
    _write_skill(tmp_path, "alpha-voice", "Clone a voice", VOICE_BODY)
    _write_skill(
        tmp_path, "mid-notes", "Take structured notes",
        "## Capture\nWrite down decisions and their stated rationale.\n"
        "## Review\nRe-read the captured notes before the next meeting.\n",
    )
    _write_skill(
        tmp_path, "mid-notes-two", "Take structured notes",
        "## Inbox\nCollect loose thoughts before they are lost entirely.\n"
        "## Triage\nPromote anything still relevant a full day later on.\n",
    )

    first = scan_duplicates([tmp_path])
    second = scan_duplicates([tmp_path])

    assert first == second
    assert [c.confidence for c in first] == sorted(
        (c.confidence for c in first), key=lambda c: 0 if c == "high" else 1
    )
    assert ("alpha-voice", "zebra-voice") in _pairs(first)


def test_archived_skills_are_not_scanned(tmp_path):
    """Archived skills sit under .archive/ awaiting restore. Reporting one as a
    duplicate of its own live replacement would be noise."""
    _write_skill(tmp_path, "ai-voice-cloning", "Clone a voice", VOICE_BODY)
    _write_skill(tmp_path / ".archive", "ai-voice-cloning-old", "Clone a voice", VOICE_BODY)

    assert scan_duplicates([tmp_path]) == []


def test_ownership_is_reported_for_both_sides(tmp_path):
    """Whether a candidate is even actionable depends on where each skill came
    from — a hub-installed or protected skill is not the curator's to merge."""
    _write_skill(tmp_path, "ai-voice-cloning", "Clone a voice", VOICE_BODY)
    _write_skill(tmp_path, "cosyvoice2-voice-cloning", "Clone a voice", VOICE_BODY)

    candidate = scan_duplicates([tmp_path])[0]

    assert candidate.ownership_a
    assert candidate.ownership_b


def test_empty_registry(tmp_path):
    assert scan_duplicates([tmp_path]) == []
    assert summarize([]) == {
        "possible_duplicate_pairs": 0,
        "high_confidence_pairs": 0,
        "medium_confidence_pairs": 0,
    }
    assert "None found" in format_duplicate_report([])


def test_scan_never_mutates_the_skill_library(tmp_path):
    """The whole premise of this layer is that it is safe to run. Prove it: every
    file byte-identical, no files added or removed."""
    _write_skill(tmp_path, "ai-voice-cloning", "Clone a voice", VOICE_BODY)
    _write_skill(tmp_path, "cosyvoice2-voice-cloning", "Clone a voice", VOICE_BODY)

    def snapshot():
        return {
            str(p.relative_to(tmp_path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(tmp_path.rglob("*"))
            if p.is_file()
        }

    before = snapshot()
    scan_duplicates([tmp_path])
    assert snapshot() == before


def test_default_scan_classifies_external_dirs_as_external(tmp_path, monkeypatch):
    """`skills.external_dirs` entries are read-only and externally owned, but
    `skill_usage.provenance()` classifies purely by name against the hub and
    bundled manifests — it has no notion of "external" at all. A skill that
    exists only in an external directory must not inherit whatever provenance
    an unrelated same-named manifest entry would produce; the discovered path
    has to settle it. Exercises the *default* scan (no explicit skills_dirs),
    since that's the path skill_usage.get_all_skills_dirs() feeds in
    production."""
    import agent.skill_utils as skill_utils

    local_root = tmp_path / "local"
    external_root = tmp_path / "external"
    local_root.mkdir()
    external_root.mkdir()

    _write_skill(local_root, "ai-voice-cloning", "Clone a voice", VOICE_BODY)
    _write_skill(external_root, "cosyvoice2-voice-cloning", "Clone a voice", VOICE_BODY)

    monkeypatch.setattr(skill_utils, "get_all_skills_dirs", lambda: [local_root, external_root])
    monkeypatch.setattr(skill_utils, "get_external_skills_dirs", lambda: [external_root])

    candidates = scan_duplicates()  # default scan — no explicit skills_dirs

    assert len(candidates) == 1
    ownership = {candidates[0].name_a: candidates[0].ownership_a,
                 candidates[0].name_b: candidates[0].ownership_b}
    assert ownership["cosyvoice2-voice-cloning"] == "external, read-only"
    assert ownership["ai-voice-cloning"] != "external, read-only"


def test_registry_boilerplate_headings_are_discounted(tmp_path):
    """Skill families invent their own templates. Once a heading is everywhere in
    the registry it stops identifying anything, whatever template produced it."""
    def templated(unique_prose):
        # Same section skeleton every time; the prose underneath is what differs.
        return (
            f"## Working With This Skill\n{unique_prose[0]}\n"
            f"## For Beginners\n{unique_prose[1]}\n"
            f"## Reference Files\n{unique_prose[2]}\n"
        )

    family = {
        "axolotl": (
            "YAML-recipe trainer for supervised fine-tuning",
            (
                "Configure the YAML recipe before launching any training run.",
                "Begin from the LoRA example recipe shipped in the examples tree.",
                "Dataset format notes and recipe fields are documented separately.",
            ),
        ),
        "unsloth": (
            "Kernel-accelerated low-rank adapters on a single GPU",
            (
                "Patch the model with the accelerated kernels at import time.",
                "Start from the free notebook covering a four-bit adapter run.",
                "Kernel compatibility per architecture is listed in its own file.",
            ),
        ),
        "pytorch-fsdp": (
            "Shard very large models across many ranks",
            (
                "Shard optimizer state across ranks before the first forward pass.",
                "Read the sharding-strategy primer to pick a wrapping policy.",
                "Checkpoint layout and resharding rules are described elsewhere.",
            ),
        ),
    }
    for name, (description, prose) in family.items():
        _write_skill(tmp_path, name, description, templated(prose))

    # Enough other skills carrying the same template that it reads as common.
    fillers = [
        "Render invoices", "Rotate database credentials", "Summarize podcasts",
        "Diff terraform plans", "Crop product photos", "Chase overdue tickets",
        "Publish release notes", "Tail production logs", "Seed demo fixtures",
    ]
    for i, desc in enumerate(fillers):
        _write_skill(
            tmp_path, f"filler-{i}", desc,
            templated((f"Step {i} alpha detail.", f"Step {i} beta detail.",
                       f"Step {i} gamma detail.")),
        )

    candidates = scan_duplicates([tmp_path])

    # The family shares nothing but the template, so it must not surface.
    assert ("axolotl", "unsloth") not in _pairs(candidates)
    assert ("axolotl", "pytorch-fsdp") not in _pairs(candidates)
    assert ("pytorch-fsdp", "unsloth") not in _pairs(candidates)
