"""LF-canonical state files must land LF bytes on every platform.

Besides ``skill_manage`` (fixed in t_913dc878), several in-process writers rewrote or
appended to machine state files through the text-mode default (``newline=None``), which
translates every ``"\\n"`` to ``os.linesep`` — CRLF on Windows. Observed damage on the
real home: ``memories/MEMORY.md`` + ``USER.md`` all-CRLF, ``skills/.usage.json`` all-CRLF,
``skills/.bundled_manifest`` + ``.hub/lock.json`` all-CRLF, and
``skills/.curator_ledger.jsonl`` MIXED (LF body, CRLF lines appended afterwards).

Each site now opts out with ``newline=""``. These tests cover the two shapes that broke —
**append** (an existing LF body must stay LF) and **full rewrite** — and assert on raw
bytes, so they fail on Windows for either failure mode (translation, or a mixed-ending
append) and pin the same bytes on Linux.
"""

import json
from pathlib import Path


def _no_cr(path: Path) -> bytes:
    raw = path.read_bytes()
    assert b"\r\n" not in raw, f"{path.name} was written with CRLF: {raw[:80]!r}"
    return raw


# --------------------------------------------------------------------------- #
# append shape — the mixed-ending case
# --------------------------------------------------------------------------- #

def test_ledger_append_keeps_the_lf_body_and_writes_an_lf_line(tmp_path, monkeypatch):
    """``skill_ledger.append_entry`` opens the JSONL with ``newline=""``: the historical
    body keeps its LF endings and the new line matches — no mixed file."""
    from tools import skill_ledger

    home = tmp_path / "home"
    skills = home / "skills"
    skills.mkdir(parents=True)
    monkeypatch.setattr(skill_ledger, "get_hermes_home", lambda: home)
    monkeypatch.setattr(skill_ledger, "ledger_enabled", lambda: True)

    ledger = skills / ".curator_ledger.jsonl"
    body = b'{"id":"first"}\n{"id":"second"}\n'
    ledger.write_bytes(body)

    assert skill_ledger.append_entry("patch", "my-skill") is not None

    raw = _no_cr(ledger)
    assert raw.startswith(body), "the pre-existing LF body must be appended to, not rewritten"
    assert raw.count(b"\n") == 3
    assert json.loads(raw.splitlines()[2])["skill"] == "my-skill"


# --------------------------------------------------------------------------- #
# full-rewrite shape
# --------------------------------------------------------------------------- #

def test_memory_store_rewrite_is_lf(tmp_path):
    """The memory store joins entries with ``\\n§\\n`` and writes them untranslated."""
    from tools.memory_tool_store import ENTRY_DELIMITER, MemoryStore

    path = tmp_path / "MEMORY.md"
    MemoryStore._write_file(path, ["alpha", "beta"])
    assert path.read_bytes() == ENTRY_DELIMITER.join(["alpha", "beta"]).encode("utf-8")


def test_memory_store_rewrite_normalizes_a_crlf_file_to_lf(tmp_path):
    """A store that was written by an older build is rewritten LF (the fix converges)."""
    from tools.memory_tool_store import MemoryStore

    path = tmp_path / "USER.md"
    path.write_bytes("alpha\r\n\u00a7\r\nbeta\r\n".encode("utf-8"))
    MemoryStore._write_file(path, ["alpha", "beta"])
    assert path.read_bytes() == "alpha\n\u00a7\nbeta".encode("utf-8")


def test_memory_drift_backup_is_a_byte_copy(tmp_path, monkeypatch):
    """The drift snapshot must reproduce the file on disk, endings included.

    ``raw`` deliberately does not round-trip (blank line after the delimiter — the
    hand-edited shape the drift guard exists for) and carries CRLF bytes, so a
    translating backup write would be caught.
    """
    from tools.memory_tool_store import MemoryStore

    path = tmp_path / "MEMORY.md"
    raw = "entry one\n\u00a7\n\nentry two\r\nstray shell-appended line\r\n"
    path.write_bytes(raw.encode("utf-8"))

    store = MemoryStore()
    monkeypatch.setattr(MemoryStore, "_path_for", staticmethod(lambda target: path))
    backup = store._detect_external_drift("memory", raw)
    assert backup is not None
    assert Path(backup).read_bytes() == raw.encode("utf-8")


def test_usage_ledger_rewrite_is_lf(tmp_path, monkeypatch):
    from tools import skill_usage

    target = tmp_path / ".usage.json"
    monkeypatch.setattr(skill_usage, "_usage_file", lambda: target)
    assert skill_usage.save_usage({"my-skill": {"uses": 3}}) is True
    _no_cr(target)


def test_curator_suppression_list_rewrite_is_lf(tmp_path, monkeypatch):
    from tools import skill_usage

    skills = tmp_path / "skills"
    skills.mkdir()
    monkeypatch.setattr(skill_usage, "_skills_dir", lambda: skills)
    skill_usage._toggle_suppressed_name("old-builtin", add=True)
    assert (skills / ".curator_suppressed").read_bytes() == b"old-builtin\n"


def test_bundled_manifest_rewrite_is_lf():
    from hermes_constants import get_hermes_home
    from tools import skills_sync

    skills_sync._write_manifest({"alpha": "h1", "beta": "h2"})
    target = get_hermes_home() / "skills" / ".bundled_manifest"
    assert target.read_bytes() == b"alpha:h1\nbeta:h2\n"


def test_sync_state_rewrite_is_lf():
    from hermes_constants import get_hermes_home
    from tools import skills_sync_client

    skills_sync_client.write_sync_state({"profile": {"root": "abc"}})
    _no_cr(get_hermes_home() / "skills" / ".sync_state")


def test_hub_lock_backfill_writes_lf(tmp_path, monkeypatch):
    """The optional-provenance backfill is the only writer of ``.hub/lock.json``."""
    from tools import skills_sync, skills_sync_optional

    skills = tmp_path / "skills"
    optional = tmp_path / "optional-skills"
    src = optional / "creative" / "meme-generation"
    dest = skills / "creative" / "meme-generation"
    src.mkdir(parents=True)
    dest.mkdir(parents=True)
    (src / "SKILL.md").write_text("---\nname: meme-generation\n---\n", encoding="utf-8")
    (dest / "SKILL.md").write_text("---\nname: meme-generation\n---\n", encoding="utf-8")

    monkeypatch.setattr(skills_sync, "_get_optional_dir", lambda: optional)
    monkeypatch.setattr(skills_sync, "_skills_dir", lambda: skills)
    monkeypatch.setattr(skills_sync, "_dir_hash", lambda directory: "identical-hash")
    monkeypatch.setattr(skills_sync_optional, "_iter_optional_skills",
                        lambda *a, **k: iter([(src / "SKILL.md", src, "creative/meme-generation")]))
    monkeypatch.setattr(skills_sync_optional, "_load_hub_lock", lambda: None)

    assert skills_sync_optional._backfill_optional_provenance(quiet=True) == ["meme-generation"]
    raw = _no_cr(skills / ".hub" / "lock.json")
    assert json.loads(raw.decode("utf-8"))["installed"]["meme-generation"]["source"] == "official"


def test_checkpoint_ledger_rewrite_is_lf(tmp_path):
    from tools.checkpoint_manager import _save_ledger

    _save_ledger(tmp_path, "deadbeef", {"abcdef": {"ts": 1, "paths": ["a.txt"]}})
    _no_cr(tmp_path / "ledgers" / "deadbeef.json")


def test_agent_import_memory_merge_is_lf(tmp_path):
    """``hermes import-agent`` merges CLAUDE.md entries into memories/MEMORY.md — the same
    file the memory store owns, so it must write the same bytes."""
    from hermes_cli.agent_import import AgentImporter

    src = tmp_path / "CLAUDE.md"
    src.write_text("# Rules\n\n- Always run the focused tests before committing.\n"
                   "- Never commit directly to main.\n", encoding="utf-8")
    target = tmp_path / "home"

    importer = AgentImporter("claude-code", tmp_path, target, execute=True)
    importer.import_context_file(src, "context-file")

    destination = target / "memories" / "MEMORY.md"
    assert destination.exists(), importer.build_report()
    _no_cr(destination)
