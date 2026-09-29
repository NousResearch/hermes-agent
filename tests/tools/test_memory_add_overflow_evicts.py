"""add()-overflow evicts oldest-first and archives, instead of refusing.

Covers the at-capacity dead end: once MEMORY.md sits at its char limit every later
add is refused and the fact is lost, because the model has to run a consolidation
pass in the same turn and cannot always. On overflow the oldest entries are now
rotated to ARCHIVE.jsonl (reversible) and the new fact lands.

Every fixture asserts its OWN triggering precondition, measured through the store,
before the assertion under test runs — a fixture that never creates the condition
reports a meaningless result, and "bad fixture" and "bad code" look identical.
"""
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from tools import memory_tool
from tools.memory_tool_store import ENTRY_DELIMITER, EVICT_HEADROOM, MemoryStore

BIG = "the fact that must be saved even though memory is already full"
_last = 0
D = " pad" * 8  # 32 chars, no trailing space (add() strips)


@pytest.fixture
def store_dir(monkeypatch):
    """Memory dir the store resolves per call (MEMORY.md, USER.md, ARCHIVE.jsonl).

    mkdtemp rather than tmp_path: pytest 9.1.1's tmp_path teardown raises KeyError in
    this environment, which turns every test into an error regardless of outcome.
    """
    d = Path(tempfile.mkdtemp(prefix="mem-evict-"))
    monkeypatch.setattr(memory_tool, "get_memory_dir", lambda: d)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def size(entries):
    return len(ENTRY_DELIMITER.join(entries))


def make_store(limit, store_dir, target="memory"):
    st = MemoryStore(memory_char_limit=limit, user_char_limit=limit)
    st.load_from_disk()
    return st


def fill_to_cap(st, limit, target="memory"):
    """Fill using the store's own accessor until the NEXT add of BIG would overflow."""
    entries = st.memory_entries if target == "memory" else st.user_entries
    n = 0
    while size(entries + [BIG]) <= limit:
        assert st.add(target, f"filler {n}" + D).get("success"), "filler add failed"
        entries = st.memory_entries if target == "memory" else st.user_entries
        n += 1
        assert n < 200, "fill loop failed to converge"
    return entries


def read_archive(store_dir):
    p = store_dir / "ARCHIVE.jsonl"
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_overflow_succeeds_evicts_oldest_and_archives(store_dir):
    global _last
    limit = 400
    st = make_store(limit, store_dir)
    st.add("memory", "oldest keeper" + D)
    st.add("memory", "newest keeper" + D)
    pre = fill_to_cap(st, limit)

    # PRECONDITION, through the store: the next add genuinely overflows.
    assert size(pre + [BIG]) > limit, f"FIXTURE INVALID: {size(pre + [BIG])} <= {limit}"

    res = st.add("memory", BIG)

    assert res["success"] is True, res
    on_disk = (store_dir / "MEMORY.md").read_text(encoding="utf-8")
    assert size([e for e in on_disk.split(ENTRY_DELIMITER) if e.strip()]) <= limit
    assert BIG in on_disk, "the new fact must be present"
    assert "oldest keeper" not in on_disk, "oldest entries are evicted first"

    records = read_archive(store_dir)
    assert records, "eviction must be archived, never dropped"
    assert records[0]["entry"] == "oldest keeper" + D, "archived oldest-first"
    assert records[0]["action"] == "add_overflow"
    assert res["evicted"] == len(records)
    assert res["archive"] == "ARCHIVE.jsonl"
    # Reversible: every evicted entry is recoverable verbatim from the archive.
    assert [r["entry"] for r in records] == res["evicted_entries"]


def test_headroom_leaves_room_for_the_next_add(store_dir):
    limit = 400
    st = make_store(limit, store_dir)
    fill_to_cap(st, limit)
    first = st.add("memory", BIG)
    assert first["success"] is True

    # The eviction target leaves EVICT_HEADROOM free, so one more add fits untouched.
    after_first = (store_dir / "MEMORY.md").read_text(encoding="utf-8")
    assert len(after_first) <= limit - EVICT_HEADROOM + 1

    second = st.add("memory", "one more fact right after")
    assert second["success"] is True
    assert "evicted" not in second, "no second eviction needed within the headroom"


def test_entry_too_big_for_an_empty_store_still_refuses(store_dir):
    limit = 200
    st = make_store(limit, store_dir)

    res = st.add("memory", "x" * 300)

    assert res["success"] is False, "an entry that cannot fit an empty store must refuse"
    assert "exceed the limit" in res["error"]
    assert not (store_dir / "MEMORY.md").exists() or not (store_dir / "MEMORY.md").read_text(encoding="utf-8").strip()
    assert read_archive(store_dir) == [], "nothing is archived when nothing was evicted"


def test_archive_write_failure_refuses_and_changes_nothing(store_dir, monkeypatch):
    limit = 400
    st = make_store(limit, store_dir)
    st.add("memory", "oldest keeper" + D)
    pre = fill_to_cap(st, limit)
    assert size(pre + [BIG]) > limit, "FIXTURE INVALID: fixture is not at the cap"

    import tools.memory_tool_store as mts
    real, mts._append_archive = mts._append_archive, lambda records: "OSError: read-only file system"
    try:
        res = st.add("memory", BIG)
    finally:
        mts._append_archive = real

    # Dropping entries with no record is the one outcome worse than refusing.
    assert res["success"] is False
    assert "archive write failed" in res["error"]
    after = (store_dir / "MEMORY.md").read_text(encoding="utf-8")
    assert "oldest keeper" in after, "no entry may be lost when the archive fails"
    assert BIG not in after
    assert read_archive(store_dir) == []


def test_user_profile_uses_the_same_path(store_dir):
    limit = 400
    st = make_store(limit, store_dir)
    st.add("user", "preference that must survive" + D)
    pre = fill_to_cap(st, limit, target="user")
    assert size(pre + [BIG]) > limit, "FIXTURE INVALID: USER.md fixture is not at the cap"

    res = st.add("user", BIG)

    assert res["success"] is True, res
    assert BIG in (store_dir / "USER.md").read_text(encoding="utf-8")
    rec = read_archive(store_dir)
    assert rec and rec[0]["target"] == "user"


def test_over_cap_on_load_heals_itself(store_dir):
    """The state an external writer leaves behind (#123569): file over the cap on
    load means every later add is refused forever. It now recovers on the next add."""
    limit = 300
    entries = [f"external {i}" + D for i in range(12)]
    (store_dir / "MEMORY.md").write_text(ENTRY_DELIMITER.join(entries), encoding="utf-8")
    st = make_store(limit, store_dir)
    loaded = size(st.memory_entries)
    assert loaded > limit, f"FIXTURE INVALID: loaded {loaded} <= {limit}"

    res = st.add("memory", "the fact that was previously blocked forever")

    assert res["success"] is True, res
    on_disk = (store_dir / "MEMORY.md").read_text(encoding="utf-8")
    assert size([e for e in on_disk.split(ENTRY_DELIMITER) if e.strip()]) <= limit
    assert "the fact that was previously blocked forever" in on_disk
    assert read_archive(store_dir), "recovery archives what it removes"


def test_prompt_snapshot_stays_frozen(store_dir):
    """Eviction must not disturb the load-time snapshot: it is the prefix-cache surface."""
    limit = 400
    st = make_store(limit, store_dir)
    fill_to_cap(st, limit)
    before = st.format_for_system_prompt("memory")
    n_before = len(st.memory_entries)

    st.add("memory", BIG)

    assert st.format_for_system_prompt("memory") == before, "snapshot must stay byte-stable"
    assert len(st.memory_entries) < n_before, "live list is what shrinks"
    assert st.memory_entries == [
        e for e in (store_dir / "MEMORY.md").read_text(encoding="utf-8").split(ENTRY_DELIMITER) if e.strip()]
