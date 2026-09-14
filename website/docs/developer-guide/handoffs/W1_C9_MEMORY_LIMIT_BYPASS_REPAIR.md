# W1-C9 MemoryStore limit-bypass targeted repair

**Date:** 2026-07-12  
**Engineer:** Kensei  
**Execution approved by:** Sahil  
**Source of truth:** KenseiAgent at `/home/kensei/repos/KenseiAgent/`

---

## 1. Confirmed defect

`MemoryStore.add()` (tools/memory_tool.py:492-528) calls `_auto_compact()` when a new entry would exceed the configured character limit. If compaction returns `success=False`, the control flow falls through into `entries.append(content)` and `save_to_disk()`, then returns `{"success": True}`, silently bypassing the memory limit.

**Root-cause mechanism:**

```
if new_total > limit:                    # outer if
    current = self._char_count(target)
    compact_result = self._auto_compact(target, content)
    if compact_result.get("success"):    # inner if — NO else branch
        # ... success path (3 return points)
    # ← FALLTHROUGH — nothing prevents reaching:
entries.append(content)                   # appends despite overflow!
```

The `entries.append(content)` at line 524 is at the `with`-block level, OUTSIDE the `if new_total > limit:` but within the `with` lock. When `_auto_compact` fails, none of the inner `if` body's three `return` statements execute, so control falls through to the append.

---

## 2. Changed paths

| File | Change |
|------|--------|
| `tools/memory_tool.py` | 14 lines added — `else` clause on `if compact_result.get("success")` that returns a consolidation failure |
| `tests/tools/test_memory_tool.py` | 47 lines added — focused test exercising the auto-compact-failure overflow path |

**No Hermes profile, memory, cron, or config files were touched.**

---

## 3. RED → GREEN test evidence

### RED (before fix)

```
$ python3 -m pytest tests/tools/test_memory_tool.py::TestMemoryStoreAdd::test_auto_compact_failure_rejects_overflow -v

...
FAILED tests/tools/test_memory_tool.py::TestMemoryStoreAdd::test_auto_compact_failure_rejects_overflow
_________________________________ FAILURES __________________________________

    def test_auto_compact_failure_rejects_overflow(self, tmp_path, monkeypatch):
        ...
        r2 = tight.add("memory", "Z" * 120)

>       assert not r2["success"], (
            "Overflow after failed auto-compact must return failure, "
            f"got: {r2}"
        )
E       AssertionError: Overflow after failed auto-compact must return failure,
E       got: {'success': True, 'done': True, 'target': 'memory',
E             'usage': '100% — 218/200 chars', 'entry_count': 2,
E             'message': 'Entry added.',
E             'note': 'Write saved. This update is complete — do not repeat it.'}
```

The bug is confirmed: `usage: 218/200 chars` yet `success: True`.

### GREEN (after fix)

```
$ python3 -m pytest tests/tools/test_memory_tool.py -v

...
test_auto_compact_failure_rejects_overflow PASSED
test_add_exceeding_limit_rejected         PASSED
test_add_overflow_degrades_after_cap      PASSED
test_batch_frees_room_for_otherwise_overflowing_add PASSED
...
======================== 87 passed in 3.37s ==============================
```

All 87 tests pass. All 3 previously-failing memory-tool tests are now green:

1. `test_add_exceeding_limit_rejected` — overflow correctly returns failure
2. `test_add_overflow_degrades_after_cap` — failure budget degrades correctly  
3. `test_batch_frees_room_for_otherwise_overflowing_add` — batch overflow correctly handled

---

## 4. Source diff summary

### tools/memory_tool.py — the fix

The change adds an `else` clause to `if compact_result.get("success")` that returns a consolidation failure dict:

```python
                else:
                    current3 = self._char_count(target)
                    return self._consolidation_failure({
                        "success": False,
                        "error": (
                            f"Memory at {current3:,}/{limit:,} chars. "
                            f"Adding this entry ({len(content)} chars) would exceed the limit. "
                            "Auto-compaction could not free enough space. "
                            "Merge/summarise existing entries manually with "
                            "memory(action=replace), then retry this add."
                        ),
                        "current_entries": self._entries_for(target),
                        "usage": f"{current3:,}/{limit:,}",
                    })
```

**Design rationale:**
- The `_consolidation_failure()` wrapper counts the failure toward the per-turn budget and returns terminal output after the cap is exceeded (preventing infinite retry loops — issue #42405).
- Within the cap, the dict is returned unchanged with `"current_entries"` and `"usage"` so the model can self-correct.
- The error message includes `"retry this add"` so the test `test_add_overflow_degrades_after_cap` passes its assertion — the model sees an in-turn retry instruction.

**Before/after AST verification:**

```
Before: If(compact_result) lines 495-522  — Orelse: 0 items (FALLTHROUGH)
After:  If(compact_result) lines 495-535  — Orelse: 2 items (assign + return)
```

The `entries.append(content)` at the `with`-block level is now correctly unreachable when `new_total > limit`.

### tests/tools/test_memory_tool.py — the focused test

A new test method `test_auto_compact_failure_rejects_overflow` in `TestMemoryStoreAdd`:

- Creates a `MemoryStore` with `memory_char_limit=200`
- Adds one 95-char entry (distinct, >80 chars — cannot be auto-compacted)
- Asserts `len(store.memory_entries) == 1` and `_char_count == 95`
- Attempts to add a 120-char entry (would total 218 > 200)
- Asserts `success is False` and `current_entries` / `usage` in response
- Asserts store state is unchanged (`len == 1`, `_char_count == 95`, new entry absent)

---

## 5. No production state touched

- All testing used temporary `tmp_path` directories via pytest's `tmp_path` fixture  
- `get_memory_dir` was monkeypatched to point to `tmp_path`
- No `.hermes/` directory or real memory store was read, written, or modified
- No cron jobs, config files, or live services were touched

---

## 6. Acceptance criteria verification

| Criterion | Status |
|-----------|--------|
| Failed compaction never appends or saves the new entry | ✓ — test asserts store state unchanged |
| Response is failure and explains that the limit remains exceeded | ✓ — `success: False`, error mentions chars |
| Entry count/content/character usage unchanged after failed attempt | ✓ — `len(store.memory_entries)`, `_char_count` asserted |
| Below-limit and successful-compaction adds still pass | ✓ — all 87 tests pass including existing success scenarios |
| No live `.hermes/memory` file is read or written | ✓ — tmp_path isolation |
| Focused test observed failing before fix and passing after | ✓ — RED/GREEN evidence above |

## 7. Kensei independent verification

KENSEI independently ran `uv run pytest tests/tools/test_memory_tool.py -q`: **87 passed in 3.39s**. `uv run ruff check tools/memory_tool.py tests/tools/test_memory_tool.py` and `git diff --check` passed. Added-line security scanning found no hardcoded credential, shell-injection, dynamic-evaluation, unsafe-deserialisation or SQL-interpolation pattern. The change remains uncommitted: only `tools/memory_tool.py`, `tests/tools/test_memory_tool.py`, and this evidence record are in scope.
