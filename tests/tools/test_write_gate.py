"""Tests for tools/write_gate.py — the 7-stage acceptance pipeline (issue #135039).

Each fixture imports ``acceptance_pipeline`` directly from ``tools.write_gate``,
isolating the gate from the store layer.  The store is used only as a
size/counter/entry source; writes are NOT applied through the store here — the
gate decides admit/refuse.

The ``TestDocTargetPathUnderGate`` class is the integration proof (INTEG-4): it
drives chain-doc writes through the FULL tool path (``tools.memory_tool``) in
the doc-chain world (the sandbox head with doc-aware store primitives), proving
the gate scopes size/limit/stale-entry PER TARGET (INTEG-1/INTEG-2) instead of
charging every target against the anchor's budget.  Prior batteries ran the
gate against ``target='memory'`` only — the absence that let the scoping bug
ship green.
"""
import json

import pytest

from tools.write_gate import (
    GateRefusal,
    acceptance_pipeline,
    verify_post_write,
    CHAR_CAP_SOFT_THRESHOLD,
    _stage1_char_cap,
    _stage1_projected_hard_cap,
    _stage2_date_currency,
    _stage3_provenance,
    _stage4_procedure_shape,
    _stage5_grammar_threat,
    _stage6_stale_entry,
)
from tools.memory_tool_store import MemoryStore
from tools.memory_tool import memory_tool
from datetime import UTC


# --- Test helpers ------------------------------------------------------------

def _make_store(limit: int = 2200, entries: list[str] | None = None):
    """Build a MemoryStore with preset in-memory anchor entries (no disk I/O)."""
    store = MemoryStore(memory_char_limit=limit, user_char_limit=1375)
    store.memory_entries = list(entries or [])
    store.user_entries = []
    return store


def _run(store, action="add", target="memory", content="", old_text=None,
         operations=None, caller="user"):
    """Convenience wrapper for acceptance_pipeline.

    Resolves the entries snapshot through the store's doc-aware arm
    (``_entries_for``), mirroring the real Phase-7 hook — so a ``doc:`` target
    is anchored against the DOC's entries, never the anchor's (INTEG-2).
    """
    return acceptance_pipeline(
        store, action=action, target=target, content=content,
        old_text=old_text, operations=operations, caller=caller,
        entries=list(store._entries_for(target)),
    )


def _make_audit_path(tmp_path, monkeypatch):
    """Redirect the JSONL audit log to a temp file."""
    monkeypatch.setattr("tools.write_gate._audit_log_path",
                        lambda: tmp_path / "write_gate_audit.jsonl")


def _read_log(tmp_path):
    """Parse the JSONL audit log into a list of dicts ([] when absent)."""
    log_path = tmp_path / "write_gate_audit.jsonl"
    if not log_path.exists():
        return []
    text = log_path.read_text(encoding="utf-8").strip()
    return [json.loads(l) for l in text.split("\n") if l]


def _make_doc_store(limit: int = 10000, doc_name: str = "001-conventions",
                    doc_entries: list[str] | None = None, anchor_fill: int = 0):
    """Build a store in the doc-chain world: an anchor referencing ``doc_name``
    plus that doc's own entries in ``_doc_entries`` (no disk I/O)."""
    store = MemoryStore(memory_char_limit=limit, user_char_limit=1375)
    store.memory_entries = [f"Dated memory chain: @doc:{doc_name}"]
    # Optionally push the anchor near its cap with varied filler (dedup-safe).
    for i in range(anchor_fill):
        store.memory_entries.append("filler-" + str(i) + "-" + "f" * 900)
    store._doc_entries[doc_name] = list(doc_entries or [])
    store.user_entries = []
    return store


# =========================================================================
# 1. test_char_cap_soft_threshold_85pct
#    Refuse-on-over-cap: char-cap 4000, soft 0.85 threshold.
# =========================================================================

class TestCharCapSoftThreshold85pct:
    def test_refuses_when_projected_exceeds_85pct(self):
        """A write that would push the store past 85% of its limit is refused."""
        store = _make_store(limit=4000, entries=["x" * 3000])  # 3000 / 4000 = 75%
        # Adding 800 chars → 3800, which is > 4000 * 0.85 = 3400
        refusal = _stage1_char_cap(store, "x" * 800, "memory")
        assert refusal is not None
        assert refusal.failure_class == "over_budget"
        assert refusal.action == "REFUSE"

    def test_admits_when_under_85pct(self):
        """A write that stays within 85% is admitted."""
        store = _make_store(limit=4000, entries=["x" * 2000])  # 2000 / 4000 = 50%
        refusal = _stage1_char_cap(store, "x" * 200, "memory")
        assert refusal is None

    def test_full_pipeline_refuses_over_cap(self):
        """Through acceptance_pipeline, over-cap content is refused."""
        store = _make_store(limit=4000, entries=["x" * 3000])
        refusal = _run(store, content="x" * 800)
        assert refusal is not None
        assert refusal.failure_class == "over_budget"

    def test_soft_refusal_marks_hard_cap_false(self):
        """A write refused by the 85% soft arm carries detail['hard_cap'] is False
        and rationale names the soft threshold (write_gate.py:247-258)."""
        store = _make_store(limit=4000, entries=["x" * 3000])  # 75%, +800 → 3800 > 3400 (85%)
        refusal = _run(store, content="x" * 800)
        assert refusal is not None
        assert refusal.failure_class == "over_budget"
        assert refusal.detail["hard_cap"] is False
        assert "(soft 85%" in refusal.rationale

    def test_hard_cap_refusal_marks_hard_cap_true(self):
        """A growing replace past the hard limit is refused by _stage1_projected_hard_cap
        with rationale naming the hard limit (write_gate.py:267-296).

        Note: the projected hard-cap arm emits "(hard limit)" in the rationale
        but does NOT set detail['hard_cap'] (only the soft arm does) — the
        rationale is the discriminator, asserted here.
        """
        store = _make_store(limit=500, entries=["short"])
        refusal = _run(store, action="replace", content="x" * 600, old_text="short")
        assert refusal is not None
        assert refusal.failure_class == "over_budget"
        assert "(hard limit)" in refusal.rationale


# =========================================================================
# 2. test_provenance_before_grammar
#    Refuse-on-imperative-agent-caller: caller==user passes,
#    caller==agent imperative -> REFUSE to skills/.
# =========================================================================

class TestProvenanceBeforeGrammar:
    def test_user_caller_passes_imperative(self):
        """caller == 'user' with imperative content is admitted (standing preference)."""
        refusal = _stage3_provenance("user", "run the build script")
        assert refusal is None

    def test_agent_caller_imperative_refused(self):
        """caller != 'user' + imperative -> REFUSE, reroute to skills/."""
        refusal = _stage3_provenance("background_review", "run the build script")
        assert refusal is not None
        assert refusal.action == "REFUSE"
        assert refusal.reroute_target == "skills/"
        assert refusal.failure_class == "scan_blocked"

    def test_agent_caller_non_imperative_passes(self):
        """caller != 'user' but non-imperative content is admitted."""
        refusal = _stage3_provenance("background_review", "the build script succeeded")
        assert refusal is None

    def test_provenance_runs_before_grammar(self):
        """Provenance refusal takes priority: imperative agent content is refused
        by the provenance lens, not the grammar/threat check."""
        store = _make_store()
        refusal = _run(store, content="Run the nightly batch job", caller="background_review")
        assert refusal is not None
        assert "provenance" in refusal.rationale

    def test_unknown_caller_treated_as_non_user(self):
        """An 'unknown' caller (provenance resolution failed) is non-user (T2-1):
        agent-inferred imperative content is never promoted to user-trusted."""
        refusal = _stage3_provenance("unknown", "run the deploy script")
        assert refusal is not None
        assert refusal.action == "REFUSE"
        assert refusal.reroute_target == "skills/"


# =========================================================================
# 3. test_whole_entry_exact_match_reanchor
#    Whole-entry exact match via _find_unique_match; fail-closed on drift.
# =========================================================================

class TestWholeEntryExactMatchReanchor:
    def test_exact_match_passes(self):
        """When old_text exactly equals an entry, Stage 6 finds no drift."""
        entry = "Deploy target is Fly.io with manual approval."
        refusal = _stage6_stale_entry([entry], entry)
        assert refusal is None

    def test_drifted_entry_refused(self):
        """If old_text no longer exactly matches any entry, refuse with anchor_drift."""
        refusal = _stage6_stale_entry(["current entry text"], "old pinned entry text")
        assert refusal is not None
        assert refusal.failure_class == "stale_entry"
        assert "anchor_drift" in refusal.rationale or "stale" in refusal.rationale.lower()

    def test_ambiguous_match_refused(self):
        """When old_text matches multiple distinct entries, refuse (fail-closed)."""
        refusal = _stage6_stale_entry(["entry A contains text", "entry B also text here"], "text")
        assert refusal is not None
        assert refusal.failure_class == "stale_entry"


# =========================================================================
# 4. test_stale_entry_refusal
#    Same as #3 but focused on the full pipeline: a stale old_text is refused.
# =========================================================================

class TestStaleEntryRefusal:
    def test_stale_old_text_refused_through_pipeline(self):
        """A replace against an old_text not in the store is refused."""
        store = _make_store(entries=["live entry"])
        refusal = _run(store, action="replace", content="new", old_text="stale pinned text")
        assert refusal is not None
        assert refusal.failure_class == "stale_entry"

    def test_remove_with_stale_old_text_refused(self):
        """A remove against non-matching old_text is refused."""
        store = _make_store(entries=["something else"])
        refusal = _run(store, action="remove", old_text="does not exist")
        assert refusal is not None
        assert refusal.failure_class == "stale_entry"


# =========================================================================
# 5. test_reroute_target_field
#    Refuse-names-reroute-target: Door-B receipt carries reroute_target.
# =========================================================================

class TestRerouteTargetField:
    def test_provenance_refusal_has_reroute_target(self):
        store = _make_store()
        refusal = _run(store, content="Run the deploy script", caller="background_review")
        assert refusal is not None
        assert refusal.reroute_target == "skills/"

    def test_procedure_shape_refusal_has_reroute_target(self):
        store = _make_store()
        refusal = _run(store, content="1. Set up the environment\n2. Run tests\n3. Deploy")
        assert refusal is not None
        assert refusal.reroute_target == "skills/"

    def test_over_budget_refusal_names_chain_doc_destination(self):
        """T1-3: the over_budget soft refusal names a concrete chain-doc destination
        (not None), so the model knows where the content should live instead."""
        store = _make_store(limit=10000, entries=["Dated memory chain: @doc:001-conventions"]
                            + ["filler-" + str(i) + "-" + "f" * 900 for i in range(9)])
        refusal = _run(store, content="x" * 5000)  # pushes anchor past 85%
        assert refusal is not None
        assert refusal.failure_class == "over_budget"
        assert refusal.reroute_target is not None
        assert "doc:001-conventions" in refusal.reroute_target

    def test_over_budget_refusal_without_referenced_doc_falls_back(self):
        """When the anchor references no chain doc, the reroute falls back to the
        generic 'chain-doc' label (still non-None)."""
        store = _make_store(limit=10000, entries=["filler-" + str(i) + "-" + "f" * 900 for i in range(11)])
        refusal = _run(store, content="x" * 5000)
        assert refusal is not None
        assert refusal.failure_class == "over_budget"
        assert refusal.reroute_target == "chain-doc"


# =========================================================================
# 6. test_jsonl_audit_log_fields
#    Change-log carries action + rationale + reroute_target (Door-B shape).
# =========================================================================

class TestJsonlAuditLogFields:
    def test_refusal_logged_with_all_fields(self, tmp_path, monkeypatch):
        _make_audit_path(tmp_path, monkeypatch)
        store = _make_store(limit=100, entries=["x" * 90])
        _run(store, content="x" * 50)
        lines = _read_log(tmp_path)
        assert lines, "a refusal must append exactly one log line"
        for entry in lines:
            assert "action" in entry
            assert "rationale" in entry  # T2-6: reason -> rationale
            assert "reroute_target" in entry
            assert "timestamp" in entry

    def test_admit_logged_with_rationale(self, tmp_path, monkeypatch):
        """T1-1: a plain admitted write appends exactly one ADMIT line."""
        _make_audit_path(tmp_path, monkeypatch)
        store = _make_store()
        _run(store, content="A simple factual note about the project.")
        lines = _read_log(tmp_path)
        admits = [l for l in lines if l.get("action") == "ADMIT"]
        assert len(admits) == 1, "exactly one ADMIT line per admitted write"
        assert admits[0]["rationale"] == "accepted"

    def test_recent_date_flagged_on_admit_line(self, tmp_path, monkeypatch):
        """T2-2: a RECENT date is folded into the single ADMIT line as an advisory."""
        from datetime import datetime, timezone
        _make_audit_path(tmp_path, monkeypatch)
        store = _make_store()
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        _run(store, content=f"Note from {today}: review this.")
        lines = _read_log(tmp_path)
        admits = [l for l in lines if l.get("action") == "ADMIT"]
        assert len(admits) == 1
        assert admits[0].get("advisory") == "date_currency"


# =========================================================================
# 7. test_post_write_readback_eradication  (Stage 7)
#    Old text gone from the target AFTER write — asserted by code, not just
#    by a manual battery run.
# =========================================================================

class TestPostWriteReadbackEradication:
    def test_old_text_eradicated_after_replace(self, tmp_path, monkeypatch):
        """After a successful store.replace, verify_post_write confirms eradication."""
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        store = MemoryStore(memory_char_limit=5000, user_char_limit=3000)
        store.load_from_disk()
        old_entry = "The deploy target is Fly.io with manual approval."
        store.add("memory", old_entry)

        # The gate must pass for a valid replace (old_text matches the entry).
        refusal = _run(store, action="replace", target="memory",
                       content="The deploy target is Fly.io with automated canary.",
                       old_text=old_entry)
        assert refusal is None  # gate admits

        # Apply the actual write.
        result = store.replace("memory", old_entry,
                               "The deploy target is Fly.io with automated canary.")
        assert result["success"] is True

        # Stage 7 read-back: old text must be gone; verify_post_write returns None.
        assert verify_post_write(store, "memory", "replace", old_entry) is None
        path = store._path_for("memory")
        assert old_entry not in path.read_text(encoding="utf-8")

    def test_old_text_eradicated_after_remove(self, tmp_path, monkeypatch):
        """After a successful store.remove, the old text is gone from disk."""
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        store = MemoryStore(memory_char_limit=5000, user_char_limit=3000)
        store.load_from_disk()
        entry = "Temporary staging note that should be removed."
        store.add("memory", entry)

        refusal = _run(store, action="remove", target="memory", old_text=entry)
        assert refusal is None  # gate admits

        store.remove("memory", entry)
        path = store._path_for("memory")
        disk_text = path.read_text(encoding="utf-8") if path.exists() else ""
        assert entry not in disk_text

    def test_readback_falsifier(self, tmp_path, monkeypatch):
        """Stage-7 falsifier: if the post-write read is broken (old_text still
        present), verify_post_write REFUSES with failure_class readback_failed —
        proving the read-back is code-enforced, not a prose claim (T1-2)."""
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        _make_audit_path(tmp_path, monkeypatch)
        store = MemoryStore(memory_char_limit=5000, user_char_limit=3000)
        store.load_from_disk()
        entry = "An entry that was NOT actually removed."
        store.add("memory", entry)

        # Simulate a write that failed to eradicate old_text (store unchanged).
        result = verify_post_write(store, "memory", "remove", entry)
        assert result is not None, "read-back must refuse when old_text persists"
        parsed = json.loads(result)
        assert parsed["failure_class"] == "readback_failed"

    def test_readback_doc_variant(self, tmp_path, monkeypatch):
        """Stage-7 doc variant: the span is confirmed gone from the DOC file
        (store._path_for('doc:<name>')), not only the anchor (INTEG-5/T1-2)."""
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        store = MemoryStore(memory_char_limit=5000, user_char_limit=3000)
        store.load_from_disk()
        doc = "conventions-2026-09"
        store.add("memory", f"Dated memory chain: @doc:{doc}")
        store.add(f"doc:{doc}", "Alpha beta gamma")
        target = f"doc:{doc}"

        # Patch the span inside the doc.
        r = store.patch(target, "Alpha beta gamma", "Alpha beta DELTA")
        assert r["success"] is True
        # Read-back scoped to the DOC target confirms the old span is gone there.
        assert verify_post_write(store, target, "patch", "Alpha beta gamma") is None
        doc_path = store._path_for(target)
        assert "Alpha beta gamma" not in doc_path.read_text(encoding="utf-8")


# =========================================================================
# 8. test_store_size_never_exceeds_cap_after_any_write_sequence
#    Author invariant: measured size never exceeds the cap after any sequence.
# =========================================================================

class TestStoreSizeNeverExceedsCapAfterAnyWriteSequence:
    def test_sequence_of_writes_stays_under_cap(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        store = MemoryStore(memory_char_limit=500, user_char_limit=300)
        store.load_from_disk()

        sequence = [
            "first fact about the project",
            "second fact about deployment",
            "third fact about testing",
            "fourth fact about CI",
        ]
        cap = 500
        for item in sequence:
            # The gate must refuse if adding would exceed the cap.
            refusal = _run(store, content=item)
            if refusal is not None:
                assert store._char_count("memory") <= cap
                continue
            store.add("memory", item)
            assert store._char_count("memory") <= cap, (
                f"after adding {item!r}: {store._char_count('memory')} > {cap}"
            )

    @pytest.mark.parametrize("limit, entries, new_content", [
        (500, ["a" * 400], "b" * 200),   # would exceed 425 (85%) → refused
        (500, ["a" * 300], "b" * 100),   # 400 <= 425 → admitted, stays under 500
        (100, ["x" * 90], "short add"),   # would exceed hard cap → refused
    ])
    def test_parametrized_cap_invariant(self, tmp_path, monkeypatch, limit, entries, new_content):
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        store = MemoryStore(memory_char_limit=limit, user_char_limit=limit)
        store.load_from_disk()
        for e in entries:
            store.add("memory", e)

        refusal = _run(store, content=new_content)
        if refusal is not None and refusal.failure_class in ("over_budget",):
            assert store._char_count("memory") <= limit
        else:
            store.add("memory", new_content)
            assert store._char_count("memory") <= limit


# =========================================================================
# T1-7: growing-replace cap bypass — a short->long rewrite must not slip
# past the hard cap (add_contents is add-only in the main loop).
# =========================================================================

class TestGrowingReplaceCapBypass:
    def test_growing_replace_past_hard_cap_refused(self):
        """A replace whose new content grows the store past the hard limit is refused."""
        store = _make_store(limit=500, entries=["short"])  # 5 chars
        refusal = _run(store, action="replace", content="x" * 600, old_text="short")
        assert refusal is not None
        assert refusal.failure_class == "over_budget"
        assert "grows the store" in refusal.rationale

    def test_growing_patch_past_hard_cap_refused(self):
        """A patch that grows the store past the hard limit is refused."""
        store = _make_store(limit=100, entries=["abcdef"])  # 6 chars
        refusal = _run(store, action="patch", content="abcdef" + "Z" * 200, old_text="abcdef")
        assert refusal is not None
        assert refusal.failure_class == "over_budget"

    def test_shrinking_replace_admitted(self):
        """A replace that shrinks the store is admitted (net_delta <= 0)."""
        store = _make_store(limit=500, entries=["x" * 400])
        refusal = _run(store, action="replace", content="tiny", old_text="x" * 400)
        assert refusal is None

    def test_projected_hard_cap_arm_directly(self):
        """Unit: the projected hard-cap arm computes net_delta and refuses on overflow."""
        store = _make_store(limit=100, entries=["ab"])  # 2 chars
        # replace "ab" (2) with 200 chars: net +198 → projected 200 > 100
        refusal = _stage1_projected_hard_cap(store, "memory", "replace", "y" * 200, "ab")
        assert refusal is not None
        assert refusal.failure_class == "over_budget"
        # A shrinking edit returns None.
        assert _stage1_projected_hard_cap(store, "memory", "replace", "y", "abcdefgh") is None

    def test_growing_replace_on_doc_target_admitted(self):
        """A growing replace on an uncapped doc target is admitted (no cap)."""
        store = _make_doc_store(limit=100, doc_entries=["short"])
        refusal = _run(store, action="replace", target="doc:001-conventions",
                       content="x" * 5000, old_text="short")
        assert refusal is None


# =========================================================================
# T2-2: date-currency facets.
# =========================================================================

class TestDateCurrencyFacets:
    def test_future_date_is_not_fresh(self):
        """T2-2b: a future date (2099) does NOT fire the freshness advisory."""
        assert _stage2_date_currency("Deployed 2099-01-15") is None

    def test_recent_date_is_fresh(self):
        """A date within the 48h window fires the advisory annotation."""
        from datetime import datetime, timezone
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        flag = _stage2_date_currency(f"Deployed {today}")
        assert flag is not None
        assert flag["advisory"] == "date_currency"

    def test_old_date_is_not_fresh(self):
        """A date far outside the window does not fire."""
        assert _stage2_date_currency("Deployed 1999-01-15") is None

    def test_doc_target_skips_date_advisory(self, tmp_path, monkeypatch):
        """T2-2c: the date advisory does not fire for a doc: target (a new dated
        note belongs in the chain, not the session-log)."""
        from datetime import datetime, timezone
        _make_audit_path(tmp_path, monkeypatch)
        store = _make_doc_store(doc_entries=[])
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        _run(store, target="doc:001-conventions", content=f"Dated note {today}")
        lines = _read_log(tmp_path)
        admits = [l for l in lines if l.get("action") == "ADMIT"]
        assert len(admits) == 1
        assert "advisory" not in admits[0], "doc target must skip the date advisory"


# =========================================================================
# T1-8: batch-op stale-entry boundary — the gate delegates batch matching to
# the store; the untested contract is that the delegation holds end-to-end.
# =========================================================================

class TestBatchStaleEntryBoundary:
    def test_batch_with_drifted_old_text_delegates_to_store(self, tmp_path, monkeypatch):
        """The gate skips Stage 6 for batches (returns None / admits), delegating
        per-op matching to the store's _pin_matched_entries; driven through the
        full tool path, the STORE refuses the drifted op."""
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        store = MemoryStore(memory_char_limit=5000, user_char_limit=3000)
        store.load_from_disk()
        store.add("memory", "live entry one")

        # Gate level: a batch with a drifted old_text is admitted (delegated).
        ops = [{"action": "patch", "old_text": "drifted span", "content": "new"}]
        refusal = _run(store, operations=ops, target="memory")
        assert refusal is None, "gate delegates batch stale-matching to the store"

        # Full tool path: the store refuses the drifted op end-to-end.
        result = json.loads(memory_tool(store=store, target="memory", operations=ops))
        assert result["success"] is False

    def test_batch_skips_stage6_explicitly(self, monkeypatch):
        """Stage 6 (stale-entry) is NOT called for batch inputs — the store's
        _pin_matched_entries handles per-op matching (write_gate.py:499-500).
        This locks the skip as a contract, not an accident."""
        store = _make_store(limit=5000, entries=["live entry one"])
        ops = [{"action": "patch", "old_text": "drifted span", "content": "new"}]

        calls = []
        real_stage6 = _stage6_stale_entry
        def spy(entries, old_text):
            calls.append((entries, old_text))
            return real_stage6(entries, old_text)
        monkeypatch.setattr("tools.write_gate._stage6_stale_entry", spy)

        refusal = _run(store, operations=ops, target="memory")
        assert refusal is None, "gate delegates batch stale-matching to the store"
        assert calls == [], f"Stage 6 must be skipped on batch paths, got {len(calls)} call(s)"

    def test_batch_doc_target_variant(self, tmp_path, monkeypatch):
        """Doc-flavour batch: a drifted old_text against a chain doc is refused
        by the store through the full tool path (INTEG-4 doc batch variant)."""
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        store = MemoryStore(memory_char_limit=5000, user_char_limit=3000)
        store.load_from_disk()
        doc = "conventions-2026-09"
        store.add("memory", f"Dated memory chain: @doc:{doc}")
        store.add(f"doc:{doc}", "Alpha span in the doc")
        ops = [{"action": "patch", "old_text": "no such span", "content": "new"}]
        result = json.loads(memory_tool(store=store, target="memory", doc=doc, operations=ops))
        assert result["success"] is False


# =========================================================================
# T1-6: failure-class taxonomy — the code splits by gate; both tokens valid.
# =========================================================================

class TestFailureClassToken:
    def test_acceptance_refusal_uses_stage_specific_class(self):
        """An acceptance-gate refusal carries the STAGE's failure class
        (over_budget / scan_blocked / stale_entry), not the generic default."""
        store = _make_store(limit=4000, entries=["x" * 3000])
        refusal = _run(store, content="x" * 800)
        assert refusal.failure_class == "over_budget"

    def test_write_gated_default_present(self):
        """The GateRefusal default is 'write_gated' (the package's token)."""
        r = GateRefusal(action="REFUSE", rationale="test")
        assert r.failure_class == "write_gated"

    def test_both_tokens_in_registry(self):
        """T1-6 restatement: both 'gate_refused' (platform gate) and 'write_gated'
        (acceptance gate) are valid MEMORY_OP_FAILURE_CLASSES; the opt-in stays."""
        from hermes_cli.observability.shared_metrics_contract import MEMORY_OP_FAILURE_CLASSES
        assert "gate_refused" in MEMORY_OP_FAILURE_CLASSES
        assert "write_gated" in MEMORY_OP_FAILURE_CLASSES
        assert "readback_failed" in MEMORY_OP_FAILURE_CLASSES
        assert "date_currency" in MEMORY_OP_FAILURE_CLASSES


# =========================================================================
# T2-4 / OD-9: chained-memory anchor-cost boundary (arithmetic receipt anchor).
#    VERIFIED receipt: v3 measured exactly 250 fuller entries (37 chars + 3-char
#    delimiter = 40/entry) at 9,997/10,000; the 251st is refused cleanly.
# =========================================================================

class TestCapBoundedChainLength:
    def test_fuller_entry_shape_cap_boundary_formula(self):
        entry_len = 37
        delim_len = 3
        cap = 10_000
        expected = (cap + delim_len) // (entry_len + delim_len)
        assert expected == 250


# =========================================================================
# Author generic test 4: every accepted/rejected write appends exactly one
# change-log line (NON-VACUOUS — replaces the tautological guard).
# =========================================================================

class TestEveryWriteAppendsExactlyOneChangeLogLine:
    def test_plain_admit_appends_exactly_one_line(self, tmp_path, monkeypatch):
        """A plain admit appends exactly one ADMIT line (T1-1)."""
        _make_audit_path(tmp_path, monkeypatch)
        store = _make_store()
        _run(store, content="A simple factual note about the project.")
        lines = _read_log(tmp_path)
        assert len(lines) == 1
        assert lines[0]["action"] == "ADMIT"
        assert lines[0]["rationale"] == "accepted"

    def test_admit_with_date_still_one_line(self, tmp_path, monkeypatch):
        """An admit with a recent date is STILL exactly one line (the date flag
        is folded into the ADMIT line, not a second line)."""
        from datetime import datetime, timezone
        _make_audit_path(tmp_path, monkeypatch)
        store = _make_store()
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        _run(store, content=f"Dated note {today}")
        lines = _read_log(tmp_path)
        assert len(lines) == 1
        assert lines[0]["action"] == "ADMIT"

    def test_refusal_appends_exactly_one_line(self, tmp_path, monkeypatch):
        """A refusal appends exactly one line."""
        _make_audit_path(tmp_path, monkeypatch)
        store = _make_store(limit=100, entries=["x" * 90])
        _run(store, content="y" * 50)
        lines = _read_log(tmp_path)
        assert len(lines) == 1
        assert lines[0]["action"] == "REFUSE"

    def test_batch_appends_one_line_per_op(self, tmp_path, monkeypatch):
        """A batch admits as one unit → exactly one ADMIT line."""
        _make_audit_path(tmp_path, monkeypatch)
        store = _make_store()
        ops = [{"action": "add", "content": "one"}, {"action": "add", "content": "two"}]
        _run(store, operations=ops, target="memory")
        lines = _read_log(tmp_path)
        assert len(lines) == 1
        assert lines[0]["action"] == "ADMIT"

    def test_three_refusals_three_lines(self, tmp_path, monkeypatch):
        """Three separate refusals → exactly three REFUSE lines."""
        _make_audit_path(tmp_path, monkeypatch)
        _run(_make_store(), content="Run the deploy script", caller="background_review")
        _run(_make_store(), content="1. Step one\n2. Step two", caller="user")
        _run(_make_store(limit=50, entries=["x" * 49]), content="overflow content here")
        lines = _read_log(tmp_path)
        refusals = [l for l in lines if l.get("action") == "REFUSE"]
        assert len(refusals) == 3


# =========================================================================
# FALSIFIER GUARD: test_gate_bypass_proves_enforcement
#    Documents the falsifier triad: if the gate is bypassed (returns None
#    always), at least one refusal-expected fixture FAILS, proving the gate
#    enforces — not just exists.  (T1-4A: the monkeypatch target is the bound
#    reference the test actually calls, so the bypass genuinely swaps.)
# =========================================================================

class TestGateBypassProvesEnforcement:
    def test_bypass_proves_enforcement(self, monkeypatch):
        """Falsifier triad in one passing test: baseline REFUSES the over-cap
        write; swapping the binding the test actually calls makes the refusal
        DISAPPEAR — proving the gate (not a no-op) is what enforces it.

        T1-4A: the monkeypatch target is the module-bound name ``_run`` calls
        (``acceptance_pipeline`` imported into this module), so the swap is
        REAL and the RED evidence is auditable — not the self-attesting patch
        of ``tools.write_gate.acceptance_pipeline`` that never took effect.
        """
        # Baseline: the real gate refuses the over-cap write.
        assert _run(_make_store(limit=4000, entries=["x" * 3000]), content="x" * 800) is not None
        # Bypass: swap the bound reference; the refusal disappears.
        monkeypatch.setattr(
            "tests.tools.test_write_gate.acceptance_pipeline",
            lambda *a, **kw: None,
        )
        assert _run(_make_store(limit=4000, entries=["x" * 3000]), content="x" * 800) is None


# =========================================================================
# T1-4B: wiring proof — the Phase-7 hook actually invokes the gate before the
# store write, through the real memory_tool entry (not by calling the pipeline
# directly).  A future refactor that drops the hook makes this fail.
# =========================================================================

class TestGateWiringProof:
    def test_hook_fires_gate_on_over_budget_write(self, tmp_path, monkeypatch):
        """Over-budget content via the REAL writer returns a refusal tool_error
        string; the store is never touched (the gate fired BEFORE the write)."""
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        store = MemoryStore(memory_char_limit=100, user_char_limit=100)
        store.load_from_disk()
        store.add("memory", "x" * 90)
        before = store._char_count("memory")

        result = json.loads(memory_tool(store=store, action="add", target="memory", content="y" * 50))
        assert result["success"] is False
        assert "char-cap" in result.get("error", "")
        # The store was never mutated: the gate refused before the write.
        assert store._char_count("memory") == before

    def test_hook_dropped_makes_wiring_fail(self, tmp_path, monkeypatch):
        """Falsifier: if the Phase-7 gate call is dropped from the hook, the
        over-budget write proceeds (no refusal) — proving the hook is wired in
        and the suite would catch its removal."""
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        # Simulate the hook being dropped by neutering the gate import to a no-op.
        import tools.write_gate as wg
        monkeypatch.setattr(wg, "acceptance_pipeline", lambda *a, **kw: None)
        store = MemoryStore(memory_char_limit=100, user_char_limit=100)
        store.load_from_disk()
        store.add("memory", "x" * 90)

        json.loads(memory_tool(store=store, action="add", target="memory", content="y" * 50))
        # With the gate neutered, the write is NOT refused by the acceptance gate
        # (it proceeds to the store's own cap, which refuses with its own message).
        # Either way the over-cap content does not silently land.
        assert store._char_count("memory") <= 100


# =========================================================================
# INTEG-4: THE DOC-TARGET BATTERY — the integration proof.
#    Drives chain-doc writes through the FULL tool path in the doc-chain world
#    (the sandbox head with doc-aware store primitives).  This is the probe the
#    anchor-only battery never ran — the absence that let INTEG-1/2 ship green.
#    It proves the gate scopes size/limit/stale-entry PER TARGET.
# =========================================================================

class TestDocTargetPathUnderGate:
    """Chain-doc writes through the full memory_tool path, against a store in the
    doc-chain world.  Cites INTEG-1..3 and the sandbox-head store primitives
    (``_char_limit`` None for docs, ``_entries_for`` per-target)."""

    def _store(self, tmp_path, monkeypatch, limit=10000, fill=9):
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        store = MemoryStore(memory_char_limit=limit, user_char_limit=1375)
        store.load_from_disk()
        # Reference the chain doc from the anchor (required by _anchor_gate).
        memory_tool(store=store, action="add", target="memory", content="Dated memory chain: @doc:001-conventions")
        # Push the anchor near its cap with varied filler (dedup-safe).
        for i in range(fill):
            memory_tool(store=store, action="add", target="memory", content="filler-" + str(i) + "-" + "f" * 900)
        return store

    def test_doc_add_over_anchor_cap_admits_anchor_unchanged(self, tmp_path, monkeypatch):
        """INTEG-1: a 5000-char add to doc:001-conventions while the anchor is at
        ~85% ADMITS (chain docs are uncapped) and the anchor is UNCHANGED."""
        store = self._store(tmp_path, monkeypatch)
        anchor_before = store._char_count("memory")
        assert anchor_before > int(10000 * 0.85) - 1000  # anchor is near/over soft threshold

        result = json.loads(memory_tool(
            store=store, action="add", target="memory", doc="001-conventions", content="z" * 5000))
        assert result["success"] is True, result
        # The anchor never grew: the doc's size never lands on the anchor's budget.
        assert store._char_count("memory") == anchor_before
        # The doc itself holds the 5000 chars.
        assert "z" * 5000 in (tmp_path / "docs" / "001-conventions.md").read_text(encoding="utf-8")

    def test_doc_patch_span_admits(self, tmp_path, monkeypatch):
        """INTEG-2: a patch of a span residing in doc:001-conventions ADMITS —
        the stale-entry search anchors against the DOC's entries, not the anchor's."""
        store = self._store(tmp_path, monkeypatch)
        memory_tool(store=store, action="add", target="memory", doc="001-conventions", content="ALPHA beta gamma")
        result = json.loads(memory_tool(
            store=store, action="patch", target="memory", doc="001-conventions",
            old_text="ALPHA beta gamma", content="ALPHA beta DELTA"))
        assert result["success"] is True, result
        doc_text = (tmp_path / "docs" / "001-conventions.md").read_text(encoding="utf-8")
        assert "ALPHA beta DELTA" in doc_text
        assert "ALPHA beta gamma" not in doc_text

    def test_doc_remove_span_admits(self, tmp_path, monkeypatch):
        """INTEG-2: a remove of a doc-resident entry ADMITS (anchored on the doc)."""
        store = self._store(tmp_path, monkeypatch)
        memory_tool(store=store, action="add", target="memory", doc="001-conventions", content="REMOVABLE entry text")
        result = json.loads(memory_tool(
            store=store, action="remove", target="memory", doc="001-conventions",
            old_text="REMOVABLE entry text"))
        assert result["success"] is True, result

    def test_doc_add_over_budget_at_low_anchor_admits(self, tmp_path, monkeypatch):
        """INTEG-1: at a LOW anchor the doc add still admits — confirming no
        anchor growth and that the doc path never depends on anchor headroom."""
        store = self._store(tmp_path, monkeypatch, fill=0)
        anchor_before = store._char_count("memory")
        result = json.loads(memory_tool(
            store=store, action="add", target="memory", doc="001-conventions", content="w" * 8000))
        assert result["success"] is True, result
        assert store._char_count("memory") == anchor_before

    def test_unreferenced_doc_refused_by_store_gate_first(self, tmp_path, monkeypatch):
        """INTEG-4(c): a write to an UNREFERENCED doc is refused by the store's
        _anchor_gate FIRST (the acceptance gate's mis-scoping never masks it)."""
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        store = MemoryStore(memory_char_limit=10000, user_char_limit=1375)
        store.load_from_disk()
        # No @doc:orphan reference in the anchor.
        result = json.loads(memory_tool(
            store=store, action="add", target="memory", doc="orphan", content="nope"))
        assert result["success"] is False
        assert "@doc:orphan" in result.get("error", "")
        assert not (tmp_path / "docs" / "orphan.md").exists()

    def test_doc_write_logs_one_admit_line(self, tmp_path, monkeypatch):
        """A chain-doc add through the full path appends exactly one ADMIT line,
        scoped to the doc target."""
        _make_audit_path(tmp_path, monkeypatch)
        store = self._store(tmp_path, monkeypatch, fill=0)
        memory_tool(store=store, action="add", target="memory", doc="001-conventions", content="v" * 100)
        lines = _read_log(tmp_path)
        admits = [l for l in lines if l.get("action") == "ADMIT"]
        assert len(admits) >= 1
        assert admits[-1].get("target") == "doc:001-conventions"
