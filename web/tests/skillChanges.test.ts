import assert from "node:assert/strict";
import test from "node:test";

import {
  latestChangeBySkill,
  loadSkillChangesBestEffort,
  reasonKindLabel,
  reviewStatusLabel,
} from "../src/lib/skillChanges.ts";

test("reasonKindLabel keeps explicit provenance separate from missing reasons", () => {
  assert.equal(reasonKindLabel("explicit"), "Explicit reason");
  assert.equal(reasonKindLabel("system"), "System-detected change");
  assert.equal(reasonKindLabel("unattributed"), "No reason captured");
  assert.equal(reasonKindLabel("model_summary"), "Model-generated summary");
  assert.equal(reasonKindLabel("unknown"), "Unknown reason source");
});

test("reviewStatusLabel maps dashboard review states", () => {
  assert.equal(reviewStatusLabel("unreviewed"), "Unreviewed");
  assert.equal(reviewStatusLabel("reviewed"), "Reviewed");
  assert.equal(reviewStatusLabel("needs_followup"), "Needs follow-up");
  assert.equal(reviewStatusLabel("other"), "other");
});

test("loadSkillChangesBestEffort degrades to an empty history when the endpoint fails", async () => {
  let notified = false;
  const changes = await loadSkillChangesBestEffort(
    async () => {
      throw new Error("history unavailable");
    },
    () => {
      notified = true;
    },
  );

  assert.deepEqual(changes, []);
  assert.equal(notified, true);
});

test("latestChangeBySkill keeps the newest event per skill from a newest-first list", () => {
  const changes = [
    {
      event_id: "new-alpha",
      timestamp: "2026-04-27T08:00:00Z",
      skill: "alpha",
      category: null,
      action: "patch",
      actor: "hermes-agent",
      source: "unit-test",
      session_id: null,
      reason: "new",
      reason_kind: "explicit",
      before_hash: null,
      after_hash: null,
      changed_files: [],
      diff_path: null,
      metadata: {},
      review_status: "unreviewed",
      reviewed_at: null,
      review_note: null,
    },
    {
      event_id: "beta",
      timestamp: "2026-04-27T07:00:00Z",
      skill: "beta",
      category: null,
      action: "edit",
      actor: "hermes-agent",
      source: "unit-test",
      session_id: null,
      reason: "beta",
      reason_kind: "explicit",
      before_hash: null,
      after_hash: null,
      changed_files: [],
      diff_path: null,
      metadata: {},
      review_status: "reviewed",
      reviewed_at: "2026-04-27T07:10:00Z",
      review_note: null,
    },
    {
      event_id: "old-alpha",
      timestamp: "2026-04-27T06:00:00Z",
      skill: "alpha",
      category: null,
      action: "create",
      actor: "hermes-agent",
      source: "unit-test",
      session_id: null,
      reason: "old",
      reason_kind: "explicit",
      before_hash: null,
      after_hash: null,
      changed_files: [],
      diff_path: null,
      metadata: {},
      review_status: "reviewed",
      reviewed_at: "2026-04-27T06:10:00Z",
      review_note: null,
    },
  ] as const;

  const bySkill = latestChangeBySkill(changes);

  assert.equal(bySkill.get("alpha")?.event_id, "new-alpha");
  assert.equal(bySkill.get("beta")?.event_id, "beta");
  assert.equal(bySkill.size, 2);
});
