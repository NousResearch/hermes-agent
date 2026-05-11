import assert from "node:assert/strict";
import test from "node:test";

import {
  canRecordProposalDecision,
  extractFetchErrorMessage,
  proposalDecisionActionLabel,
  proposalDecisionActionStatuses,
  proposalDecisionTransitionHint,
  proposalPreviewText,
  unavailableProposalDecisionStatuses,
} from "../src/lib/skillGovernance.ts";

test("canRecordProposalDecision follows allowed statuses from the backend", () => {
  const proposal = {
    allowed_decision_statuses: ["pending", "deferred", "rejected"],
  };

  assert.equal(canRecordProposalDecision(proposal, "approved"), false);
  assert.equal(canRecordProposalDecision(proposal, "deferred"), true);
});

test("proposalPreviewText prefers a unified diff and falls back to the Curator dry-run excerpt", () => {
  assert.deepEqual(
    proposalPreviewText({ diff_text: "--- old\n+++ new" }),
    { title: "Unified diff", body: "--- old\n+++ new", kind: "diff" },
  );

  assert.deepEqual(
    proposalPreviewText({ artifact_texts: { "curator_report_excerpt.md": "dry-run excerpt" } }),
    { title: "Dry-run excerpt", body: "dry-run excerpt", kind: "artifact" },
  );

  assert.equal(proposalPreviewText({}).kind, "empty");
});

test("extractFetchErrorMessage surfaces API detail instead of hiding it behind a generic toast", () => {
  const message = extractFetchErrorMessage(
    new Error('400: {"detail":"Invalid Skill Governance transition \'bad_test_target\' -> \'approved\'"}'),
  );

  assert.equal(message, "Invalid Skill Governance transition 'bad_test_target' -> 'approved'");
});

test("bad-test-target proposals expose a visible reopen route before approval", () => {
  const proposal = {
    decision_status: "bad_test_target",
    allowed_decision_statuses: ["pending", "deferred", "bad_test_target", "rejected"],
  };

  assert.equal(proposalDecisionActionLabel("pending"), "Reopen");
  assert.deepEqual(proposalDecisionActionStatuses(proposal), ["pending", "deferred", "rejected"]);
  assert.deepEqual(unavailableProposalDecisionStatuses(proposal), ["approved", "needs_changes"]);
  assert.equal(
    proposalDecisionTransitionHint(proposal),
    "Approved is not available from Bad test target. Reopen to Pending PM decision first if you want to approve it.",
  );
});
