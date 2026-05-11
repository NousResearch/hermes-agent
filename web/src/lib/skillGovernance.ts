type ProposalLike = {
  decision_status?: string | null;
  allowed_decision_statuses?: string[] | null;
  diff_text?: string | null;
  artifact_texts?: Record<string, string> | null;
};

export const PROPOSAL_DECISION_ORDER = [
  "pending",
  "approved",
  "needs_changes",
  "deferred",
  "bad_test_target",
  "rejected",
] as const;

export function proposalStatusLabel(status: string): string {
  const labels: Record<string, string> = {
    pending: "Pending PM decision",
    approved: "Approved",
    rejected: "Rejected",
    deferred: "Deferred",
    needs_changes: "Needs changes",
    bad_test_target: "Bad test target",
  };
  return labels[status] ?? status.replace(/_/g, " ");
}

export function proposalDecisionActionLabel(status: string): string {
  const labels: Record<string, string> = {
    pending: "Reopen",
    approved: "Approve record",
    rejected: "Reject",
    deferred: "Defer",
    needs_changes: "Needs changes",
    bad_test_target: "Bad test target",
  };
  return labels[status] ?? proposalStatusLabel(status);
}

export function proposalDecisionNote(status: string): string {
  const notes: Record<string, string> = {
    pending: "Reopened for PM decision; MVP does not apply or mutate skills.",
    approved: "Approved in dashboard as a PM decision only; MVP does not apply or mutate skills.",
    rejected: "Rejected in dashboard; no skill mutation should be attempted for this proposal.",
    deferred: "Deferred in dashboard for later review; no skill mutation should be attempted now.",
    needs_changes: "Needs changes before approval; keep proposal as review artifact only.",
    bad_test_target: "Marked as a poor first Curator validation target; no skill mutation should be attempted.",
  };
  return notes[status] ?? "Decision recorded in dashboard; MVP does not apply or mutate skills.";
}

export function allowedProposalDecisionStatuses(proposal: ProposalLike): string[] {
  const allowed = new Set(proposal.allowed_decision_statuses ?? PROPOSAL_DECISION_ORDER);
  return PROPOSAL_DECISION_ORDER.filter((status) => allowed.has(status));
}

export function canRecordProposalDecision(proposal: ProposalLike, status: string): boolean {
  return allowedProposalDecisionStatuses(proposal).includes(status as (typeof PROPOSAL_DECISION_ORDER)[number]);
}

export function proposalDecisionActionStatuses(proposal: ProposalLike): string[] {
  return allowedProposalDecisionStatuses(proposal).filter((status) => status !== proposal.decision_status);
}

export function unavailableProposalDecisionStatuses(proposal: ProposalLike): string[] {
  const allowed = new Set(allowedProposalDecisionStatuses(proposal));
  return PROPOSAL_DECISION_ORDER.filter((status) => !allowed.has(status));
}

export function proposalDecisionTransitionHint(proposal: ProposalLike): string | null {
  const unavailable = unavailableProposalDecisionStatuses(proposal);
  const currentStatus = proposal.decision_status ? proposalStatusLabel(proposal.decision_status) : "this status";
  if (unavailable.includes("approved") && canRecordProposalDecision(proposal, "pending")) {
    return `Approved is not available from ${currentStatus}. Reopen to Pending PM decision first if you want to approve it.`;
  }
  if (unavailable.length > 0) {
    return `Unavailable from ${currentStatus}: ${unavailable.map(proposalStatusLabel).join(", ")}.`;
  }
  return null;
}

export function proposalPreviewText(proposal: ProposalLike): {
  title: string;
  body: string;
  kind: "diff" | "artifact" | "empty";
} {
  const diff = proposal.diff_text?.trim();
  if (diff) {
    return { title: "Unified diff", body: proposal.diff_text ?? "", kind: "diff" };
  }

  const artifacts = proposal.artifact_texts ?? {};
  const preferred = artifacts["curator_report_excerpt.md"];
  if (preferred?.trim()) {
    return { title: "Dry-run excerpt", body: preferred, kind: "artifact" };
  }

  const first = Object.entries(artifacts).find(([, value]) => value.trim());
  if (first) {
    return { title: `Artifact: ${first[0]}`, body: first[1], kind: "artifact" };
  }

  return {
    title: "No diff artifact",
    body: "No unified diff has been generated for this proposal yet. This MVP is decision-only and does not apply or mutate skills.",
    kind: "empty",
  };
}

export function extractFetchErrorMessage(error: unknown): string {
  if (!(error instanceof Error)) return "Unknown error";
  const message = error.message;
  const jsonStart = message.indexOf("{");
  if (jsonStart >= 0) {
    try {
      const parsed = JSON.parse(message.slice(jsonStart));
      if (typeof parsed.detail === "string" && parsed.detail.trim()) {
        return parsed.detail;
      }
    } catch {
      // Fall through to raw error message.
    }
  }
  return message;
}
