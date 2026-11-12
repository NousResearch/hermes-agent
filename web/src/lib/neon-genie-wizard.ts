export type NeonGenieMissionId =
  | "product-audit"
  | "opportunity"
  | "zero-option"
  | "commercial"
  | "fragmentation"
  | "evidence"
  | "agentic"
  | "audit"
  | "wayfinder";

export interface NeonGenieMission {
  id: NeonGenieMissionId;
  label: string;
  description: string;
  profiles: string[];
  outputs: string[];
}

export interface NeonGenieBrief {
  mission: NeonGenieMissionId;
  requestedOutcome: string;
  targetUser: string;
  currentState: string;
  desiredState: string;
  evidence: string;
  constraints: string;
  researchEnabled: boolean;
  maxFetches: string;
  allowDrafting: boolean;
  requireHumanReview: boolean;
}

export const NEON_GENIE_MISSIONS: NeonGenieMission[] = [
  {
    id: "product-audit",
    label: "Product audit",
    description: "Clarify the product boundary, buyer, commercial logic, and execution readiness.",
    profiles: ["product_architecture", "commercial", "wayfinder_handoff"],
    outputs: ["product packet", "commercial simulation", "Wayfinder handoff readiness"],
  },
  {
    id: "opportunity",
    label: "Opportunity mining",
    description: "Turn a blocked transition or weak signal into a testable opportunity.",
    profiles: ["opportunity_mining"],
    outputs: ["opportunity packet", "proof plan", "open DataRequests"],
  },
  {
    id: "zero-option",
    label: "Zero-option loop",
    description: "Find the smallest honest first-cash loop using only declared resources.",
    profiles: ["zero_option"],
    outputs: ["zero-option packet", "micro-loop", "resource gaps"],
  },
  {
    id: "commercial",
    label: "Commercial model",
    description: "Map buyer, beneficiary, budget authority, offer, and pricing evidence.",
    profiles: ["commercial"],
    outputs: ["commercial simulation", "buyer DataRequests", "pricing evidence"],
  },
  {
    id: "fragmentation",
    label: "Fragmentation scan",
    description: "Find costly handoffs and duplicated work across systems or teams.",
    profiles: ["fragmentation"],
    outputs: ["fragmentation packet", "priority transitions", "proof tests"],
  },
  {
    id: "evidence",
    label: "Evidence intelligence",
    description: "Separate public research from private facts and identify decision-critical gaps.",
    profiles: ["evidence_intelligence"],
    outputs: ["evidence packet", "claim ledger", "DataRequests"],
  },
  {
    id: "agentic",
    label: "Agentic service graph",
    description: "Decompose a service into useful agent actions and reject ornamental automation.",
    profiles: ["agentic_services"],
    outputs: ["agentic service graph", "authority boundaries", "commercial fit"],
  },
  {
    id: "audit",
    label: "Audit-first offer",
    description: "Package an evidence-bound diagnostic and cost-of-inaction analysis.",
    profiles: ["audit_delivery"],
    outputs: ["audit delivery packet", "diagnostic scope", "cost-of-inaction model"],
  },
  {
    id: "wayfinder",
    label: "Wayfinder handoff",
    description: "Freeze validated product intent into an engineering-ready advisory handoff.",
    profiles: ["product_architecture", "wayfinder_handoff"],
    outputs: ["Wayfinder execution packet", "blocked gates", "change-control rules"],
  },
];

function section(label: string, value: string): string | null {
  const clean = value.trim();
  return clean ? `${label}:\n${clean}` : null;
}

export function buildNeonGeniePrompt(brief: NeonGenieBrief): string {
  const mission =
    NEON_GENIE_MISSIONS.find((candidate) => candidate.id === brief.mission) ??
    NEON_GENIE_MISSIONS[0];
  const maxFetches = brief.maxFetches.trim();
  const parts = [
    `Use the neon-genie skill for a ${mission.label.toLowerCase()} run.`,
    `Load the smallest profile set needed, starting with: ${mission.profiles.join(", ")}.`,
    section("Requested outcome", brief.requestedOutcome),
    section("Target user or beneficiary", brief.targetUser),
    section("Current state", brief.currentState),
    section("Desired state", brief.desiredState),
    section("Known evidence and canonical sources", brief.evidence),
    section("Constraints and explicit exclusions", brief.constraints),
    `Requested outputs: ${mission.outputs.join(", ")}.`,
    `Research: enabled=${brief.researchEnabled ? "true" : "false"}${
      maxFetches ? `, max_fetches=${maxFetches}` : ""
    }.`,
    `Authority: research=${brief.researchEnabled ? "true" : "false"}, drafting=${
      brief.allowDrafting ? "true" : "false"
    }, execution=false, spending=false, publishing=false.`,
    `Human review required: ${brief.requireHumanReview ? "true" : "false"}.`,
    "Run in order: OPEN (understand) → ALIGN (gather evidence) → ASCEND (build packets) → CLEAR (check gates) → SEAL (summarize and identify open requests).",
    "Label every decision-critical claim OBSERVED, INFERRED, SPECULATIVE, or NOT_COMPUTABLE. Cite sources for OBSERVED claims. Research public facts when enabled. Convert missing private facts into explicit DataRequests instead of guessing. Fail closed when buyer, proof, access, or authority is missing.",
    "Remain advisory only. Do not spend, publish, contact anyone, mutate repositories, or imply execution rights. Return the main packet, a concise run receipt, blocking DataRequests, and the next smallest proof action.",
  ];

  return parts.filter((part): part is string => Boolean(part)).join("\n\n");
}
