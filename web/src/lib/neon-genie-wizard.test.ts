import { describe, expect, it } from "vitest";
import { buildNeonGeniePrompt, type NeonGenieBrief } from "./neon-genie-wizard";

const baseBrief: NeonGenieBrief = {
  mission: "commercial",
  requestedOutcome: "Determine whether we can support a paid diagnostic.",
  targetUser: "Operations leaders",
  currentState: "The product exists, but the buyer and budget authority are unknown.",
  desiredState: "A testable offer with explicit evidence gaps.",
  evidence: "https://example.com/product",
  constraints: "Do not invent customers or modify repositories.",
  researchEnabled: true,
  maxFetches: "6",
  allowDrafting: true,
  requireHumanReview: true,
};

describe("buildNeonGeniePrompt", () => {
  it("builds a governed mission prompt with evidence and authority gates", () => {
    const prompt = buildNeonGeniePrompt(baseBrief);

    expect(prompt).toContain("commercial model run");
    expect(prompt).toContain("starting with: commercial.");
    expect(prompt).toContain("OBSERVED, INFERRED, SPECULATIVE, or NOT_COMPUTABLE");
    expect(prompt).toContain("DataRequests");
    expect(prompt).toContain("max_fetches=6");
    expect(prompt).toContain("execution=false, spending=false, publishing=false");
    expect(prompt).toContain("Do not spend, publish, contact anyone, mutate repositories");
  });

  it("omits empty optional context and disables research explicitly", () => {
    const prompt = buildNeonGeniePrompt({
      ...baseBrief,
      targetUser: " ",
      evidence: "",
      researchEnabled: false,
      maxFetches: "",
    });

    expect(prompt).not.toContain("Target user or beneficiary:");
    expect(prompt).not.toContain("Known evidence and canonical sources:");
    expect(prompt).toContain("Research: enabled=false.");
    expect(prompt).toContain("Authority: research=false");
  });

  it("uses the smallest declared profile set for routed missions", () => {
    const expectedProfiles = {
      opportunity: "opportunity_mining",
      "zero-option": "zero_option",
      commercial: "commercial",
      fragmentation: "fragmentation",
      evidence: "evidence_intelligence",
      agentic: "agentic_services",
      audit: "audit_delivery",
    } as const;

    for (const [mission, profile] of Object.entries(expectedProfiles)) {
      const prompt = buildNeonGeniePrompt({
        ...baseBrief,
        mission: mission as NeonGenieBrief["mission"],
      });
      expect(prompt).toContain(`starting with: ${profile}.`);
    }
  });
});
