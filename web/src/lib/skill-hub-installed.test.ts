import { describe, expect, it } from "vitest";

import type { SkillHubInstalledEntry } from "@/lib/api";
import { installedHubEntry } from "@/lib/skill-hub-installed";

const entry = (name: string): SkillHubInstalledEntry => ({
  name,
  trust_level: "community",
  scan_verdict: "safe",
});

// The map the backend returns with one ClawHub install and one official skill:
// each lock id, plus the ClawHub row under the clawhub/<slug> card id.
const installed = {
  "@tmstudio667-commits/agent-git-oracle": entry("agent-git-oracle"),
  "clawhub/agent-git-oracle": entry("agent-git-oracle"),
  "official/research/arxiv": entry("arxiv"),
};

describe("installedHubEntry", () => {
  it("finds a ClawHub hit's bare slug under the card id", () => {
    const hit = { identifier: "agent-git-oracle", name: "Agent Git Oracle", source: "clawhub" };

    expect(installedHubEntry(installed, hit)?.name).toBe("agent-git-oracle");
  });

  it("never lets a ClawHub install answer for another source's bare id", () => {
    const hit = { identifier: "agent-git-oracle", name: "agent-git-oracle", source: "skills-sh" };

    expect(installedHubEntry(installed, hit)).toBeUndefined();
  });

  it("matches an owner-qualified hit and other sources by their exact id", () => {
    expect(
      installedHubEntry(installed, {
        identifier: "@tmstudio667-commits/agent-git-oracle",
        name: "Agent Git Oracle",
        source: "clawhub",
      })?.name,
    ).toBe("agent-git-oracle");
    expect(
      installedHubEntry(installed, { identifier: "official/research/arxiv", name: "arxiv", source: "official" })
        ?.name,
    ).toBe("arxiv");
    expect(
      installedHubEntry(installed, { identifier: "official/research/arxiv-lite", name: "x", source: "official" }),
    ).toBeUndefined();
  });
});
