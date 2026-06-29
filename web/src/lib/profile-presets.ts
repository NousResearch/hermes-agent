export interface ProfileExample {
  name: string;
  title: string;
  description: string;
  bestFor: string;
}

export interface ProfileBundleRole {
  slug: string;
  title: string;
  description: string;
  focus: string[];
}

export interface ProfileBundle {
  id: string;
  title: string;
  summary: string;
  prefix: string;
  roles: ProfileBundleRole[];
}

export const PROFILE_FIELD_GUIDANCE = {
  name: {
    hint: "Use a short role name you would actually summon.",
    tooltip: "Good names are concrete and lowercase: coder, repo-researcher, support-triage.",
  },
  description: {
    hint:
      "Say what this profile is great at, what context it should prefer, and what good output looks like.",
    tooltip:
      "One focused sentence beats a vague personality. Example: Great at reading unfamiliar codebases, finding the smallest safe fix, and explaining tradeoffs.",
  },
  cloneFrom: {
    hint:
      "Clone when this profile should share keys, tools, and setup with an existing agent.",
    tooltip:
      "Start blank only when you want a profile to build its own habits, skills, and configuration from scratch.",
  },
  model: {
    hint:
      "Pick for the job style: stronger reasoning for coding or research, faster models for routine drafting.",
    tooltip:
      "You can leave this unset and choose later from the Models page or with the profile command.",
  },
  skills: {
    hint:
      "Start with the skills this profile will actually use; focused capability makes behavior easier to predict.",
    tooltip:
      "Coder profiles usually want repo, test, and review skills. Research profiles usually want search, docs, and summarization skills.",
  },
  mcp: {
    hint:
      "Add external systems only when they are part of this profile's job.",
    tooltip:
      "Examples: GitHub for code review, Jira for delivery work, Drive for docs-heavy research.",
  },
  soul: {
    hint:
      "Use SOUL.md for stable preferences: tone, decision rules, boundaries, and when to ask before acting.",
    tooltip:
      "Keep it durable. Session-specific instructions belong in the conversation, not the profile identity.",
  },
} as const;

export const PROFILE_EXAMPLES: ProfileExample[] = [
  {
    name: "coder",
    title: "Coder",
    description:
      "Great at reading unfamiliar codebases, identifying the smallest safe fix, and explaining tradeoffs before changing files.",
    bestFor: "Implementation, debugging, refactors",
  },
  {
    name: "repo-researcher",
    title: "Researcher",
    description:
      "Great at scanning docs, issues, and prior decisions, then turning messy context into concise implementation plans.",
    bestFor: "Discovery, specs, technical briefs",
  },
  {
    name: "ops-assistant",
    title: "Ops Assistant",
    description:
      "Great at watching recurring workflows, triaging exceptions, and producing clear next actions with owners.",
    bestFor: "Triage, status, follow-through",
  },
];

export const PROFILE_BUNDLES: ProfileBundle[] = [
  {
    id: "development",
    title: "Development Team",
    summary:
      "A product-to-release crew for building features with planning, architecture, implementation, QA, and validation roles.",
    prefix: "dev",
    roles: [
      {
        slug: "product-manager",
        title: "Product Manager",
        description:
          "Great at turning user goals into scoped requirements, acceptance criteria, and release-ready priorities.",
        focus: ["requirements", "scope", "acceptance criteria"],
      },
      {
        slug: "architect",
        title: "Architect",
        description:
          "Great at mapping system boundaries, spotting integration risks, and choosing designs that preserve existing contracts.",
        focus: ["architecture", "interfaces", "risk"],
      },
      {
        slug: "developer",
        title: "Developer",
        description:
          "Great at implementing focused code changes, following local patterns, and verifying behavior with relevant tests.",
        focus: ["implementation", "tests", "repo patterns"],
      },
      {
        slug: "qa",
        title: "QA",
        description:
          "Great at finding edge cases, designing regression checks, and translating vague behavior into testable scenarios.",
        focus: ["test plans", "edge cases", "regressions"],
      },
      {
        slug: "validator",
        title: "Validator",
        description:
          "Great at independently reviewing finished work against the original goal, evidence, and release risk.",
        focus: ["final review", "evidence", "release risk"],
      },
    ],
  },
  {
    id: "finance",
    title: "Finance Ops",
    summary:
      "A finance workflow crew for source review, accounting treatment, analysis, reporting, and controls validation.",
    prefix: "fin",
    roles: [
      {
        slug: "data-steward",
        title: "Data Steward",
        description:
          "Great at checking source completeness, naming messy inputs, and flagging gaps before analysis begins.",
        focus: ["source data", "completeness", "lineage"],
      },
      {
        slug: "accountant",
        title: "Accountant",
        description:
          "Great at applying accounting treatment, reconciling entries, and explaining assumptions in plain language.",
        focus: ["reconciliation", "treatment", "assumptions"],
      },
      {
        slug: "analyst",
        title: "Analyst",
        description:
          "Great at modeling trends, comparing scenarios, and calling out drivers that matter to the decision.",
        focus: ["analysis", "variance", "drivers"],
      },
      {
        slug: "reporting-lead",
        title: "Reporting Lead",
        description:
          "Great at turning finance work into concise summaries, tables, and executive-ready narratives.",
        focus: ["reporting", "summaries", "tables"],
      },
      {
        slug: "controls-reviewer",
        title: "Controls Reviewer",
        description:
          "Great at checking approvals, separation of duties, audit trail quality, and unresolved control risks.",
        focus: ["controls", "audit trail", "approval risk"],
      },
    ],
  },
];

export function sanitizeProfilePrefix(value: string, fallback: string): string {
  const cleaned = value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/^-+/, "")
    .replace(/-+$/, "");
  return cleaned || fallback;
}

export function bundleProfileName(prefix: string, role: ProfileBundleRole): string {
  return `${prefix}-${role.slug}`.slice(0, 64);
}
