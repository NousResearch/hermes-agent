export interface ProfileBundleRole {
  slug: string;
  title: string;
  description: string;
  focus: readonly string[];
  recommendedSkills?: readonly string[];
  recommendedMcpServers?: readonly string[];
}

export interface ProfileBundleDefinition {
  id: string;
  title: string;
  summary: string;
  defaultPrefix: string;
  roles: readonly ProfileBundleRole[];
}

export interface ProfileBundlePlanItem {
  exists: boolean;
  name: string;
  role: ProfileBundleRole;
}

/**
 * Extensible catalog for dashboard profile bundles.
 *
 * To add a new starter team, append one ProfileBundleDefinition here. Keep the
 * bundle as plain data: the UI and creation flow below derive profile names,
 * conflict warnings, and role descriptions from the same shape.
 */
const PROFILE_BUNDLE_DEFINITIONS: readonly ProfileBundleDefinition[] = [
  {
    id: "development",
    title: "Development Team",
    summary:
      "A product-to-release crew for building features with planning, architecture, implementation, QA, and validation roles.",
    defaultPrefix: "dev",
    roles: [
      {
        slug: "product-manager",
        title: "Product Manager",
        description:
          "Great at turning user goals into scoped requirements, acceptance criteria, and release-ready priorities.",
        focus: ["requirements", "scope", "acceptance criteria"],
        recommendedSkills: ["requirements", "planning", "status"],
      },
      {
        slug: "architect",
        title: "Architect",
        description:
          "Great at mapping system boundaries, spotting integration risks, and choosing designs that preserve existing contracts.",
        focus: ["architecture", "interfaces", "risk"],
        recommendedSkills: ["architecture", "code review", "docs"],
      },
      {
        slug: "developer",
        title: "Developer",
        description:
          "Great at implementing focused code changes, following local patterns, and verifying behavior with relevant tests.",
        focus: ["implementation", "tests", "repo patterns"],
        recommendedSkills: ["coding", "git", "tests"],
      },
      {
        slug: "qa",
        title: "QA",
        description:
          "Great at finding edge cases, designing regression checks, and translating vague behavior into testable scenarios.",
        focus: ["test plans", "edge cases", "regressions"],
        recommendedSkills: ["testing", "bug triage", "release notes"],
      },
      {
        slug: "validator",
        title: "Validator",
        description:
          "Great at independently reviewing finished work against the original goal, evidence, and release risk.",
        focus: ["final review", "evidence", "release risk"],
        recommendedSkills: ["review", "verification", "risk"],
      },
    ],
  },
  {
    id: "finance",
    title: "Finance Ops",
    summary:
      "A finance workflow crew for source review, accounting treatment, analysis, reporting, and controls validation.",
    defaultPrefix: "fin",
    roles: [
      {
        slug: "data-steward",
        title: "Data Steward",
        description:
          "Great at checking source completeness, naming messy inputs, and flagging gaps before analysis begins.",
        focus: ["source data", "completeness", "lineage"],
        recommendedSkills: ["data review", "source audit", "spreadsheets"],
      },
      {
        slug: "accountant",
        title: "Accountant",
        description:
          "Great at applying accounting treatment, reconciling entries, and explaining assumptions in plain language.",
        focus: ["reconciliation", "treatment", "assumptions"],
        recommendedSkills: ["accounting", "reconciliation", "audit notes"],
      },
      {
        slug: "analyst",
        title: "Analyst",
        description:
          "Great at modeling trends, comparing scenarios, and calling out drivers that matter to the decision.",
        focus: ["analysis", "variance", "drivers"],
        recommendedSkills: ["analysis", "forecasting", "spreadsheets"],
      },
      {
        slug: "reporting-lead",
        title: "Reporting Lead",
        description:
          "Great at turning finance work into concise summaries, tables, and executive-ready narratives.",
        focus: ["reporting", "summaries", "tables"],
        recommendedSkills: ["reporting", "presentation", "summaries"],
      },
      {
        slug: "controls-reviewer",
        title: "Controls Reviewer",
        description:
          "Great at checking approvals, separation of duties, audit trail quality, and unresolved control risks.",
        focus: ["controls", "audit trail", "approval risk"],
        recommendedSkills: ["controls", "audit", "risk review"],
      },
    ],
  },
].map(validateProfileBundle);

export function listProfileBundles(): readonly ProfileBundleDefinition[] {
  return PROFILE_BUNDLE_DEFINITIONS;
}

export function getProfileBundle(id: string): ProfileBundleDefinition | undefined {
  return PROFILE_BUNDLE_DEFINITIONS.find((bundle) => bundle.id === id);
}

export function defaultProfileBundle(): ProfileBundleDefinition {
  const bundle = PROFILE_BUNDLE_DEFINITIONS[0];
  if (!bundle) {
    throw new Error("At least one profile bundle must be defined");
  }
  return bundle;
}

export function sanitizeBundlePrefix(value: string, fallback: string): string {
  const cleaned = value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/^-+/, "")
    .replace(/-+$/, "");
  return cleaned || fallback;
}

export function bundleProfileName(
  prefix: string,
  role: ProfileBundleRole,
): string {
  return `${prefix}-${role.slug}`.slice(0, 64);
}

export function buildProfileBundlePlan(
  bundle: ProfileBundleDefinition,
  prefix: string,
  existingNames: ReadonlySet<string>,
): ProfileBundlePlanItem[] {
  return bundle.roles.map((role) => {
    const name = bundleProfileName(prefix, role);
    return { exists: existingNames.has(name), name, role };
  });
}

function validateProfileBundle(
  bundle: ProfileBundleDefinition,
): ProfileBundleDefinition {
  if (!bundle.id || !bundle.title || !bundle.defaultPrefix) {
    throw new Error("Profile bundle requires id, title, and defaultPrefix");
  }
  if (!bundle.roles.length) {
    throw new Error(`Profile bundle ${bundle.id} must define at least one role`);
  }
  const slugs = new Set<string>();
  for (const role of bundle.roles) {
    if (!role.slug || !role.title || !role.description) {
      throw new Error(`Profile bundle ${bundle.id} has an incomplete role`);
    }
    if (slugs.has(role.slug)) {
      throw new Error(`Profile bundle ${bundle.id} repeats role ${role.slug}`);
    }
    slugs.add(role.slug);
  }
  return bundle;
}
