export interface BotProfile {
  suggested_name: string;
  description: string;
  soul: string;
  starter_prompt: string;
}

export interface BotCapabilities {
  skills: string[];
  toolsets: string[];
}

export interface BotSetupRequirement {
  kind: "toolset" | "command" | "connector" | "plugin";
  id: string;
  required: boolean;
  purpose: string;
}

export interface BotRoutine {
  id: string;
  name: string;
  prompt: string;
  schedule: string;
}

export interface BotWorkflowSample {
  kind: "input" | "template";
  title: string;
  path: string;
  preview: string;
  truncated: boolean;
}

export interface BotWorkflowPackage {
  skill: string;
  exampleTasks: string[];
  samples: BotWorkflowSample[];
}

export interface BotPresentation {
  emoji: string;
  color: string;
}

export interface CatalogBot {
  name: string;
  version: string;
  maintainer: string;
  maintainerSlug: string;
  tier: "official" | "community" | string;
  category: string;
  tags: string[];
  title: string;
  summary: string;
  profile: BotProfile;
  capabilities: BotCapabilities;
  setup: { requirements: BotSetupRequirement[] };
  routines: BotRoutine[];
  workflowPackage?: BotWorkflowPackage;
  starterExample: string;
  presentation: BotPresentation;
  authorPath: string;
}

export interface BotCatalogMeta {
  generatedAt?: string;
  total?: number;
  byTier?: Record<string, number>;
  byCategory?: Record<string, number>;
  removedCount?: number;
}

const CATEGORY_LABELS: Record<string, string> = {
  research: "Research & analysis",
  productivity: "Productivity",
  product: "Product & strategy",
  sales: "Sales",
  design: "Design",
  recruiting: "Recruiting & people",
  finance: "Personal finance",
  meetings: "Meetings & follow-through",
  email: "Email & communication",
  development: "Software development",
  automation: "Automation",
  creative: "Creative work",
  general: "General",
};

const CATEGORY_BLURBS: Record<string, string> = {
  research: "Find current evidence, compare sources, and turn it into decisions.",
  productivity: "Move recurring work from a blank page to a finished outcome.",
  product: "Track markets, competitors, and product decisions with evidence.",
  sales: "Turn real conversations into stronger discovery, positioning, and next steps.",
  design: "Find the usability, visual, copy, and accessibility fixes that matter most.",
  recruiting: "Keep interview logistics, preparation, and follow-through moving safely.",
  finance: "Make better use of the products and benefits you already have.",
  meetings: "Convert notes into decisions, owners, and polished follow-through.",
  email: "Triage communication and surface what needs your attention.",
  development: "Plan, build, review, and maintain software with a focused teammate.",
  automation: "Own repeatable workflows while keeping you in control.",
  creative: "Develop ideas into clear, polished work.",
  general: "Purpose-built Hermes profiles for focused work.",
};

export function categoryLabel(category: string): string {
  return CATEGORY_LABELS[category] || category.replace(/[-_]/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export function categoryBlurb(category: string): string {
  return CATEGORY_BLURBS[category] || CATEGORY_BLURBS.general;
}

export function botPagePath(name: string): string {
  return `/bots/${encodeURIComponent(name)}`;
}

export function botAuthorPath(slug: string): string {
  return `/bots/by/${encodeURIComponent(slug)}`;
}

export function botInstallLink(name: string): string {
  return `hermes://bot/install?catalog=${encodeURIComponent(name)}`;
}

export function botSearchText(bot: CatalogBot): string {
  return [
    bot.title,
    bot.name,
    bot.summary,
    bot.profile?.description,
    bot.maintainer,
    bot.category,
    ...(bot.tags || []),
    ...(bot.capabilities?.skills || []),
    ...(bot.capabilities?.toolsets || []),
    ...(bot.setup?.requirements || []).flatMap((requirement) => [requirement.id, requirement.kind, requirement.purpose]),
    ...(bot.routines || []).flatMap((routine) => [routine.name, routine.prompt, routine.schedule]),
    ...(bot.workflowPackage?.exampleTasks || []),
  ].filter(Boolean).join(" ").toLowerCase();
}
