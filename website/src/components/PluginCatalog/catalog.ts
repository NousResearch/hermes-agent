// Shared catalog vocabulary for the Plugins grid (src/pages/plugins), the per-plugin pages and the
// author pages (plugins/plugin-catalog-pages generates their routes). One definition of the entry
// shape, the tier/category taxonomy and the link builders keeps the three surfaces from drifting.

import { pluginCatalogInstallUrl } from "../../../../apps/shared/src/catalog-install";

export interface PluginCapabilities {
  providesTools?: string[];
  providesHooks?: string[];
  providesMiddleware?: string[];
  requiresEnv?: string[];
}

export interface CatalogPlugin {
  name: string;
  description: string;
  repo: string;
  sha: string;
  shaShort: string;
  tier: string;
  category: string;
  maintainer: string;
  /** URL segment of the author page (/plugins/by/<slug>); one per maintainer. */
  maintainerSlug?: string;
  subdir?: string;
  requiresHermes?: string;
  platforms?: string[];
  capabilities?: PluginCapabilities;
  docsUrl?: string;
  /** Human label for the pin ("1.4.0"); cosmetic, shown beside the sha. */
  version?: string;
  /** Card banner image (GitHub-hosted https URL enforced by the extractor). */
  image?: string;
  /** Gallery on the plugin page; GitHub-hosted https URLs, at most 6. */
  screenshots?: string[];
  /** The plugin page renders the README from the pinned commit when true. */
  readme?: boolean;
  readmeUrl?: string;
  installCommand: string;
  /** GitHub stargazers at the last daily probe; null when the repo is not on GitHub or unprobed. */
  stars?: number | null;
  /** ISO dates from git history (first listing / last re-pin); present once the dates extractor runs. */
  addedAt?: string | null;
  updatedAt?: string | null;
  /** Lowercase pre-joined haystack for the search filter (built at load). */
  _search?: string;
}

export interface CatalogMeta {
  generatedAt?: string;
  total?: number;
  byTier?: Record<string, number>;
  byCategory?: Record<string, number>;
  removedCount?: number;
  starsFetchedAt?: string | null;
}

// Docs section describing the PR-based submission workflow.
export const SUBMIT_PLUGIN_URL = "/user-guide/features/plugin-catalog#submitting-a-plugin-to-the-catalog";

/** Deep link into the Desktop app's Install Plugin dialog, catalog mode: the app
 *  resolves the reviewed pin itself, so the page never hands it a repo URL. */
export function desktopInstallLink(name: string): string {
  return pluginCatalogInstallUrl({ name });
}

/** Site route of an entry's page (Docusaurus prefixes baseUrl/locale via <Link>). */
export function pluginPagePath(name: string): string {
  return `/plugins/${encodeURIComponent(name)}`;
}

/** Site route of a maintainer's page. */
export function authorPagePath(slug: string): string {
  return `/plugins/by/${encodeURIComponent(slug)}`;
}

export function repoUrl(plugin: Pick<CatalogPlugin, "repo">): string {
  return plugin.repo.replace(/\.git$/, "").replace(/\/$/, "");
}

/** Browse link to the exact reviewed tree (GitHub `tree/<sha>`, GitLab `-/tree/<sha>`). */
export function pinUrl(plugin: Pick<CatalogPlugin, "repo" | "sha" | "subdir">): string {
  const base = repoUrl(plugin);
  const tree = /^https:\/\/gitlab\.com\//.test(base) ? `${base}/-/tree/${plugin.sha}` : `${base}/tree/${plugin.sha}`;
  return plugin.subdir ? `${tree}/${plugin.subdir.replace(/^\/+|\/+$/g, "")}` : tree;
}

export const TIER_CONFIG: Record<
  string,
  { label: string; color: string; bg: string; border: string; icon: string }
> = {
  official: {
    label: "Official",
    color: "#ffd700",
    bg: "rgba(255, 215, 0, 0.08)",
    border: "rgba(255, 215, 0, 0.25)",
    icon: "\u{2713}",
  },
  community: {
    label: "Community",
    color: "#94a3b8",
    bg: "rgba(148, 163, 184, 0.08)",
    border: "rgba(148, 163, 184, 0.2)",
    icon: "\u{2756}",
  },
};

// Browse taxonomy. Order here is the order of the filter pills and of the
// grouped sections; keep it in sync with CATALOG_CATEGORIES in
// hermes_cli/plugin_catalog.py and website/scripts/extract-plugins.py.
export const CATEGORY_CONFIG: Record<string, { label: string; icon: string; blurb: string }> = {
  desktop: { label: "Desktop", icon: "\u{1F5A5}\u{FE0F}", blurb: "Panes, tabs and views for Hermes Desktop" },
  memory: { label: "Memory", icon: "\u{1F9E0}", blurb: "Memory providers and context engines" },
  platform: { label: "Platforms", icon: "\u{1F4AC}", blurb: "Messaging and channel adapters" },
  web: { label: "Web & Browser", icon: "\u{1F310}", blurb: "Search backends, extraction and browser control" },
  tools: { label: "Tools", icon: "\u{1F6E0}\u{FE0F}", blurb: "New tools the agent can call" },
  voice: { label: "Voice", icon: "\u{1F399}\u{FE0F}", blurb: "Speech, TTS and realtime audio" },
  automation: { label: "Automation", icon: "\u{23F1}\u{FE0F}", blurb: "Hooks, wake triggers and session automation" },
  models: { label: "Models", icon: "\u{2728}", blurb: "Model and inference providers" },
  general: { label: "General", icon: "\u{1F4E6}", blurb: "Plugins that span several areas" },
};
export const CATEGORY_ORDER = Object.keys(CATEGORY_CONFIG);

export function categoryOf(plugin: Pick<CatalogPlugin, "category">) {
  return CATEGORY_CONFIG[plugin.category] || CATEGORY_CONFIG.general;
}

export function tierOf(plugin: Pick<CatalogPlugin, "tier">) {
  return TIER_CONFIG[plugin.tier] || TIER_CONFIG.community;
}

export function formatStars(n: number): string {
  return n >= 1000 ? `${(n / 1000).toFixed(n >= 10_000 ? 0 : 1)}k` : String(n);
}

export function formatRelativeTime(iso?: string | null): string | null {
  if (!iso) return null;
  const then = new Date(iso).getTime();
  if (!Number.isFinite(then)) return null;
  const diffMs = Date.now() - then;
  if (diffMs < 0) return "just now";
  const mins = Math.floor(diffMs / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins} minute${mins === 1 ? "" : "s"} ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours} hour${hours === 1 ? "" : "s"} ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days} day${days === 1 ? "" : "s"} ago`;
  const months = Math.floor(days / 30);
  return `${months} month${months === 1 ? "" : "s"} ago`;
}

/** "Sep 20, 2026" for an ISO date; null when unparseable. */
export function formatDate(iso?: string | null): string | null {
  if (!iso) return null;
  const d = new Date(iso);
  if (!Number.isFinite(d.getTime())) return null;
  return d.toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric", timeZone: "UTC" });
}

/** Split a description into prose and its "Disclosure — …" sentences (catalog convention for
 *  behaviour a user opts into: auto-payments, vendor client identity, etc.). */
export function splitDisclosure(description: string): { prose: string; disclosures: string[] } {
  const marker = /\bDisclosure\s*[—–-]\s*/g;
  const idx = description.search(marker);
  if (idx === -1) return { prose: description, disclosures: [] };
  const prose = description.slice(0, idx).trim();
  const disclosures = description
    .slice(idx)
    .split(marker)
    .map((s) => s.trim())
    .filter(Boolean);
  return { prose, disclosures };
}
