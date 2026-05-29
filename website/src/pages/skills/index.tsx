import React, { useState, useMemo, useCallback, useRef, useEffect } from "react";
import Layout from "@theme/Layout";
import styles from "./styles.module.css";

interface Skill {
  name: string;
  description: string;
  overview?: string;
  category: string;
  categoryLabel: string;
  source: string;
  tags: string[];
  platforms: string[];
  author: string;
  version: string;
  license?: string;
  envVars?: string[];
  commands?: string[];
  docsPath?: string;
  identifier?: string;
  installCmd?: string;
  /** Lowercase pre-joined haystack used by the search filter.
   *  Built once at load time so per-keystroke filtering is a single
   *  `.includes()` per skill instead of array-join + toLowerCase on
   *  every render. Skipped on the wire — added in the loader. */
  _search?: string;
}

const allSkills: Skill[] = [];

interface IndexMeta {
  extractedAt?: string;
  indexGeneratedAt?: string;
  totalSkills?: number;
  externalSource?: string;
  bySource?: Record<string, number>;
}
const indexMeta: IndexMeta = {};

function formatRelativeTime(iso?: string): string | null {
  if (!iso) return null;
  const then = new Date(iso).getTime();
  if (!Number.isFinite(then)) return null;
  const now = Date.now();
  const diffMs = now - then;
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

const CATEGORY_ICONS: Record<string, string> = {
  apple: "\u{f179}",
  "autonomous-ai-agents": "\u{1F916}",
  blockchain: "\u{26D3}",
  communication: "\u{1F4AC}",
  creative: "\u{1F3A8}",
  "data-science": "\u{1F4CA}",
  devops: "\u{2699}",
  dogfood: "\u{1F436}",
  domain: "\u{1F310}",
  email: "\u{2709}",
  feeds: "\u{1F4E1}",
  gaming: "\u{1F3AE}",
  gifs: "\u{1F3AC}",
  github: "\u{1F4BB}",
  health: "\u{2764}",
  "inference-sh": "\u{26A1}",
  leisure: "\u{2615}",
  mcp: "\u{1F50C}",
  media: "\u{1F3B5}",
  migration: "\u{1F4E6}",
  mlops: "\u{1F9EA}",
  "note-taking": "\u{1F4DD}",
  productivity: "\u{2705}",
  "red-teaming": "\u{1F6E1}",
  research: "\u{1F50D}",
  security: "\u{1F512}",
  "smart-home": "\u{1F3E0}",
  "social-media": "\u{1F4F1}",
  "software-development": "\u{1F4BB}",
  translation: "\u{1F30D}",
  other: "\u{1F4E6}",
};

const SOURCE_CONFIG: Record<
  string,
  { label: string; color: string; bg: string; border: string; icon: string }
> = {
  "built-in": {
    label: "Built-in",
    color: "#4ade80",
    bg: "rgba(74, 222, 128, 0.08)",
    border: "rgba(74, 222, 128, 0.2)",
    icon: "\u{2713}",
  },
  optional: {
    label: "Optional",
    color: "#fbbf24",
    bg: "rgba(251, 191, 36, 0.08)",
    border: "rgba(251, 191, 36, 0.2)",
    icon: "\u{2B50}",
  },
  Anthropic: {
    label: "Anthropic",
    color: "#d4845a",
    bg: "rgba(212, 132, 90, 0.08)",
    border: "rgba(212, 132, 90, 0.2)",
    icon: "\u{25C6}",
  },
  LobeHub: {
    label: "LobeHub",
    color: "#60a5fa",
    bg: "rgba(96, 165, 250, 0.08)",
    border: "rgba(96, 165, 250, 0.2)",
    icon: "\u{25CB}",
  },
  "Claude Marketplace": {
    label: "Marketplace",
    color: "#a78bfa",
    bg: "rgba(167, 139, 250, 0.08)",
    border: "rgba(167, 139, 250, 0.2)",
    icon: "\u{25A0}",
  },
  "skills.sh": {
    label: "skills.sh",
    color: "#34d399",
    bg: "rgba(52, 211, 153, 0.08)",
    border: "rgba(52, 211, 153, 0.2)",
    icon: "\u{2734}",
  },
  ClawHub: {
    label: "ClawHub",
    color: "#f472b6",
    bg: "rgba(244, 114, 182, 0.08)",
    border: "rgba(244, 114, 182, 0.2)",
    icon: "\u{2726}",
  },
  "browse.sh": {
    label: "browse.sh",
    color: "#22d3ee",
    bg: "rgba(34, 211, 238, 0.08)",
    border: "rgba(34, 211, 238, 0.2)",
    icon: "\u{29BF}",
  },
  OpenAI: {
    label: "OpenAI",
    color: "#10b981",
    bg: "rgba(16, 185, 129, 0.08)",
    border: "rgba(16, 185, 129, 0.2)",
    icon: "\u{2737}",
  },
  HuggingFace: {
    label: "HuggingFace",
    color: "#fbbf24",
    bg: "rgba(251, 191, 36, 0.08)",
    border: "rgba(251, 191, 36, 0.2)",
    icon: "\u{1F917}",
  },
  NVIDIA: {
    label: "NVIDIA",
    color: "#76b900",
    bg: "rgba(118, 185, 0, 0.08)",
    border: "rgba(118, 185, 0, 0.25)",
    icon: "\u{25B6}",
  },
  VoltAgent: {
    label: "VoltAgent",
    color: "#facc15",
    bg: "rgba(250, 204, 21, 0.08)",
    border: "rgba(250, 204, 21, 0.2)",
    icon: "\u{26A1}",
  },
  GitHub: {
    label: "GitHub",
    color: "#94a3b8",
    bg: "rgba(148, 163, 184, 0.08)",
    border: "rgba(148, 163, 184, 0.2)",
    icon: "\u{2756}",
  },
  "Well-Known": {
    label: "Well-Known",
    color: "#818cf8",
    bg: "rgba(129, 140, 248, 0.08)",
    border: "rgba(129, 140, 248, 0.2)",
    icon: "\u{2756}",
  },
  gstack: {
    label: "gstack",
    color: "#fb923c",
    bg: "rgba(251, 146, 60, 0.08)",
    border: "rgba(251, 146, 60, 0.2)",
    icon: "\u{2756}",
  },
  MiniMax: {
    label: "MiniMax",
    color: "#f87171",
    bg: "rgba(248, 113, 113, 0.08)",
    border: "rgba(248, 113, 113, 0.2)",
    icon: "\u{2756}",
  },
};

const SOURCE_ORDER = [
  "all",
  "built-in",
  "optional",
  "Anthropic",
  "OpenAI",
  "HuggingFace",
  "NVIDIA",
  "skills.sh",
  "ClawHub",
  "browse.sh",
  "LobeHub",
  "Claude Marketplace",
  "VoltAgent",
  "Well-Known",
  "GitHub",
  "gstack",
  "MiniMax",
];

function highlightMatch(text: string, query: string): React.ReactNode {
  if (!query || !text) return text;
  const idx = text.toLowerCase().indexOf(query.toLowerCase());
  if (idx === -1) return text;
  return (
    <>
      {text.slice(0, idx)}
      <mark className={styles.highlight}>{text.slice(idx, idx + query.length)}</mark>
      {text.slice(idx + query.length)}
    </>
  );
}

function SkillCard({
  skill,
  query,
  expanded,
  onToggle,
  onCategoryClick,
  onTagClick,
  style,
}: {
  skill: Skill;
  query: string;
  expanded: boolean;
  onToggle: () => void;
  onCategoryClick: (cat: string) => void;
  onTagClick: (tag: string) => void;
  style?: React.CSSProperties;
}) {
  const src = SOURCE_CONFIG[skill.source] || SOURCE_CONFIG["optional"];
  const icon = CATEGORY_ICONS[skill.category] || "\u{1F4E6}";

  return (
    <div
      className={`${styles.card} ${expanded ? styles.cardExpanded : ""}`}
      onClick={onToggle}
      style={style}
    >
      <div className={styles.cardAccent} style={{ background: src.color }} />

      <div className={styles.cardInner}>
        <div className={styles.cardTop}>
          <span className={styles.cardIcon}>{icon}</span>
          <div className={styles.cardTitleGroup}>
            <h3 className={styles.cardTitle}>
              {highlightMatch(skill.name, query)}
            </h3>
            <span
              className={styles.sourcePill}
              style={{
                color: src.color,
                background: src.bg,
                borderColor: src.border,
              }}
            >
              {src.icon} {src.label}
            </span>
          </div>
        </div>

        <p className={`${styles.cardDesc} ${expanded ? styles.cardDescFull : ""}`}>
          {highlightMatch(skill.description || "No description available.", query)}
        </p>

        <div className={styles.cardMeta}>
          <button
            className={styles.catButton}
            onClick={(e) => {
              e.stopPropagation();
              onCategoryClick(skill.category);
            }}
            title={`Filter by ${skill.categoryLabel}`}
          >
            {skill.categoryLabel || skill.category}
          </button>
          {skill.platforms?.map((p) => (
            <span key={p} className={styles.platformPill}>
              {p === "macos" ? "\u{F8FF} macOS" : p === "linux" ? "\u{1F427} Linux" : p}
            </span>
          ))}
        </div>

        {expanded && (
          <div className={styles.cardDetail}>
            {skill.overview && (
              <div className={styles.overviewBlock}>
                <span className={styles.detailLabel}>Overview</span>
                <p className={styles.overviewText}>{skill.overview}</p>
              </div>
            )}
            {(skill.envVars?.length || skill.commands?.length) ? (
              <div className={styles.prereqBlock}>
                <span className={styles.detailLabel}>Prerequisites</span>
                {skill.envVars?.length ? (
                  <div className={styles.prereqRow}>
                    <span className={styles.prereqKind}>env</span>
                    <span className={styles.prereqList}>
                      {skill.envVars.map((v) => (
                        <code key={v} className={styles.prereqItem}>{v}</code>
                      ))}
                    </span>
                  </div>
                ) : null}
                {skill.commands?.length ? (
                  <div className={styles.prereqRow}>
                    <span className={styles.prereqKind}>cmd</span>
                    <span className={styles.prereqList}>
                      {skill.commands.map((c) => (
                        <code key={c} className={styles.prereqItem}>{c}</code>
                      ))}
                    </span>
                  </div>
                ) : null}
              </div>
            ) : null}
            {skill.tags?.length > 0 && (
              <div className={styles.tagRow}>
                {skill.tags.map((tag) => (
                  <button
                    key={tag}
                    className={styles.tagPill}
                    onClick={(e) => {
                      e.stopPropagation();
                      onTagClick(tag);
                    }}
                  >
                    {tag}
                  </button>
                ))}
              </div>
            )}
            {skill.author && (
              <div className={styles.authorRow}>
                <span className={styles.authorLabel}>Author</span>
                <span className={styles.authorValue}>{skill.author}</span>
              </div>
            )}
            {skill.version && (
              <div className={styles.authorRow}>
                <span className={styles.authorLabel}>Version</span>
                <span className={styles.authorValue}>{skill.version}</span>
              </div>
            )}
            {skill.license && (
              <div className={styles.authorRow}>
                <span className={styles.authorLabel}>License</span>
                <span className={styles.authorValue}>{skill.license}</span>
              </div>
            )}
            <div className={styles.installHint}>
              <code>{skill.installCmd || `hermes skills install ${skill.name}`}</code>
            </div>
            {skill.docsPath && (
              <a
                className={styles.docsLink}
                href={`/docs/user-guide/skills/${skill.docsPath}`}
                onClick={(e) => e.stopPropagation()}
              >
                View full documentation →
              </a>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function StatCard({ value, label, color }: { value: number; label: string; color: string }) {
  return (
    <div className={styles.stat}>
      <span className={styles.statValue} style={{ color }}>
        {value}
      </span>
      <span className={styles.statLabel}>{label}</span>
    </div>
  );
}

const PAGE_SIZE = 60;

// Routes Docusaurus serves the static API JSON from. `baseUrl` is `/docs/`,
// `static/api/` ends up at `/docs/api/`. Hardcoding here is fine because the
// same `baseUrl` is enforced repo-wide; if it ever changes, this is the only
// place that needs to follow.
const SKILLS_URL = "/docs/api/skills.json";
const META_URL = "/docs/api/skills-meta.json";

function buildSearchHaystack(s: Skill): string {
  // Pre-compute the lowercase blob the search filter scans. Done once at
  // load time instead of per-keystroke per-skill. With 50k+ skills the
  // per-keystroke variant was unusably slow.
  return [
    s.name,
    s.description,
    s.overview,
    s.categoryLabel,
    s.author,
    ...(s.tags || []),
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

export default function SkillsDashboard() {
  // Lazy-loaded data. Was bundled into the JS chunk (~22 MB at 50k skills,
  // which made the initial page load unusable on mobile). Now fetched on
  // mount from the same CDN that serves the docs.
  const [data, setData] = useState<{ skills: Skill[]; meta: IndexMeta } | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [search, setSearch] = useState("");
  // Debounced copy of `search` — used by the filter. Without the debounce,
  // typing into the search box ran .filter() over the whole catalog on
  // every keystroke, which on a 50k-item list felt like the page had
  // hung. 150ms gives a snappy feel without lagging behind the user.
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [sourceFilter, setSourceFilter] = useState("all");
  const [categoryFilter, setCategoryFilter] = useState("all");
  const [expandedCard, setExpandedCard] = useState<string | null>(null);
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const searchRef = useRef<HTMLInputElement>(null);
  const gridRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [sk, mt] = await Promise.all([
          fetch(SKILLS_URL).then((r) => {
            if (!r.ok) throw new Error(`skills.json HTTP ${r.status}`);
            return r.json();
          }),
          fetch(META_URL).then((r) => (r.ok ? r.json() : {})).catch(() => ({})),
        ]);
        if (cancelled) return;
        const skillsArr = Array.isArray(sk) ? (sk as Skill[]) : [];
        // Stamp the precomputed search haystack onto each row.
        for (const s of skillsArr) s._search = buildSearchHaystack(s);
        setData({ skills: skillsArr, meta: mt || {} });
      } catch (err) {
        if (cancelled) return;
        setLoadError(err instanceof Error ? err.message : String(err));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Debounce the search input — 150ms feels instant while preventing the
  // filter from running on every individual keystroke.
  useEffect(() => {
    const t = setTimeout(() => setDebouncedSearch(search), 150);
    return () => clearTimeout(t);
  }, [search]);

  const allSkillsLocal: Skill[] = data?.skills ?? [];
  const indexMetaLocal: IndexMeta = data?.meta ?? indexMeta;

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "/" && document.activeElement?.tagName !== "INPUT") {
        e.preventDefault();
        searchRef.current?.focus();
      }
      if (e.key === "Escape") {
        searchRef.current?.blur();
        setExpandedCard(null);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  const sources = useMemo(() => {
    const set = new Set(allSkillsLocal.map((s) => s.source));
    return SOURCE_ORDER.filter((s) => s === "all" || set.has(s));
  }, [allSkillsLocal]);

  const categoryEntries = useMemo(() => {
    const pool =
      sourceFilter === "all"
        ? allSkillsLocal
        : allSkillsLocal.filter((s) => s.source === sourceFilter);
    const map = new Map<string, { label: string; count: number }>();
    for (const s of pool) {
      const key = s.category || "uncategorized";
      const existing = map.get(key);
      if (existing) {
        existing.count++;
      } else {
        map.set(key, {
          label: s.categoryLabel || s.category || "Uncategorized",
          count: 1,
        });
      }
    }
    return Array.from(map.entries())
      .sort((a, b) => b[1].count - a[1].count)
      .map(([key, { label, count }]) => ({ key, label, count }));
  }, [sourceFilter, allSkillsLocal]);

  const filtered = useMemo(() => {
    const q = debouncedSearch.toLowerCase().trim();
    return allSkillsLocal.filter((s) => {
      if (sourceFilter !== "all" && s.source !== sourceFilter) return false;
      if (categoryFilter !== "all" && s.category !== categoryFilter) return false;
      if (q) {
        // _search is pre-built in the load effect — single .includes() per row.
        return (s._search || "").includes(q);
      }
      return true;
    });
  }, [debouncedSearch, sourceFilter, categoryFilter, allSkillsLocal]);

  useEffect(() => {
    setVisibleCount(PAGE_SIZE);
    setExpandedCard(null);
  }, [debouncedSearch, sourceFilter, categoryFilter]);

  const visible = filtered.slice(0, visibleCount);
  const hasMore = visibleCount < filtered.length;

  const handleSourceChange = useCallback(
    (src: string) => {
      setSourceFilter(src);
      setCategoryFilter("all");
    },
    []
  );

  const handleCategoryClick = useCallback((cat: string) => {
    setCategoryFilter(cat);
    gridRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    setSidebarOpen(false);
  }, []);

  const handleTagClick = useCallback((tag: string) => {
    setSearch(tag);
    searchRef.current?.focus();
  }, []);

  const clearAll = useCallback(() => {
    setSearch("");
    setSourceFilter("all");
    setCategoryFilter("all");
  }, []);

  return (
    <Layout
      title="Skills Hub"
      description="Browse all skills and plugins available for Hermes Agent"
    >
      <div className={styles.page}>
        <header className={styles.hero}>
          <div className={styles.heroGlow} />
          <div className={styles.heroContent}>
            <p className={styles.heroEyebrow}>Hermes Agent</p>
            <h1 className={styles.heroTitle}>Skills Hub</h1>
            <p className={styles.heroSub}>
              Discover, search, and install from{" "}
              <strong className={styles.heroAccent}>
                {data ? allSkillsLocal.length.toLocaleString() : "…"}
              </strong>{" "}
              skills across {sources.length - 1} registries
              {loadError && (
                <span style={{ color: "#f87171", marginLeft: 8 }}>
                  · failed to load catalog ({loadError})
                </span>
              )}
            </p>
            {(indexMetaLocal?.indexGeneratedAt || indexMetaLocal?.extractedAt) && (
              <p className={styles.heroSub} style={{ fontSize: "0.85rem", opacity: 0.75 }}>
                Catalog refreshed{" "}
                <span title={indexMetaLocal.indexGeneratedAt || indexMetaLocal.extractedAt}>
                  {formatRelativeTime(
                    indexMetaLocal.indexGeneratedAt || indexMetaLocal.extractedAt,
                  ) || "recently"}
                </span>
                {" "}· auto-rebuilt twice daily
              </p>
            )}

            <div className={styles.statsRow}>
              <StatCard
                value={allSkillsLocal.filter((s) => s.source === "built-in").length}
                label="Built-in"
                color="#4ade80"
              />
              <StatCard
                value={allSkillsLocal.filter((s) => s.source === "optional").length}
                label="Optional"
                color="#fbbf24"
              />
              <StatCard
                value={
                  allSkillsLocal.filter(
                    (s) => s.source !== "built-in" && s.source !== "optional"
                  ).length
                }
                label="Community"
                color="#60a5fa"
              />
              <StatCard
                value={new Set(allSkillsLocal.map((s) => s.category)).size}
                label="Categories"
                color="#a78bfa"
              />
            </div>
          </div>
        </header>

        <div className={styles.controlsBar}>
          <div className={styles.searchWrap}>
            <svg className={styles.searchIcon} viewBox="0 0 20 20" fill="currentColor" width="18" height="18">
              <path
                fillRule="evenodd"
                d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z"
                clipRule="evenodd"
              />
            </svg>
            <input
              ref={searchRef}
              type="text"
              placeholder='Search skills... (press "/" to focus)'
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className={styles.searchInput}
            />
            {search && (
              <button className={styles.clearBtn} onClick={() => setSearch("")}>
                <svg viewBox="0 0 20 20" fill="currentColor" width="16" height="16">
                  <path
                    fillRule="evenodd"
                    d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                    clipRule="evenodd"
                  />
                </svg>
              </button>
            )}
          </div>

          <div className={styles.sourcePills}>
            {sources.map((src) => {
              const active = sourceFilter === src;
              const conf = SOURCE_CONFIG[src];
              const count =
                src === "all"
                  ? allSkillsLocal.length
                  : allSkillsLocal.filter((s) => s.source === src).length;
              return (
                <button
                  key={src}
                  className={`${styles.srcPill} ${active ? styles.srcPillActive : ""}`}
                  onClick={() => handleSourceChange(src)}
                  style={
                    active && conf
                      ? ({
                          "--pill-color": conf.color,
                          "--pill-bg": conf.bg,
                          "--pill-border": conf.border,
                        } as React.CSSProperties)
                      : undefined
                  }
                >
                  {src === "all" ? "All" : conf?.label || src}
                  <span className={styles.srcCount}>{count}</span>
                </button>
              );
            })}
          </div>
        </div>

        <div className={styles.layout}>
          <button
            className={styles.sidebarToggle}
            onClick={() => setSidebarOpen(!sidebarOpen)}
          >
            <svg viewBox="0 0 20 20" fill="currentColor" width="18" height="18">
              <path
                fillRule="evenodd"
                d="M3 5a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM3 10a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM3 15a1 1 0 011-1h6a1 1 0 110 2H4a1 1 0 01-1-1z"
                clipRule="evenodd"
              />
            </svg>
            Categories
            {categoryFilter !== "all" && (
              <span className={styles.activeCatBadge}>
                {categoryEntries.find((c) => c.key === categoryFilter)?.label}
              </span>
            )}
          </button>

          <aside className={`${styles.sidebar} ${sidebarOpen ? styles.sidebarOpen : ""}`}>
            <div className={styles.sidebarHeader}>
              <h2 className={styles.sidebarTitle}>Categories</h2>
              {categoryFilter !== "all" && (
                <button className={styles.sidebarClear} onClick={() => setCategoryFilter("all")}>
                  Clear
                </button>
              )}
            </div>
            <nav className={styles.catList}>
              <button
                className={`${styles.catItem} ${categoryFilter === "all" ? styles.catItemActive : ""}`}
                onClick={() => {
                  setCategoryFilter("all");
                  setSidebarOpen(false);
                }}
              >
                <span className={styles.catItemIcon}>{"\u{1F4CB}"}</span>
                <span className={styles.catItemLabel}>All Skills</span>
                <span className={styles.catItemCount}>{filtered.length}</span>
              </button>
              {categoryEntries.map((cat) => (
                <button
                  key={cat.key}
                  className={`${styles.catItem} ${categoryFilter === cat.key ? styles.catItemActive : ""}`}
                  onClick={() => handleCategoryClick(cat.key)}
                >
                  <span className={styles.catItemIcon}>
                    {CATEGORY_ICONS[cat.key] || "\u{1F4E6}"}
                  </span>
                  <span className={styles.catItemLabel}>{cat.label}</span>
                  <span className={styles.catItemCount}>{cat.count}</span>
                </button>
              ))}
            </nav>
          </aside>

          <main className={styles.main} ref={gridRef}>
            {(search || sourceFilter !== "all" || categoryFilter !== "all") && (
              <div className={styles.filterSummary}>
                <span className={styles.filterCount}>
                  {filtered.length} result{filtered.length !== 1 ? "s" : ""}
                </span>
                {search && (
                  <span className={styles.filterChip}>
                    &ldquo;{search}&rdquo;
                    <button onClick={() => setSearch("")}>&times;</button>
                  </span>
                )}
                {sourceFilter !== "all" && (
                  <span className={styles.filterChip}>
                    {SOURCE_CONFIG[sourceFilter]?.label || sourceFilter}
                    <button onClick={() => setSourceFilter("all")}>&times;</button>
                  </span>
                )}
                {categoryFilter !== "all" && (
                  <span className={styles.filterChip}>
                    {categoryEntries.find((c) => c.key === categoryFilter)?.label ||
                      categoryFilter}
                    <button onClick={() => setCategoryFilter("all")}>&times;</button>
                  </span>
                )}
                <button className={styles.clearAllBtn} onClick={clearAll}>
                  Clear all
                </button>
              </div>
            )}

            {visible.length > 0 ? (
              <>
                <div className={styles.grid}>
                  {visible.map((skill, i) => {
                    const key = `${skill.source}-${skill.name}-${i}`;
                    return (
                      <SkillCard
                        key={key}
                        skill={skill}
                        query={search}
                        expanded={expandedCard === key}
                        onToggle={() =>
                          setExpandedCard(expandedCard === key ? null : key)
                        }
                        onCategoryClick={handleCategoryClick}
                        onTagClick={handleTagClick}
                        style={{ animationDelay: `${Math.min(i, 20) * 25}ms` }}
                      />
                    );
                  })}
                </div>
                {hasMore && (
                  <div className={styles.loadMoreWrap}>
                    <button
                      className={styles.loadMoreBtn}
                      onClick={() => setVisibleCount((v) => v + PAGE_SIZE)}
                    >
                      Show more ({filtered.length - visibleCount} remaining)
                    </button>
                  </div>
                )}
              </>
            ) : (
              <div className={styles.empty}>
                <div className={styles.emptyIcon}>{"\u{1F50D}"}</div>
                <h3 className={styles.emptyTitle}>No skills found</h3>
                <p className={styles.emptyDesc}>
                  Try a different search term or clear your filters.
                </p>
                <button className={styles.emptyReset} onClick={clearAll}>
                  Reset all filters
                </button>
              </div>
            )}
          </main>
        </div>
      </div>

      {sidebarOpen && (
        <div className={styles.backdrop} onClick={() => setSidebarOpen(false)} />
      )}
    </Layout>
  );
}
