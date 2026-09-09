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
  /** Clickable URL to the skill's origin (repo / detail page). Synthesized
   *  in extract-skills.py for community skills that have no generated docs
   *  page, so the expanded card always has somewhere to send the user. */
  sourceUrl?: string;
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

const ZH_HANT_CHROME: Record<string, string> = {
  "Docs": "文件",
  "Skills": "技能",
  "Download": "下載",
  "English": "繁體中文",
  "Home": "首頁",
  "GitHub": "GitHub",
  "Discord": "Discord",
  "Search": "搜尋",
  "HERMES AGENT": "HERMES AGENT",
  "Hermes Agent": "HERMES AGENT",
  "Skills Hub": "技能中心",
  "Discover, search, and install from": "探索、搜尋並安裝來自",
  "Catalog refreshed": "目錄更新於",
  "Built-in": "內建",
  "Optional": "選用",
  "Community": "社群",
  "Categories": "分類",
  "Copy install command": "複製安裝指令",
  "Copied": "已複製",
  "Copy": "複製",
  "Overview": "概覽",
  "Prerequisites": "必要條件",
  "Author": "作者",
  "Version": "版本",
  "License": "授權",
  "View full documentation": "查看完整文件",
  "View source": "查看來源",
  "+ Add to this Agent": "+ 加入此 Agent",
  "All": "全部",
  "All Skills": "所有技能",
  "Clear": "清除",
  "Clear all": "全部清除",
  "Loading the catalog…": "正在載入目錄…",
  "No skills found": "找不到技能",
  "Reset all filters": "重設所有篩選條件",
  "Search skills... (press \"/\" to focus)": "搜尋技能…（按「/」聚焦）",
  "Fetching 88k+ skills across every registry. One moment.": "正在從各個登錄來源取得 88,000 多項技能，請稍候。",
  "Try a different search term or clear your filters.": "請改用其他關鍵字，或清除篩選條件。",
};

function uiText(text: string, isZhHant: boolean): string {
  return isZhHant ? ZH_HANT_CHROME[text] || text : text;
}

function formatRelativeTime(iso?: string, isZhHant = false): string | null {
  if (!iso) return null;
  const then = new Date(iso).getTime();
  if (!Number.isFinite(then)) return null;
  const now = Date.now();
  const diffMs = now - then;
  if (diffMs < 0) return isZhHant ? "剛剛" : "just now";
  const mins = Math.floor(diffMs / 60_000);
  if (mins < 1) return isZhHant ? "剛剛" : "just now";
  if (mins < 60) return isZhHant ? `${mins} 分鐘前` : `${mins} minute${mins === 1 ? "" : "s"} ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return isZhHant ? `${hours} 小時前` : `${hours} hour${hours === 1 ? "" : "s"} ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return isZhHant ? `${days} 天前` : `${days} day${days === 1 ? "" : "s"} ago`;
  const months = Math.floor(days / 30);
  return isZhHant ? `${months} 個月前` : `${months} month${months === 1 ? "" : "s"} ago`;
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

function CopyButton({ text, isZhHant }: { text: string; isZhHant: boolean }) {
  const [copied, setCopied] = useState(false);
  const onCopy = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      navigator.clipboard?.writeText(text).then(
        () => {
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        },
        () => {},
      );
    },
    [text],
  );
  return (
    <button
      className={styles.copyBtn}
      onClick={onCopy}
      title={uiText("Copy install command", isZhHant)}
      aria-label={uiText("Copy install command", isZhHant)}
    >
      {copied ? (
        <svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14">
          <path
            fillRule="evenodd"
            d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z"
            clipRule="evenodd"
          />
        </svg>
      ) : (
        <svg viewBox="0 0 20 20" fill="currentColor" width="14" height="14">
          <path d="M7 3.5A1.5 1.5 0 018.5 2h3.879a1.5 1.5 0 011.06.44l3.122 3.12A1.5 1.5 0 0117 6.622V12.5a1.5 1.5 0 01-1.5 1.5h-1v-3.379a3 3 0 00-.879-2.121L10.5 5.379A3 3 0 008.379 4.5H7v-1z" />
          <path d="M4.5 6A1.5 1.5 0 003 7.5v9A1.5 1.5 0 004.5 18h7a1.5 1.5 0 001.5-1.5v-5.879a1.5 1.5 0 00-.44-1.06L9.44 6.439A1.5 1.5 0 008.378 6H4.5z" />
        </svg>
      )}
      <span className={styles.copyBtnLabel}>
        {uiText(copied ? "Copied" : "Copy", isZhHant)}
      </span>
    </button>
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
  onPick,
  isZhHant,
}: {
  skill: Skill;
  query: string;
  expanded: boolean;
  onToggle: () => void;
  onCategoryClick: (cat: string) => void;
  onTagClick: (tag: string) => void;
  style?: React.CSSProperties;
  /** Picker embed mode: render "+ Add to this Agent" and call this. */
  onPick?: (skill: Skill) => void;
  isZhHant: boolean;
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
              {src.icon} {uiText(src.label, isZhHant)}
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
                <span className={styles.detailLabel}>{uiText("Overview", isZhHant)}</span>
                <p className={styles.overviewText}>{skill.overview}</p>
              </div>
            )}
            {(skill.envVars?.length || skill.commands?.length) ? (
              <div className={styles.prereqBlock}>
                <span className={styles.detailLabel}>{uiText("Prerequisites", isZhHant)}</span>
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
                <span className={styles.authorLabel}>{uiText("Author", isZhHant)}</span>
                <span className={styles.authorValue}>{skill.author}</span>
              </div>
            )}
            {skill.version && (
              <div className={styles.authorRow}>
                <span className={styles.authorLabel}>{uiText("Version", isZhHant)}</span>
                <span className={styles.authorValue}>{skill.version}</span>
              </div>
            )}
            {skill.license && (
              <div className={styles.authorRow}>
                <span className={styles.authorLabel}>{uiText("License", isZhHant)}</span>
                <span className={styles.authorValue}>{skill.license}</span>
              </div>
            )}
            <div className={styles.installHint}>
              <code>{skill.installCmd || `hermes skills install ${skill.name}`}</code>
              <CopyButton
                text={skill.installCmd || `hermes skills install ${skill.name}`}
                isZhHant={isZhHant}
              />
            </div>
            {onPick ? (
              <button
                className={styles.pickBtn}
                onClick={(e) => {
                  e.stopPropagation();
                  onPick(skill);
                }}
              >
                {uiText("+ Add to this Agent", isZhHant)}
              </button>
            ) : null}
            <div className={styles.cardLinks}>
              {skill.docsPath ? (
                <a
                  className={styles.docsLink}
                  href={`/docs/user-guide/skills/${skill.docsPath}`}
                  onClick={(e) => e.stopPropagation()}
                >
                  {uiText("View full documentation", isZhHant)} →
                </a>
              ) : skill.sourceUrl ? (
                <a
                  className={styles.docsLink}
                  href={skill.sourceUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={(e) => e.stopPropagation()}
                >
                  {uiText("View source", isZhHant)} ↗
                </a>
              ) : null}
            </div>
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
const ZH_HANT_DESCRIPTIONS_URL = "/docs/api/skills.zh-hant.json";

function localizeSiteChrome() {
  const navbar = document.querySelector(".navbar");
  if (!navbar) return;

  const walker = document.createTreeWalker(navbar, NodeFilter.SHOW_TEXT);
  let node = walker.nextNode();
  while (node) {
    const original = node.textContent || "";
    const trimmed = original.trim();
    const translated = ZH_HANT_CHROME[trimmed];
    if (translated && translated !== trimmed) {
      node.textContent = original.replace(trimmed, translated);
    }
    node = walker.nextNode();
  }

  navbar.querySelectorAll<HTMLElement>("[aria-label], [title], [placeholder]").forEach((el) => {
    for (const attr of ["aria-label", "title", "placeholder"]) {
      const value = el.getAttribute(attr);
      if (!value) continue;
      if (ZH_HANT_CHROME[value]) {
        el.setAttribute(attr, ZH_HANT_CHROME[value]);
      } else if (value.startsWith("Search")) {
        el.setAttribute(attr, value.replace("Search", ZH_HANT_CHROME["Search"]));
      }
    }
  });
}

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
  // Picker embed mode (?embed=picker): the page is being iframed by a host
  // app (Hermes desktop's Bot Mode agent editor) as a skill PICKER. Site
  // chrome is hidden via a CSS class and every card gains an
  // "+ Add to this Agent" button that posts
  //   { type: 'hermes-skill-pick', name, identifier, installCmd, source }
  // to the parent window. The HOST performs the actual install through its
  // own gateway (skills.manage) — the page never installs anything, so
  // there is no origin to trust in this direction; parents must validate
  // event.origin themselves before acting on the message.
  const pickerMode =
    typeof window !== "undefined" &&
    new URLSearchParams(window.location.search).get("embed") === "picker";
  const isZhHant =
    typeof window !== "undefined" &&
    new URLSearchParams(window.location.search).get("lang") === "zh-hant";

  useEffect(() => {
    if (!isZhHant) return;
    document.documentElement.lang = "zh-Hant";
    localizeSiteChrome();
    const observer = new MutationObserver(localizeSiteChrome);
    observer.observe(document.body, { childList: true, subtree: true });
    return () => observer.disconnect();
  }, [isZhHant]);

  const pickSkill = useCallback(
    (skill: Skill) => {
      if (typeof window === "undefined" || window.parent === window) return;
      window.parent.postMessage(
        {
          type: "hermes-skill-pick",
          name: skill.name,
          identifier: skill.identifier || skill.name,
          installCmd: skill.installCmd || `hermes skills install ${skill.name}`,
          source: skill.source,
        },
        "*"
      );
    },
    []
  );

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
        const zhHantDescriptionsPromise: Promise<Record<string, string>> = isZhHant
          ? fetch(ZH_HANT_DESCRIPTIONS_URL)
              .then((r) => (r.ok ? r.json() : {}))
              .catch(() => ({}))
          : Promise.resolve({});
        const [sk, mt, ZH_HANT_DESCRIPTIONS] = await Promise.all([
          fetch(SKILLS_URL).then((r) => {
            if (!r.ok) throw new Error(`skills.json HTTP ${r.status}`);
            return r.json();
          }),
          fetch(META_URL).then((r) => (r.ok ? r.json() : {})).catch(() => ({})),
          zhHantDescriptionsPromise,
        ]);
        if (cancelled) return;
        const skillsArr = Array.isArray(sk) ? (sk as Skill[]) : [];
        // Localize only official built-ins. Every absent translation and every
        // third-party row keeps the description from the English catalog.
        for (const skill of skillsArr) {
          if (isZhHant && skill.source === "built-in") {
            skill.description = ZH_HANT_DESCRIPTIONS[skill.name] || skill.description;
          }
          skill._search = buildSearchHaystack(skill);
        }
        setData({ skills: skillsArr, meta: mt || {} });
      } catch (err) {
        if (cancelled) return;
        setLoadError(err instanceof Error ? err.message : String(err));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [isZhHant]);

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
      title={uiText("Skills Hub", isZhHant)}
      description={
        isZhHant
          ? "瀏覽 Hermes Agent 可用的所有技能與外掛"
          : "Browse all skills and plugins available for Hermes Agent"
      }
    >
      <div className={`${styles.page} ${pickerMode ? styles.pickerMode : ""}`}>
        <header className={styles.hero}>
          <div className={styles.heroGlow} />
          <div className={styles.heroContent}>
            <p className={styles.heroEyebrow}>{uiText("Hermes Agent", isZhHant)}</p>
            <h1 className={styles.heroTitle}>{uiText("Skills Hub", isZhHant)}</h1>
            <p className={styles.heroSub}>
              {uiText("Discover, search, and install from", isZhHant)}{" "}
              <strong className={styles.heroAccent}>
                {data ? allSkillsLocal.length.toLocaleString() : "…"}
              </strong>{" "}
              {isZhHant
                ? `項技能，涵蓋 ${sources.length - 1} 個登錄來源`
                : `skills across ${sources.length - 1} registries`}
              {loadError && (
                <span style={{ color: "#f87171", marginLeft: 8 }}>
                  · {isZhHant ? "目錄載入失敗" : "failed to load catalog"} ({loadError})
                </span>
              )}
            </p>
            {(indexMetaLocal?.indexGeneratedAt || indexMetaLocal?.extractedAt) && (
              <p className={styles.heroSub} style={{ fontSize: "0.85rem", opacity: 0.75 }}>
                {uiText("Catalog refreshed", isZhHant)}{" "}
                <span title={indexMetaLocal.indexGeneratedAt || indexMetaLocal.extractedAt}>
                  {formatRelativeTime(
                    indexMetaLocal.indexGeneratedAt || indexMetaLocal.extractedAt,
                    isZhHant,
                  ) || (isZhHant ? "最近" : "recently")}
                </span>
                {" "}· {isZhHant ? "每日自動重建兩次" : "auto-rebuilt twice daily"}
              </p>
            )}

            <div className={styles.statsRow}>
              <StatCard
                value={allSkillsLocal.filter((s) => s.source === "built-in").length}
                label={uiText("Built-in", isZhHant)}
                color="#4ade80"
              />
              <StatCard
                value={allSkillsLocal.filter((s) => s.source === "optional").length}
                label={uiText("Optional", isZhHant)}
                color="#fbbf24"
              />
              <StatCard
                value={
                  allSkillsLocal.filter(
                    (s) => s.source !== "built-in" && s.source !== "optional"
                  ).length
                }
                label={uiText("Community", isZhHant)}
                color="#60a5fa"
              />
              <StatCard
                value={new Set(allSkillsLocal.map((s) => s.category)).size}
                label={uiText("Categories", isZhHant)}
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
              placeholder={uiText('Search skills... (press "/" to focus)', isZhHant)}
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
                  {src === "all"
                    ? uiText("All", isZhHant)
                    : uiText(conf?.label || src, isZhHant)}
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
            {uiText("Categories", isZhHant)}
            {categoryFilter !== "all" && (
              <span className={styles.activeCatBadge}>
                {categoryEntries.find((c) => c.key === categoryFilter)?.label}
              </span>
            )}
          </button>

          <aside className={`${styles.sidebar} ${sidebarOpen ? styles.sidebarOpen : ""}`}>
            <div className={styles.sidebarHeader}>
              <h2 className={styles.sidebarTitle}>{uiText("Categories", isZhHant)}</h2>
              {categoryFilter !== "all" && (
                <button className={styles.sidebarClear} onClick={() => setCategoryFilter("all")}>
                  {uiText("Clear", isZhHant)}
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
                <span className={styles.catItemLabel}>{uiText("All Skills", isZhHant)}</span>
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
                  {isZhHant
                    ? `${filtered.length} 項結果`
                    : `${filtered.length} result${filtered.length !== 1 ? "s" : ""}`}
                </span>
                {search && (
                  <span className={styles.filterChip}>
                    &ldquo;{search}&rdquo;
                    <button onClick={() => setSearch("")}>&times;</button>
                  </span>
                )}
                {sourceFilter !== "all" && (
                  <span className={styles.filterChip}>
                    {uiText(SOURCE_CONFIG[sourceFilter]?.label || sourceFilter, isZhHant)}
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
                  {uiText("Clear all", isZhHant)}
                </button>
              </div>
            )}

            {!data && !loadError ? (
              <div className={styles.empty}>
                <div className={styles.loadingSpinner} />
                <h3 className={styles.emptyTitle}>{uiText("Loading the catalog…", isZhHant)}</h3>
                <p className={styles.emptyDesc}>
                  {uiText("Fetching 88k+ skills across every registry. One moment.", isZhHant)}
                </p>
              </div>
            ) : visible.length > 0 ? (
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
                        onPick={pickerMode ? pickSkill : undefined}
                        isZhHant={isZhHant}
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
                      {isZhHant
                        ? `顯示更多（剩餘 ${filtered.length - visibleCount} 項）`
                        : `Show more (${filtered.length - visibleCount} remaining)`}
                    </button>
                  </div>
                )}
              </>
            ) : (
              <div className={styles.empty}>
                <div className={styles.emptyIcon}>{"\u{1F50D}"}</div>
                <h3 className={styles.emptyTitle}>{uiText("No skills found", isZhHant)}</h3>
                <p className={styles.emptyDesc}>
                  {uiText("Try a different search term or clear your filters.", isZhHant)}
                </p>
                <button className={styles.emptyReset} onClick={clearAll}>
                  {uiText("Reset all filters", isZhHant)}
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
