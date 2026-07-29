import React, { useState, useMemo, useCallback, useRef, useEffect } from "react";
import Layout from "@theme/Layout";
import Link from "@docusaurus/Link";
import styles from "./styles.module.css";

interface PluginCapabilities {
  providesTools?: string[];
  providesHooks?: string[];
  providesMiddleware?: string[];
  requiresEnv?: string[];
}

interface CatalogPlugin {
  name: string;
  description: string;
  repo: string;
  sha: string;
  shaShort: string;
  tier: string;
  maintainer: string;
  requiresHermes?: string;
  platforms?: string[];
  capabilities?: PluginCapabilities;
  docsUrl?: string;
  installCommand: string;
  /** Lowercase pre-joined haystack for the search filter (built at load). */
  _search?: string;
}

interface CatalogMeta {
  generatedAt?: string;
  total?: number;
  byTier?: Record<string, number>;
  removedCount?: number;
}

// Routes Docusaurus serves the static API JSON from. `baseUrl` is `/docs/`,
// `static/api/` ends up at `/docs/api/` — same pattern as the Skills Hub.
const PLUGINS_URL = "/docs/api/plugins.json";
const META_URL = "/docs/api/plugins-meta.json";

const CATALOG_README_URL =
  "https://github.com/NousResearch/hermes-agent/tree/main/plugin-catalog";

const TIER_CONFIG: Record<
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

const TIER_ORDER = ["all", "official", "community"];

function formatRelativeTime(iso?: string): string | null {
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

function CopyButton({ text }: { text: string }) {
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
      title="Copy install command"
      aria-label="Copy install command"
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
      <span className={styles.copyBtnLabel}>{copied ? "Copied" : "Copy"}</span>
    </button>
  );
}

function PluginCard({
  plugin,
  query,
  expanded,
  onToggle,
  style,
}: {
  plugin: CatalogPlugin;
  query: string;
  expanded: boolean;
  onToggle: () => void;
  style?: React.CSSProperties;
}) {
  const tier = TIER_CONFIG[plugin.tier] || TIER_CONFIG.community;
  const caps = plugin.capabilities || {};
  const toolCount = caps.providesTools?.length || 0;
  const hookCount = caps.providesHooks?.length || 0;
  const middlewareCount = caps.providesMiddleware?.length || 0;
  const pinUrl = `${plugin.repo.replace(/\.git$/, "").replace(/\/$/, "")}/tree/${plugin.sha}`;

  return (
    <div
      className={`${styles.card} ${expanded ? styles.cardExpanded : ""}`}
      onClick={onToggle}
      style={style}
    >
      <div className={styles.cardAccent} style={{ background: tier.color }} />

      <div className={styles.cardInner}>
        <div className={styles.cardTop}>
          <span className={styles.cardIcon}>{"\u{1F50C}"}</span>
          <div className={styles.cardTitleGroup}>
            <h3 className={styles.cardTitle}>{highlightMatch(plugin.name, query)}</h3>
            <span
              className={styles.tierPill}
              style={{
                color: tier.color,
                background: tier.bg,
                borderColor: tier.border,
              }}
            >
              {tier.icon} {tier.label}
            </span>
          </div>
        </div>

        <p className={`${styles.cardDesc} ${expanded ? styles.cardDescFull : ""}`}>
          {highlightMatch(plugin.description || "No description available.", query)}
        </p>

        <div className={styles.cardMeta}>
          {toolCount > 0 && (
            <span className={styles.capChip}>
              {toolCount} tool{toolCount === 1 ? "" : "s"}
            </span>
          )}
          {hookCount > 0 && (
            <span className={styles.capChip}>
              {hookCount} hook{hookCount === 1 ? "" : "s"}
            </span>
          )}
          {middlewareCount > 0 && (
            <span className={styles.capChip}>
              {middlewareCount} middleware
            </span>
          )}
          {caps.requiresEnv?.map((v) => (
            <code key={v} className={styles.envChip}>
              {v}
            </code>
          ))}
          {plugin.platforms?.map((p) => (
            <span key={p} className={styles.platformPill}>
              {p === "macos" ? "\u{F8FF} macOS" : p === "linux" ? "\u{1F427} Linux" : p}
            </span>
          ))}
        </div>

        {expanded && (
          <div className={styles.cardDetail}>
            {plugin.maintainer && (
              <div className={styles.metaRow}>
                <span className={styles.metaLabel}>Maintainer</span>
                <span className={styles.metaValue}>{plugin.maintainer}</span>
              </div>
            )}
            {plugin.requiresHermes && (
              <div className={styles.metaRow}>
                <span className={styles.metaLabel}>Requires</span>
                <span className={styles.metaValue}>
                  <code>hermes {plugin.requiresHermes}</code>
                </span>
              </div>
            )}
            <div className={styles.metaRow}>
              <span className={styles.metaLabel}>Pinned</span>
              <span className={styles.metaValue}>
                <a
                  href={pinUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={(e) => e.stopPropagation()}
                  className={styles.shaLink}
                  title={plugin.sha}
                >
                  <code>{plugin.shaShort}</code> ↗
                </a>
              </span>
            </div>
            {caps.providesTools?.length ? (
              <div className={styles.metaRow}>
                <span className={styles.metaLabel}>Tools</span>
                <span className={styles.chipList}>
                  {caps.providesTools.map((t) => (
                    <code key={t} className={styles.envChip}>
                      {t}
                    </code>
                  ))}
                </span>
              </div>
            ) : null}
            <div className={styles.installHint}>
              <code>{plugin.installCommand}</code>
              <CopyButton text={plugin.installCommand} />
            </div>
            <div className={styles.cardLinks}>
              <a
                className={styles.docsLink}
                href={plugin.repo}
                target="_blank"
                rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()}
              >
                Repository ↗
              </a>
              {plugin.docsUrl ? (
                <a
                  className={styles.docsLink}
                  href={plugin.docsUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={(e) => e.stopPropagation()}
                >
                  Documentation ↗
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

function buildSearchHaystack(p: CatalogPlugin): string {
  return [
    p.name,
    p.description,
    p.maintainer,
    p.tier,
    ...(p.capabilities?.providesTools || []),
    ...(p.capabilities?.providesHooks || []),
    ...(p.capabilities?.requiresEnv || []),
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

export default function PluginCatalogPage() {
  const [data, setData] = useState<{ plugins: CatalogPlugin[]; meta: CatalogMeta } | null>(
    null,
  );
  const [loadError, setLoadError] = useState<string | null>(null);

  const [search, setSearch] = useState("");
  const [tierFilter, setTierFilter] = useState("all");
  const [expandedCard, setExpandedCard] = useState<string | null>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [pl, mt] = await Promise.all([
          fetch(PLUGINS_URL).then((r) => {
            if (!r.ok) throw new Error(`plugins.json HTTP ${r.status}`);
            return r.json();
          }),
          fetch(META_URL).then((r) => (r.ok ? r.json() : {})).catch(() => ({})),
        ]);
        if (cancelled) return;
        const arr = Array.isArray(pl) ? (pl as CatalogPlugin[]) : [];
        for (const p of arr) p._search = buildSearchHaystack(p);
        setData({ plugins: arr, meta: mt || {} });
      } catch (err) {
        if (cancelled) return;
        setLoadError(err instanceof Error ? err.message : String(err));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

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

  const allPlugins: CatalogPlugin[] = data?.plugins ?? [];
  const meta: CatalogMeta = data?.meta ?? {};

  const filtered = useMemo(() => {
    const q = search.toLowerCase().trim();
    return allPlugins.filter((p) => {
      if (tierFilter !== "all" && p.tier !== tierFilter) return false;
      if (q) return (p._search || "").includes(q);
      return true;
    });
  }, [search, tierFilter, allPlugins]);

  useEffect(() => {
    setExpandedCard(null);
  }, [search, tierFilter]);

  const clearAll = useCallback(() => {
    setSearch("");
    setTierFilter("all");
  }, []);

  const catalogEmpty = data !== null && allPlugins.length === 0;

  return (
    <Layout
      title="Plugin Catalog"
      description="Browse reviewed, SHA-pinned plugins for Hermes Agent"
    >
      <div className={styles.page}>
        <header className={styles.hero}>
          <div className={styles.heroGlow} />
          <div className={styles.heroContent}>
            <p className={styles.heroEyebrow}>Hermes Agent</p>
            <h1 className={styles.heroTitle}>Plugin Catalog</h1>
            <nav className={styles.crossNav} aria-label="Catalog pages">
              <Link className={styles.crossNavLink} to="/skills">
                Skills
              </Link>
              <span className={`${styles.crossNavLink} ${styles.crossNavActive}`}>
                Plugins
              </span>
            </nav>
            <p className={styles.heroSub}>
              Reviewed, SHA-pinned plugins you can install with one command.
              {loadError && (
                <span style={{ color: "#f87171", marginLeft: 8 }}>
                  · failed to load catalog ({loadError})
                </span>
              )}
            </p>
            {meta.generatedAt && !catalogEmpty && (
              <p className={styles.heroSub} style={{ fontSize: "0.85rem", opacity: 0.75 }}>
                Catalog refreshed{" "}
                <span title={meta.generatedAt}>
                  {formatRelativeTime(meta.generatedAt) || "recently"}
                </span>
              </p>
            )}

            {!catalogEmpty && (
              <div className={styles.statsRow}>
                <StatCard
                  value={allPlugins.filter((p) => p.tier === "official").length}
                  label="Official"
                  color="#ffd700"
                />
                <StatCard
                  value={allPlugins.filter((p) => p.tier === "community").length}
                  label="Community"
                  color="#94a3b8"
                />
                <StatCard value={meta.removedCount ?? 0} label="Removed" color="#f87171" />
              </div>
            )}
          </div>
        </header>

        {!catalogEmpty && (
          <div className={styles.controlsBar}>
            <div className={styles.searchWrap}>
              <svg
                className={styles.searchIcon}
                viewBox="0 0 20 20"
                fill="currentColor"
                width="18"
                height="18"
              >
                <path
                  fillRule="evenodd"
                  d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z"
                  clipRule="evenodd"
                />
              </svg>
              <input
                ref={searchRef}
                type="text"
                placeholder='Search plugins... (press "/" to focus)'
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

            <div className={styles.tierPills}>
              {TIER_ORDER.map((tier) => {
                const active = tierFilter === tier;
                const conf = TIER_CONFIG[tier];
                const count =
                  tier === "all"
                    ? allPlugins.length
                    : allPlugins.filter((p) => p.tier === tier).length;
                return (
                  <button
                    key={tier}
                    className={`${styles.tierBtn} ${active ? styles.tierBtnActive : ""}`}
                    onClick={() => setTierFilter(tier)}
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
                    {tier === "all" ? "All" : conf?.label || tier}
                    <span className={styles.tierCount}>{count}</span>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        <main className={styles.main}>
          {!data && !loadError ? (
            <div className={styles.empty}>
              <div className={styles.loadingSpinner} />
              <h3 className={styles.emptyTitle}>Loading the catalog…</h3>
            </div>
          ) : catalogEmpty ? (
            <div className={styles.empty}>
              <div className={styles.emptyIcon}>{"\u{1F331}"}</div>
              <h3 className={styles.emptyTitle}>The catalog is just getting started</h3>
              <p className={styles.emptyDesc}>
                The plugin catalog is a curated, human-reviewed list of Hermes
                plugins — each entry pinned to an exact commit. Want yours listed?
                Submissions are open.
              </p>
              <div className={styles.emptyActions}>
                <a
                  className={styles.emptyCta}
                  href={CATALOG_README_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  How to submit a plugin ↗
                </a>
                <Link className={styles.emptyCtaSecondary} to="/user-guide/features/plugin-catalog">
                  Read the catalog docs
                </Link>
              </div>
            </div>
          ) : filtered.length > 0 ? (
            <div className={styles.grid}>
              {filtered.map((plugin, i) => {
                const key = `${plugin.tier}-${plugin.name}`;
                return (
                  <PluginCard
                    key={key}
                    plugin={plugin}
                    query={search}
                    expanded={expandedCard === key}
                    onToggle={() => setExpandedCard(expandedCard === key ? null : key)}
                    style={{ animationDelay: `${Math.min(i, 20) * 25}ms` }}
                  />
                );
              })}
            </div>
          ) : (
            <div className={styles.empty}>
              <div className={styles.emptyIcon}>{"\u{1F50D}"}</div>
              <h3 className={styles.emptyTitle}>No plugins found</h3>
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
    </Layout>
  );
}
