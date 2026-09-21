import React, { useState, useMemo, useCallback, useRef, useEffect } from "react";
import Layout from "@theme/Layout";
import Link from "@docusaurus/Link";
import { useHistory } from "@docusaurus/router";
import useBaseUrl from "@docusaurus/useBaseUrl";
import styles from "./styles.module.css";

import {
  type CatalogPlugin,
  type CatalogMeta,
  CATEGORY_CONFIG,
  CATEGORY_ORDER,
  SUBMIT_PLUGIN_URL,
  TIER_CONFIG,
  authorPagePath,
  categoryOf,
  desktopInstallLink,
  formatDate,
  formatRelativeTime,
  formatStars,
  pinUrl,
  pluginPagePath,
  repoUrl,
  tierOf,
} from "../../components/PluginCatalog/catalog";
import CopyButton from "../../components/PluginCatalog/CopyButton";

// Routes Docusaurus serves the static API JSON from. `baseUrl` is `/docs/`,
// `static/api/` ends up at `/docs/api/` — same pattern as the Skills Hub.
const PLUGINS_URL = "/docs/api/plugins.json";
const META_URL = "/docs/api/plugins-meta.json";

const TIER_ORDER = ["all", "official", "community"];

// Sort orders. "stars" is the extractor's own order (stars desc, name), so it
// needs no client-side work; the two date sorts read the git-derived
// addedAt/updatedAt fields and push undated entries last.
type SortKey = "stars" | "newest" | "updated";
const SORT_OPTIONS: { key: SortKey; label: string; title: string }[] = [
  { key: "stars", label: "Most starred", title: "GitHub stars, most first" },
  { key: "newest", label: "Newest", title: "Most recently added to the catalog first" },
  { key: "updated", label: "Recently updated", title: "Most recently re-pinned or edited first" },
];

function dateMs(iso?: string | null): number {
  const t = iso ? new Date(iso).getTime() : NaN;
  return Number.isFinite(t) ? t : -Infinity;
}

function sortPlugins(list: CatalogPlugin[], sort: SortKey): CatalogPlugin[] {
  if (sort === "stars") return list;
  const field = sort === "newest" ? "addedAt" : "updatedAt";
  return [...list].sort(
    (a, b) => dateMs(b[field]) - dateMs(a[field]) || a.name.localeCompare(b.name),
  );
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

function PluginCard({
  plugin,
  query,
  onPick,
  onCategoryClick,
  style,
}: {
  plugin: CatalogPlugin;
  query: string;
  /** Picker embed mode: render "+ Add to this Agent" and call this. */
  onPick?: (plugin: CatalogPlugin) => void;
  onCategoryClick?: (category: string) => void;
  style?: React.CSSProperties;
}) {
  const tier = tierOf(plugin);
  const category = categoryOf(plugin);
  const caps = plugin.capabilities || {};
  const toolCount = caps.providesTools?.length || 0;
  const hookCount = caps.providesHooks?.length || 0;
  const middlewareCount = caps.providesMiddleware?.length || 0;
  const pagePath = pluginPagePath(plugin.name);
  const history = useHistory();
  const pageHref = useBaseUrl(pagePath); // <Link> adds baseUrl itself; history.push does not
  // A card IS the link to the plugin's own page: nothing expands or collapses in place. Inside
  // the Desktop picker iframe an in-frame navigation would leave the host's embed, so the page
  // opens in a new tab there instead.
  const onCardClick = onPick
    ? () => window.open(new URL(pageHref, window.location.href).toString(), "_blank", "noopener,noreferrer")
    : () => history.push(pageHref);

  return (
    <div
      className={styles.card}
      role="link"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter") onCardClick();
      }}
      onClick={onCardClick}
      style={style}
    >
      <div className={styles.cardAccent} style={{ background: tier.color }} />

      {plugin.image && (
        <img
          className={styles.cardImage}
          src={plugin.image}
          alt=""
          loading="lazy"
          decoding="async"
          referrerPolicy="no-referrer"
          onError={(e) => { e.currentTarget.style.display = "none"; }}
        />
      )}

      <div className={styles.cardInner}>
        <div className={styles.cardTop}>
          <span className={styles.cardIcon} title={category.label}>{category.icon}</span>
          <div className={styles.cardTitleGroup}>
            <h3 className={styles.cardTitle}>
              <Link className={styles.cardTitleLink} to={pagePath} onClick={(e) => e.stopPropagation()}>
                {highlightMatch(plugin.name, query)}
              </Link>
            </h3>
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
            {plugin.version && (
              <span className={styles.versionPill} title={`Version ${plugin.version} at ${plugin.sha}`}>
                v{plugin.version.replace(/^v/i, "")}
              </span>
            )}
            {typeof plugin.stars === "number" && (
              <a
                className={styles.starPill}
                href={`${repoUrl(plugin)}/stargazers`}
                target="_blank"
                rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()}
                title={`${plugin.stars.toLocaleString()} GitHub stars`}
              >
                {"\u2605"} {formatStars(plugin.stars)}
              </a>
            )}
          </div>
        </div>

        <p className={styles.cardDesc}>
          {highlightMatch(plugin.description || "No description available.", query)}
        </p>

        <div className={styles.cardMeta}>
          <button
            className={styles.categoryChip}
            onClick={(e) => {
              e.stopPropagation();
              onCategoryClick?.(plugin.category);
            }}
            title={`Filter by ${category.label}`}
          >
            {category.label}
          </button>
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

        {/* Updated is omitted while it equals Added: a fresh entry has nothing to say yet. */}
        {plugin.addedAt && (
          <div className={styles.cardDates}>
            <span title={`Added to the catalog ${formatDate(plugin.addedAt)}`}>
              Added {formatRelativeTime(plugin.addedAt) ?? formatDate(plugin.addedAt)}
            </span>
            {plugin.updatedAt && plugin.updatedAt !== plugin.addedAt && (
              <>
                <span aria-hidden="true" className={styles.cardDatesSep}>·</span>
                <span title={`Last catalog change ${formatDate(plugin.updatedAt)}`}>
                  Updated {formatRelativeTime(plugin.updatedAt) ?? formatDate(plugin.updatedAt)}
                </span>
              </>
            )}
          </div>
        )}

        {onPick ? (
          <button
            className={styles.pickBtn}
            onClick={(e) => {
              e.stopPropagation();
              onPick(plugin);
            }}
          >
            + Add to this Agent
          </button>
        ) : (
          <a
            className={styles.pickBtn}
            href={desktopInstallLink(plugin.name)}
            title="Opens the Install Plugin dialog in Hermes Desktop at the reviewed version. No app? Use the install command below."
            onClick={(e) => e.stopPropagation()}
          >
            Open in Hermes Desktop
          </a>
        )}

        {
          <div className={styles.cardDetail}>
            {plugin.maintainer && (
              <div className={styles.metaRow}>
                <span className={styles.metaLabel}>Maintainer</span>
                <span className={styles.metaValue}>
                  {plugin.maintainerSlug ? (
                    <Link to={authorPagePath(plugin.maintainerSlug)} onClick={(e) => e.stopPropagation()}>
                      {plugin.maintainer}
                    </Link>
                  ) : (
                    plugin.maintainer
                  )}
                </span>
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
                  href={pinUrl(plugin)}
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={(e) => e.stopPropagation()}
                  className={styles.shaLink}
                  title={plugin.sha}
                >
                  <code>{plugin.version ? `${plugin.version} @ ${plugin.shaShort}` : plugin.shaShort}</code> ↗
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
              <Link className={styles.docsLink} to={pagePath} onClick={(e) => e.stopPropagation()}>
                Plugin page →
              </Link>
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
        }
      </div>
    </div>
  );
}

function buildSearchHaystack(p: CatalogPlugin): string {
  return [
    p.name,
    p.description,
    p.maintainer,
    p.tier,
    p.category,
    CATEGORY_CONFIG[p.category]?.label,
    ...(p.capabilities?.providesTools || []),
    ...(p.capabilities?.providesHooks || []),
    ...(p.capabilities?.requiresEnv || []),
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

export default function PluginCatalogPage() {
  // Picker embed mode (?embed=picker): the page is iframed by a host app
  // (Hermes desktop's Capabilities > Plugins tab) as a one-click catalog
  // picker. Site chrome is hidden via CSS and every card gains an
  // "+ Add to this Agent" button that posts
  //   { type: 'hermes-plugin-pick', name, repo, sha, subdir, tier,
  //     installCmd }
  // to the parent window. The HOST performs the actual install through its
  // own gateway (plugins.manage, catalog_name=<name>) — this page never
  // installs anything; parents must validate event.origin before acting.
  const pickerMode =
    typeof window !== "undefined" &&
    new URLSearchParams(window.location.search).get("embed") === "picker";

  const pickPlugin = useCallback((plugin: CatalogPlugin) => {
    if (typeof window === "undefined" || window.parent === window) return;
    window.parent.postMessage(
      {
        type: "hermes-plugin-pick",
        name: plugin.name,
        repo: plugin.repo,
        sha: plugin.sha,
        subdir: plugin.subdir || "",
        tier: plugin.tier,
        installCmd: plugin.installCommand || `hermes plugins install ${plugin.name}`,
      },
      "*"
    );
  }, []);

  const [data, setData] = useState<{ plugins: CatalogPlugin[]; meta: CatalogMeta } | null>(
    null,
  );
  const [loadError, setLoadError] = useState<string | null>(null);

  const [search, setSearch] = useState("");
  const [tierFilter, setTierFilter] = useState("all");
  const [categoryFilter, setCategoryFilter] = useState("all");
  const [sort, setSort] = useState<SortKey>("stars");
  const [filtersOpen, setFiltersOpen] = useState(false);
  const searchRef = useRef<HTMLInputElement>(null);
  const filterPanelRef = useRef<HTMLDivElement>(null);

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
      const target = e.target instanceof HTMLElement ? e.target : null;
      const isEditable =
        target?.matches("input, textarea, select") ||
        target?.isContentEditable ||
        Boolean(target?.closest("[contenteditable='true']"));
      if (e.key === "/" && !isEditable) {
        e.preventDefault();
        e.stopImmediatePropagation();
        searchRef.current?.focus();
      }
      if (e.key === "Escape") {
        searchRef.current?.blur();
        setFiltersOpen(false);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  const allPlugins: CatalogPlugin[] = data?.plugins ?? [];
  const meta: CatalogMeta = data?.meta ?? {};

  const filtered = useMemo(() => {
    const q = search.toLowerCase().trim();
    const matching = allPlugins.filter((p) => {
      if (tierFilter !== "all" && p.tier !== tierFilter) return false;
      if (categoryFilter !== "all" && p.category !== categoryFilter) return false;
      if (q) return (p._search || "").includes(q);
      return true;
    });
    return sortPlugins(matching, sort);
  }, [search, tierFilter, categoryFilter, sort, allPlugins]);

  // Browse mode (no search, no category picked): render one section per
  // category so Memory, Desktop, Platforms… read as distinct shelves rather
  // than one undifferentiated wall. Filtering or searching flattens to a grid.
  const grouped = useMemo(() => {
    if (search.trim() || categoryFilter !== "all") return null;
    const buckets = new Map<string, CatalogPlugin[]>();
    for (const p of filtered) {
      const key = CATEGORY_CONFIG[p.category] ? p.category : "general";
      (buckets.get(key) ?? buckets.set(key, []).get(key)!).push(p);
    }
    return CATEGORY_ORDER.filter((c) => buckets.has(c)).map((c) => [c, buckets.get(c)!] as const);
  }, [filtered, search, categoryFilter]);

  const categoryCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const p of allPlugins) {
      if (tierFilter !== "all" && p.tier !== tierFilter) continue;
      const key = CATEGORY_CONFIG[p.category] ? p.category : "general";
      counts[key] = (counts[key] || 0) + 1;
    }
    return counts;
  }, [allPlugins, tierFilter]);

  useEffect(() => {
  }, [search, tierFilter, categoryFilter]);

  const clearAll = useCallback(() => {
    setSearch("");
    setTierFilter("all");
    setCategoryFilter("all");
    setFiltersOpen(false);
  }, []);

  const pickCategory = useCallback((c: string) => {
    setCategoryFilter((cur) => (cur === c ? "all" : c));
    setSearch("");
  }, []);

  const catalogEmpty = data !== null && allPlugins.length === 0;

  const renderCard = (plugin: CatalogPlugin, i: number) => {
    const key = `${plugin.tier}-${plugin.name}`;
    return (
      <PluginCard
        key={key}
        plugin={plugin}
        query={search}
        onPick={pickerMode ? pickPlugin : undefined}
        onCategoryClick={pickCategory}
        style={{ animationDelay: `${Math.min(i, 20) * 25}ms` }}
      />
    );
  };

  return (
    <Layout
      title="Plugin Catalog"
      description="Give Hermes new powers: reviewed plugins you can install in one click"
    >
      <div className={`${styles.page} ${pickerMode ? styles.pickerMode : ""}`}>
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
              Give Hermes new powers. Memory, voice, messaging, browsing, Desktop panes and more,
              built by the community.
              {loadError && (
                <span style={{ color: "#f87171", marginLeft: 8 }}>
                  · failed to load catalog ({loadError})
                </span>
              )}
            </p>
            {!catalogEmpty && (
              <p className={styles.heroSub} style={{ fontSize: "0.9rem" }}>
                Built a plugin?{" "}
                <Link className={styles.heroLink} to={SUBMIT_PLUGIN_URL}>
                  Submit it to the catalog →
                </Link>
              </p>
            )}
            {meta.generatedAt && !catalogEmpty && (
              <p className={styles.heroMeta}>
                {allPlugins.length} plugins across {Object.keys(categoryCounts).length} categories
                {" · "}updated{" "}
                <span
                  title={
                    meta.starsFetchedAt
                      ? `Catalog ${meta.generatedAt}; popularity ranking as of ${meta.starsFetchedAt}`
                      : meta.generatedAt
                  }
                >
                  {formatRelativeTime(meta.generatedAt) || "recently"}
                </span>
              </p>
            )}
          </div>
        </header>

        {!catalogEmpty && (
          <div className={styles.controlsBar}>
            <div className={styles.controlsTopRow}>
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
                placeholder="Search plugins"
                title='Tip: press "/" to jump here'
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

            <button
              type="button"
              className={styles.filterToggle}
              aria-expanded={filtersOpen}
              aria-controls="plugin-directory-filters"
              onClick={() => {
                if (filtersOpen) {
                  setFiltersOpen(false);
                  return;
                }
                setFiltersOpen(true);
                requestAnimationFrame(() => {
                  filterPanelRef.current?.scrollIntoView({ block: "start" });
                });
              }}
            >
              Filters
              {(tierFilter !== "all" || categoryFilter !== "all") && (
                <span className={styles.activeFilterCount}>
                  {Number(tierFilter !== "all") + Number(categoryFilter !== "all")}
                </span>
              )}
            </button>
            <label className={styles.compactSelect}>
              <span>Source</span>
              <select value={tierFilter} onChange={(e) => setTierFilter(e.target.value)}>
                {TIER_ORDER.map((tier) => (
                  <option key={tier} value={tier}>{tier === "all" ? "All sources" : TIER_CONFIG[tier]?.label || tier}</option>
                ))}
              </select>
            </label>
            <label className={styles.compactSelect}>
              <span>Category</span>
              <select value={categoryFilter} onChange={(e) => setCategoryFilter(e.target.value)}>
                <option value="all">All categories</option>
                {CATEGORY_ORDER.filter((c) => categoryCounts[c]).map((c) => (
                  <option key={c} value={c}>{CATEGORY_CONFIG[c].label}</option>
                ))}
              </select>
            </label>
            <label className={`${styles.compactSelect} ${styles.compactSort}`}>
              <span>Sort</span>
              <select value={sort} onChange={(e) => setSort(e.target.value as SortKey)}>
                {SORT_OPTIONS.map((opt) => <option key={opt.key} value={opt.key}>{opt.label}</option>)}
              </select>
            </label>
            </div>

            <div
              id="plugin-directory-filters"
              ref={filterPanelRef}
              className={`${styles.filterPanel} ${filtersOpen ? styles.filterPanelOpen : ""}`}
            >
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

            <div className={styles.sortPills} role="radiogroup" aria-label="Sort plugins">
              <span className={styles.sortLabel}>Sort</span>
              {SORT_OPTIONS.map((opt) => {
                const active = sort === opt.key;
                return (
                  <button
                    key={opt.key}
                    className={`${styles.tierBtn} ${active ? styles.sortBtnActive : ""}`}
                    onClick={() => setSort(opt.key)}
                    role="radio"
                    aria-checked={active}
                    title={opt.title}
                  >
                    {opt.label}
                  </button>
                );
              })}
            </div>

            <div className={styles.categoryPills} role="tablist" aria-label="Plugin categories">
              <button
                className={`${styles.categoryBtn} ${categoryFilter === "all" ? styles.categoryBtnActive : ""}`}
                onClick={() => setCategoryFilter("all")}
                role="tab"
                aria-selected={categoryFilter === "all"}
              >
                All categories
              </button>
              {CATEGORY_ORDER.filter((c) => categoryCounts[c]).map((c) => {
                const conf = CATEGORY_CONFIG[c];
                const active = categoryFilter === c;
                return (
                  <button
                    key={c}
                    className={`${styles.categoryBtn} ${active ? styles.categoryBtnActive : ""}`}
                    onClick={() => pickCategory(c)}
                    role="tab"
                    aria-selected={active}
                    title={conf.blurb}
                  >
                    <span aria-hidden="true">{conf.icon}</span> {conf.label}
                    <span className={styles.tierCount}>{categoryCounts[c]}</span>
                  </button>
                );
              })}
            </div>
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
                <Link className={styles.emptyCta} to={SUBMIT_PLUGIN_URL}>
                  How to submit a plugin
                </Link>
                <Link className={styles.emptyCtaSecondary} to="/user-guide/features/plugin-catalog">
                  Read the catalog docs
                </Link>
              </div>
            </div>
          ) : filtered.length > 0 && grouped ? (
            grouped.map(([cat, plugins]) => {
              const conf = CATEGORY_CONFIG[cat];
              return (
                <section key={cat} className={styles.categorySection} aria-labelledby={`cat-${cat}`}>
                  <header className={styles.categoryHeader}>
                    <h2 id={`cat-${cat}`} className={styles.categoryTitle}>
                      <span aria-hidden="true">{conf.icon}</span> {conf.label}
                      <span className={styles.tierCount}>{plugins.length}</span>
                    </h2>
                    <p className={styles.categoryBlurb}>{conf.blurb}</p>
                    <button className={styles.categoryViewAll} onClick={() => pickCategory(cat)}>
                      View only {conf.label} →
                    </button>
                  </header>
                  <div className={styles.grid}>
                    {plugins.map((plugin, i) => renderCard(plugin, i))}
                  </div>
                </section>
              );
            })
          ) : filtered.length > 0 ? (
            <>
              <div className={styles.resultsBar} role="status">
                {categoryFilter !== "all" && CATEGORY_CONFIG[categoryFilter] && (
                  <span>
                    <span aria-hidden="true">{CATEGORY_CONFIG[categoryFilter].icon}</span>{" "}
                    <strong>{CATEGORY_CONFIG[categoryFilter].label}</strong>
                    <span style={{ opacity: 0.7 }}> · {CATEGORY_CONFIG[categoryFilter].blurb}</span>
                  </span>
                )}
                {search.trim() && (
                  <span>
                    Results for <strong>“{search.trim()}”</strong>
                  </span>
                )}
                <span>
                  {filtered.length} plugin{filtered.length === 1 ? "" : "s"}
                </span>
                <button className={styles.resultsClear} onClick={clearAll}>
                  Clear filters
                </button>
              </div>
              <div className={styles.grid}>{filtered.map((plugin, i) => renderCard(plugin, i))}</div>
            </>
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
