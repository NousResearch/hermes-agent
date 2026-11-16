// Port of hermes-agent website/src/pages/plugins/index.tsx (Docusaurus) with
// website/src/components/PluginCatalog/catalog.ts, CopyButton.tsx and
// apps/shared/src/catalog-browse.ts inlined. The featured card, copy button and
// shared formatting come from the Skills Hub snippet, as the source shares them.
import {
  CatalogFeatured,
  CopyButton,
  formatRelativeTime,
  highlightMatch,
  pickFeatured,
} from "/snippets/skills-hub.jsx";

// Rebuilt by website/scripts/extract-plugins.py and served from the same public
// origin the source page reads; fetched at runtime rather than stored here.
export const PLUGINS_DATA_URL = "https://nousresearch.github.io/hermes-agent/docs/api/plugins.json";
export const PLUGINS_META_URL = "https://nousresearch.github.io/hermes-agent/docs/api/plugins-meta.json";
// Per-plugin and author pages are generated routes on the source site and are not
// part of this preview, so cards link out to them.
export const PLUGIN_SITE_BASE = "https://hermes-agent.nousresearch.com/docs/plugins";
export const SUBMIT_PLUGIN_URL = "/developer-guide/plugins/catalog-submission";
export const PLUGIN_CATALOG_DOCS_URL = "/user-guide/features/plugin-catalog";
export const MOBILE_PANEL_QUERY = "(max-width: 600px)";

export const TIER_ORDER = ["all", "official", "verified", "community"];

export const TIER_CONFIG = {
  official: { label: "Official", color: "#0000f2", icon: "\u{2713}" },
  verified: { label: "Verified", color: "#00b35f", icon: "\u{2714}" },
  community: { label: "Community", color: "#8a8ab0", icon: "\u{2756}" },
};

export const CATEGORY_CONFIG = {
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

export const SORT_OPTIONS = [
  { key: "stars", label: "Most starred", title: "GitHub stars, most first" },
  { key: "newest", label: "Newest", title: "Most recently added to the catalog first" },
  { key: "updated", label: "Recently updated", title: "Most recently re-pinned or edited first" },
];

export const CARD_TOOL_CHIP_LIMIT = 3;

export const categoryOf = (plugin) => CATEGORY_CONFIG[plugin.category] || CATEGORY_CONFIG.general;
export const tierOf = (plugin) => TIER_CONFIG[plugin.tier] || TIER_CONFIG.community;
export const pluginPageUrl = (name) => `${PLUGIN_SITE_BASE}/${encodeURIComponent(name)}`;
export const authorPageUrl = (slug) => `${PLUGIN_SITE_BASE}/by/${encodeURIComponent(slug)}`;
export const desktopInstallLink = (name) => `hermes://plugin/install?catalog=${encodeURIComponent(name)}`;
export const repoUrl = (plugin) => plugin.repo.replace(/\.git$/, "").replace(/\/$/, "");

export const pinUrl = (plugin) => {
  const base = repoUrl(plugin);
  const tree = /^https:\/\/gitlab\.com\//.test(base) ? `${base}/-/tree/${plugin.sha}` : `${base}/tree/${plugin.sha}`;
  return plugin.subdir ? `${tree}/${plugin.subdir.replace(/^\/+|\/+$/g, "")}` : tree;
};

// Only http(s) docs links: a hand-edited plugins.json must not produce a javascript: href.
export const docsHref = (plugin) => {
  try {
    const url = new URL(String(plugin.docsUrl || "").trim());
    return url.protocol === "https:" || url.protocol === "http:" ? url.toString() : null;
  } catch {
    return null;
  }
};

export const capToolChips = (tools, max = CARD_TOOL_CHIP_LIMIT) => {
  const all = tools ?? [];
  return { shown: all.slice(0, max), hidden: Math.max(0, all.length - max) };
};

export const formatStars = (n) => (n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : String(n));

export const formatDate = (iso) => {
  if (!iso) return null;
  const d = new Date(iso);
  if (!Number.isFinite(d.getTime())) return null;
  return d.toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric", timeZone: "UTC" });
};

// Cards show the description minus its trailing "Disclosure — …" sentences.
export const cardSummary = (description) => {
  const idx = description.search(/\bDisclosure\s*[—–:-]\s*/);
  if (idx === -1) return description;
  return description.slice(0, idx).trim() || description;
};

// The published snapshot is already sorted by stars desc, then name.
export const sortCatalogPlugins = (entries, sort) => {
  if (sort === "stars") return entries;
  const field = sort === "newest" ? "addedAt" : "updatedAt";
  const date = (value) => {
    const ms = value ? Date.parse(value) : NaN;
    return Number.isFinite(ms) ? ms : -Infinity;
  };
  return [...entries].sort((a, b) => date(b[field]) - date(a[field]) || a.name.localeCompare(b.name));
};

export const groupCatalogPlugins = (entries) => {
  const buckets = new Map();
  for (const entry of entries) {
    const key = Object.prototype.hasOwnProperty.call(CATEGORY_CONFIG, entry.category) ? entry.category : "general";
    const bucket = buckets.get(key) ?? [];
    bucket.push(entry);
    buckets.set(key, bucket);
  }
  return CATEGORY_ORDER.filter((key) => buckets.has(key)).map((key) => [key, buckets.get(key)]);
};

export const buildPluginSearchHaystack = (p) =>
  [
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

export const PluginCard = ({ plugin, query, onPick, onCategoryClick }) => {
  const tier = tierOf(plugin);
  const category = categoryOf(plugin);
  const caps = plugin.capabilities || {};
  const toolCount = caps.providesTools?.length || 0;
  const hookCount = caps.providesHooks?.length || 0;
  const middlewareCount = caps.providesMiddleware?.length || 0;
  const toolChips = capToolChips(caps.providesTools);
  const description = cardSummary(plugin.description || "") || "No description available.";
  const [imageBroken, setImageBroken] = useState(false);
  const showImage = Boolean(plugin.image) && !imageBroken;
  const pageUrl = pluginPageUrl(plugin.name);
  const docs = docsHref(plugin);
  const openPage = () => window.open(pageUrl, "_blank", "noopener,noreferrer");
  const stop = (e) => e.stopPropagation();

  return (
    <div
      className="hh-card pc-card"
      role="link"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter") openPage();
      }}
      onClick={openPage}
      style={{ "--hh-accent": tier.color }}
    >
      {showImage ? (
        <img
          className="pc-card-image"
          src={plugin.image}
          alt=""
          loading="lazy"
          decoding="async"
          referrerPolicy="no-referrer"
          onError={() => setImageBroken(true)}
        />
      ) : (
        <div className="pc-card-image pc-card-placeholder" aria-hidden="true">
          <span>{category.icon}</span>
        </div>
      )}

      <div className="hh-card-inner">
        <div className="hh-card-top">
          <span className="hh-card-icon" title={category.label}>{category.icon}</span>
          <div className="hh-card-title-group pc-title-group">
            <h3 className="hh-card-title" title={plugin.name}>
              <a className="pc-title-link" href={pageUrl} target="_blank" rel="noopener noreferrer" onClick={stop}>
                {highlightMatch(plugin.name, query)}
              </a>
            </h3>
            {typeof plugin.stars === "number" && (
              <a
                className="pc-star-pill"
                href={`${repoUrl(plugin)}/stargazers`}
                target="_blank"
                rel="noopener noreferrer"
                onClick={stop}
                title={`${plugin.stars.toLocaleString()} GitHub stars`}
              >
                {"\u2605"} {formatStars(plugin.stars)}
              </a>
            )}
          </div>
        </div>

        <p className="hh-card-desc" title={description}>{highlightMatch(description, query)}</p>

        <div className="hh-card-meta">
          <span className="hh-source-pill">
            <span className="hh-dot" aria-hidden="true" />
            {tier.label}
          </span>
          {plugin.version && (
            <span className="hh-chip hh-chip-muted" title={`Version ${plugin.version} at ${plugin.sha}`}>
              v{plugin.version.replace(/^v/i, "")}
            </span>
          )}
          <button
            type="button"
            className="hh-chip hh-chip-button"
            onClick={(e) => {
              e.stopPropagation();
              onCategoryClick?.(plugin.category);
            }}
            title={`Filter by ${category.label}`}
          >
            {category.label}
          </button>
          {toolCount > 0 && <span className="hh-chip hh-chip-muted">{toolCount} tool{toolCount === 1 ? "" : "s"}</span>}
          {hookCount > 0 && <span className="hh-chip hh-chip-muted">{hookCount} hook{hookCount === 1 ? "" : "s"}</span>}
          {middlewareCount > 0 && <span className="hh-chip hh-chip-muted">{middlewareCount} middleware</span>}
          {caps.requiresEnv?.map((v) => <code key={v} className="pc-env-chip">{v}</code>)}
          {plugin.platforms?.map((p) => (
            <span key={p} className="hh-chip hh-chip-muted">{p === "macos" ? "macOS" : p === "linux" ? "Linux" : p}</span>
          ))}
        </div>

        <div className="pc-card-dates">
          {plugin.addedAt && (
            <span title={`Added to the catalog ${formatDate(plugin.addedAt)}`}>
              Added {formatRelativeTime(plugin.addedAt) ?? formatDate(plugin.addedAt)}
            </span>
          )}
          {plugin.addedAt && plugin.updatedAt && plugin.updatedAt !== plugin.addedAt && (
            <>
              <span aria-hidden="true"> · </span>
              <span title={`Last catalog change ${formatDate(plugin.updatedAt)}`}>
                Updated {formatRelativeTime(plugin.updatedAt) ?? formatDate(plugin.updatedAt)}
              </span>
            </>
          )}
        </div>

        {onPick ? (
          <button
            type="button"
            className="hh-pick-btn"
            onClick={(e) => {
              e.stopPropagation();
              onPick(plugin);
            }}
          >
            + Add to this Agent
          </button>
        ) : (
          <a
            className="hh-pick-btn"
            href={desktopInstallLink(plugin.name)}
            title="Opens the Install Plugin dialog in Hermes Desktop at the reviewed version. No app? Use the install command below."
            onClick={stop}
          >
            Open in Hermes Desktop
          </a>
        )}

        <div className="pc-card-detail">
          <div className="pc-meta-row">
            <span className="hermes-caption pc-meta-label">Maintainer</span>
            <span className="pc-meta-value" title={plugin.maintainer || undefined}>
              {plugin.maintainerSlug ? (
                <a href={authorPageUrl(plugin.maintainerSlug)} target="_blank" rel="noopener noreferrer" onClick={stop}>
                  {plugin.maintainer}
                </a>
              ) : (
                plugin.maintainer || <span className="pc-meta-empty">—</span>
              )}
            </span>
          </div>
          <div className="pc-meta-row">
            <span className="hermes-caption pc-meta-label">Pinned</span>
            <span
              className="pc-meta-value"
              title={plugin.requiresHermes ? `${plugin.sha} · requires hermes ${plugin.requiresHermes}` : plugin.sha}
            >
              <a href={pinUrl(plugin)} target="_blank" rel="noopener noreferrer" onClick={stop}>
                <code>{plugin.version ? `${plugin.version} @ ${plugin.shaShort}` : plugin.shaShort}</code> ↗
              </a>
              {plugin.requiresHermes && (
                <>
                  <span aria-hidden="true"> · </span>
                  <code>hermes {plugin.requiresHermes}</code>
                </>
              )}
            </span>
          </div>
          <div className="pc-meta-row">
            <span className="hermes-caption pc-meta-label">Tools</span>
            {toolChips.shown.length ? (
              <span className="pc-meta-value" title={caps.providesTools?.join(", ")}>
                <code>{toolChips.shown.join(", ")}</code>
                {toolChips.hidden > 0 && <code className="pc-env-chip pc-more-chip">+{toolChips.hidden}</code>}
              </span>
            ) : (
              <span className="pc-meta-value pc-meta-empty">—</span>
            )}
          </div>
          <div className="hh-install-hint" onClick={stop}>
            <code>{plugin.installCommand}</code>
            <CopyButton text={plugin.installCommand} />
          </div>
          <div className="pc-card-links">
            <a className="hh-docs-link" href={pageUrl} target="_blank" rel="noopener noreferrer" onClick={stop}>
              Plugin page ↗
            </a>
            <a className="hh-docs-link" href={plugin.repo} target="_blank" rel="noopener noreferrer" onClick={stop}>
              Repository ↗
            </a>
            {docs ? (
              <a className="hh-docs-link" href={docs} target="_blank" rel="noopener noreferrer" onClick={stop}>
                Documentation ↗
              </a>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
};

export const PluginCatalog = () => {
  // ?embed=picker: Hermes desktop iframes this page as a plugin picker; the host
  // performs the install, this page only posts the pick to the parent window.
  const [pickerMode, setPickerMode] = useState(false);
  useEffect(() => {
    setPickerMode(new URLSearchParams(window.location.search).get("embed") === "picker");
  }, []);

  const pickPlugin = useCallback((plugin) => {
    if (window.parent === window) return;
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
      "*",
    );
  }, []);

  const [data, setData] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [search, setSearch] = useState("");
  const [tierFilter, setTierFilter] = useState("all");
  const [categoryFilter, setCategoryFilter] = useState("all");
  const [sort, setSort] = useState("stars");
  const [filtersOpen, setFiltersOpen] = useState(false);
  const searchRef = useRef(null);
  const filterPanelRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [pl, mt] = await Promise.all([
          fetch(PLUGINS_DATA_URL).then((r) => {
            if (!r.ok) throw new Error(`plugins.json HTTP ${r.status}`);
            return r.json();
          }),
          fetch(PLUGINS_META_URL)
            .then((r) => (r.ok ? r.json() : {}))
            .catch(() => ({})),
        ]);
        if (cancelled) return;
        const arr = Array.isArray(pl) ? pl : [];
        for (const p of arr) p._search = buildPluginSearchHaystack(p);
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
    const handler = (e) => {
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

  useEffect(() => {
    if (!filtersOpen) return;
    const frame = requestAnimationFrame(() => filterPanelRef.current?.scrollIntoView({ block: "nearest" }));
    return () => cancelAnimationFrame(frame);
  }, [filtersOpen]);

  useEffect(() => {
    const media = window.matchMedia(MOBILE_PANEL_QUERY);
    const sync = () => {
      if (!media.matches) setFiltersOpen(false);
    };
    media.addEventListener("change", sync);
    return () => media.removeEventListener("change", sync);
  }, []);

  const allPlugins = data?.plugins ?? [];
  const meta = data?.meta ?? {};

  const filtered = useMemo(() => {
    const q = search.toLowerCase().trim();
    const matching = allPlugins.filter((p) => {
      if (tierFilter !== "all" && p.tier !== tierFilter) return false;
      if (categoryFilter !== "all" && p.category !== categoryFilter) return false;
      if (q) return (p._search || "").includes(q);
      return true;
    });
    return sortCatalogPlugins(matching, sort);
  }, [search, tierFilter, categoryFilter, sort, allPlugins]);

  const grouped = useMemo(() => {
    if (search.trim() || categoryFilter !== "all") return null;
    return groupCatalogPlugins(filtered);
  }, [filtered, search, categoryFilter]);

  const featured = useMemo(
    () =>
      pickFeatured(allPlugins, (p) => ({
        featured: p.featured,
        official: p.tier === "official",
        pictured: Boolean(p.image),
        addedAt: p.addedAt,
        stars: p.stars,
      })),
    [allPlugins],
  );

  const tierCounts = useMemo(() => {
    const counts = {};
    for (const p of allPlugins) counts[p.tier] = (counts[p.tier] || 0) + 1;
    return counts;
  }, [allPlugins]);

  const categoryCounts = useMemo(() => {
    const counts = {};
    for (const p of allPlugins) {
      if (tierFilter !== "all" && p.tier !== tierFilter) continue;
      const key = CATEGORY_CONFIG[p.category] ? p.category : "general";
      counts[key] = (counts[key] || 0) + 1;
    }
    return counts;
  }, [allPlugins, tierFilter]);

  const clearAll = () => {
    setSearch("");
    setTierFilter("all");
    setCategoryFilter("all");
    setFiltersOpen(false);
  };
  const pickCategory = (c) => {
    setCategoryFilter((cur) => (cur === c ? "all" : c));
    setSearch("");
  };

  const catalogEmpty = data !== null && allPlugins.length === 0;
  const visibleCategories = CATEGORY_ORDER.filter((c) => categoryCounts[c]);

  const renderCard = (plugin) => (
    <PluginCard
      key={`${plugin.tier}-${plugin.name}`}
      plugin={plugin}
      query={search}
      onPick={pickerMode ? pickPlugin : undefined}
      onCategoryClick={pickCategory}
    />
  );

  return (
    <div className={`hh-page pc-page${pickerMode ? " hh-picker" : ""}`}>
      <section className="hh-hero">
        <div className="hh-container">
          <p className="hermes-caption hh-hero-eyebrow">Hermes Agent</p>
          <h1 className="hermes-display hh-hero-title">Plugin Catalog</h1>
          <div className="hh-cross-nav" role="navigation" aria-label="Catalog pages">
            <a className="hermes-btn hermes-btn-ghost" href="/skills">Skills</a>
            <span className="hermes-btn hermes-btn-paper" aria-current="page">Plugins</span>
          </div>
          <p className="hh-hero-sub">
            Give Hermes new powers. Memory, voice, messaging, browsing, Desktop panes and more, built by the community.
            {loadError && <span className="hh-hero-error"> · failed to load catalog ({loadError})</span>}
          </p>
          {!catalogEmpty && (
            <p className="hh-hero-sub hh-hero-small">
              Built a plugin? <a href={SUBMIT_PLUGIN_URL}>Submit it to the catalog →</a>
            </p>
          )}
          {meta.generatedAt && !catalogEmpty && (
            <p className="hh-hero-sub hh-hero-small">
              {allPlugins.length.toLocaleString()} plugins across {Object.keys(categoryCounts).length} categories · updated{" "}
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
          <p className="hh-hero-sub hh-hero-small">
            <a href="https://portal.nousresearch.com/terms" target="_blank" rel="noopener noreferrer">Terms</a>
            {" • "}
            <a href="https://portal.nousresearch.com/privacy" target="_blank" rel="noopener noreferrer">Privacy Policy</a>
          </p>
        </div>
      </section>

      {!catalogEmpty && (
        <div className="hh-controls">
          <div className="hh-controls-top">
            <div className="hh-search-wrap">
              <svg className="hh-search-icon" viewBox="0 0 20 20" fill="currentColor" width="18" height="18" aria-hidden="true">
                <path fillRule="evenodd" d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z" clipRule="evenodd" />
              </svg>
              <input
                ref={searchRef}
                type="text"
                placeholder="Search plugins"
                aria-label="Search plugins"
                title='Tip: press "/" to jump here'
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="hh-search-input"
              />
              {search && (
                <button type="button" className="hh-clear-btn" onClick={() => setSearch("")} aria-label="Clear search">
                  ×
                </button>
              )}
            </div>

            <button
              type="button"
              className="hh-filter-toggle"
              aria-expanded={filtersOpen}
              aria-controls="plugin-directory-filters"
              onClick={() => setFiltersOpen((open) => !open)}
            >
              Filters
              {(tierFilter !== "all" || categoryFilter !== "all") && (
                <span className="hh-active-count">{Number(tierFilter !== "all") + Number(categoryFilter !== "all")}</span>
              )}
            </button>
            <label className="hh-compact-select">
              <span className="hermes-caption">Source</span>
              <select value={tierFilter} onChange={(e) => setTierFilter(e.target.value)}>
                {TIER_ORDER.map((tier) => (
                  <option key={tier} value={tier}>{tier === "all" ? "All sources" : TIER_CONFIG[tier]?.label || tier}</option>
                ))}
              </select>
            </label>
            <label className="hh-compact-select">
              <span className="hermes-caption">Category</span>
              <select value={categoryFilter} onChange={(e) => setCategoryFilter(e.target.value)}>
                <option value="all">All categories</option>
                {visibleCategories.map((c) => (
                  <option key={c} value={c}>{CATEGORY_CONFIG[c].label}</option>
                ))}
              </select>
            </label>
            <label className="hh-compact-select">
              <span className="hermes-caption">Sort</span>
              <select value={sort} onChange={(e) => setSort(e.target.value)}>
                {SORT_OPTIONS.map((opt) => (
                  <option key={opt.key} value={opt.key}>{opt.label}</option>
                ))}
              </select>
            </label>
          </div>

          <div
            id="plugin-directory-filters"
            ref={filterPanelRef}
            className={`pc-filter-panel${filtersOpen ? " pc-filter-panel-open" : ""}`}
          >
            <div className="pc-pill-row">
              {TIER_ORDER.map((tier) => {
                const active = tierFilter === tier;
                const conf = TIER_CONFIG[tier];
                return (
                  <button
                    type="button"
                    key={tier}
                    className={`hh-src-pill${active ? " hh-src-pill-active" : ""}`}
                    aria-pressed={active}
                    onClick={() => setTierFilter(tier)}
                    style={conf ? { "--hh-accent": conf.color } : undefined}
                  >
                    {conf && <span className="hh-dot" aria-hidden="true" />}
                    {tier === "all" ? "All" : conf?.label || tier}
                    <span className="hh-src-count">{(tier === "all" ? allPlugins.length : tierCounts[tier] || 0).toLocaleString()}</span>
                  </button>
                );
              })}
            </div>

            <div className="pc-pill-row" role="radiogroup" aria-label="Sort plugins">
              <span className="hermes-caption pc-sort-label">Sort</span>
              {SORT_OPTIONS.map((opt) => {
                const active = sort === opt.key;
                return (
                  <button
                    type="button"
                    key={opt.key}
                    className={`hh-src-pill${active ? " hh-src-pill-active" : ""}`}
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

            <div className="pc-pill-row" role="tablist" aria-label="Plugin categories">
              <button
                type="button"
                className={`hh-src-pill pc-cat-pill${categoryFilter === "all" ? " hh-src-pill-active" : ""}`}
                onClick={() => setCategoryFilter("all")}
                role="tab"
                aria-selected={categoryFilter === "all"}
              >
                All categories
              </button>
              {visibleCategories.map((c) => {
                const conf = CATEGORY_CONFIG[c];
                const active = categoryFilter === c;
                return (
                  <button
                    type="button"
                    key={c}
                    className={`hh-src-pill pc-cat-pill${active ? " hh-src-pill-active" : ""}`}
                    onClick={() => pickCategory(c)}
                    role="tab"
                    aria-selected={active}
                    title={conf.blurb}
                  >
                    <span aria-hidden="true">{conf.icon}</span> {conf.label}
                    <span className="hh-src-count">{categoryCounts[c].toLocaleString()}</span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      )}

      <div className="pc-main">
        {!data && !loadError ? (
          <div className="hh-empty" aria-live="polite">
            <div className="hh-spinner" />
            <h3 className="hermes-display hh-empty-title">Loading the catalog…</h3>
          </div>
        ) : loadError ? (
          <div className="hh-empty" role="alert">
            <h3 className="hermes-display hh-empty-title">Couldn't load the catalog</h3>
            <p className="hh-empty-desc">{loadError}</p>
          </div>
        ) : catalogEmpty ? (
          <div className="hh-empty">
            <h3 className="hermes-display hh-empty-title">The catalog is just getting started</h3>
            <p className="hh-empty-desc">
              The plugin catalog is a curated, human-reviewed list of Hermes plugins — each entry pinned to an exact commit.
              Want yours listed? Submissions are open.
            </p>
            <div className="pc-empty-actions">
              <a className="hermes-btn hh-btn-solid" href={SUBMIT_PLUGIN_URL}>How to submit a plugin</a>
              <a className="hermes-btn hh-btn-outline" href={PLUGIN_CATALOG_DOCS_URL}>Read the catalog docs</a>
            </div>
          </div>
        ) : filtered.length > 0 && grouped ? (
          <>
            {featured && tierFilter === "all" && !pickerMode && (
              <CatalogFeatured
                author={featured.maintainer}
                cta="View plugin"
                description={cardSummary(featured.description || "")}
                href={pluginPageUrl(featured.name)}
                label="Featured plugin"
                tags={[tierOf(featured).label, categoryOf(featured).label, featured.version && `v${featured.version}`]}
                title={featured.name}
              />
            )}
            {grouped.map(([cat, plugins]) => {
              const conf = CATEGORY_CONFIG[cat];
              return (
                <section key={cat} className="pc-category-section" aria-labelledby={`cat-${cat}`}>
                  <div className="pc-category-header">
                    <h2 id={`cat-${cat}`} className="hermes-display pc-category-title">
                      <span aria-hidden="true" className="pc-category-icon">{conf.icon}</span> {conf.label}
                      <span className="hh-src-count pc-category-count">{plugins.length.toLocaleString()}</span>
                    </h2>
                    <p className="pc-category-blurb">{conf.blurb}</p>
                    <button type="button" className="hh-text-btn pc-category-view-all" onClick={() => pickCategory(cat)}>
                      View only {conf.label} →
                    </button>
                  </div>
                  <div className="hh-grid pc-grid">{plugins.map(renderCard)}</div>
                </section>
              );
            })}
          </>
        ) : filtered.length > 0 ? (
          <>
            <div className="hh-filter-summary" role="status">
              {categoryFilter !== "all" && CATEGORY_CONFIG[categoryFilter] && (
                <span className="pc-results-text">
                  <span aria-hidden="true">{CATEGORY_CONFIG[categoryFilter].icon}</span>{" "}
                  <strong>{CATEGORY_CONFIG[categoryFilter].label}</strong>
                  <span className="pc-results-muted"> · {CATEGORY_CONFIG[categoryFilter].blurb}</span>
                </span>
              )}
              {search.trim() && (
                <span className="pc-results-text">
                  Results for <strong>“{search.trim()}”</strong>
                </span>
              )}
              <span className="hh-filter-count">
                {filtered.length.toLocaleString()} plugin{filtered.length === 1 ? "" : "s"}
              </span>
              <button type="button" className="hh-text-btn" onClick={clearAll}>
                Clear filters
              </button>
            </div>
            <div className="hh-grid pc-grid">{filtered.map(renderCard)}</div>
          </>
        ) : (
          <div className="hh-empty">
            <h3 className="hermes-display hh-empty-title">No plugins found</h3>
            <p className="hh-empty-desc">Try a different search term or clear your filters.</p>
            <button type="button" className="hermes-btn hh-btn-solid" onClick={clearAll}>
              Reset all filters
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
