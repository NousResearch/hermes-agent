// Port of hermes-agent website/src/pages/skills/index.tsx (Docusaurus) with
// apps/shared/src/catalog-featured.ts, apps/shared/src/catalog-install.ts and
// website/src/components/CatalogFeatured inlined: Mintlify snippets cannot
// import npm packages, JSON, or files outside the docs project.

// The catalog is rebuilt twice daily by website/scripts/extract-skills.py and
// is too large to ship with the docs (~65 MB raw, 100k+ entries), so it is
// fetched at runtime from the same public origin the source page uses.
export const SKILLS_URL = "https://nousresearch.github.io/hermes-agent/docs/api/skills.json";
export const META_URL = "https://nousresearch.github.io/hermes-agent/docs/api/skills-meta.json";
export const SOURCE_SITE_IMAGE_PREFIX = "https://hermes-agent.nousresearch.com/docs/img/catalog/";
export const PLUGINS_URL = "https://hermes-agent.nousresearch.com/docs/plugins";
export const CATALOG_STALL_TIMEOUT_MS = 45000;
export const PAGE_SIZE = 60;
export const MOBILE_FILTER_QUERY = "(max-width: 600px)";

export const CATEGORY_ICONS = {
  apple: "\u{1F34E}",
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

export const SOURCE_CONFIG = {
  "built-in": { label: "Built-in", color: "#4ade80", icon: "\u{2713}" },
  optional: { label: "Optional", color: "#fbbf24", icon: "\u{2B50}" },
  Anthropic: { label: "Anthropic", color: "#d4845a", icon: "\u{25C6}" },
  LobeHub: { label: "LobeHub", color: "#60a5fa", icon: "\u{25CB}" },
  "skills.sh": { label: "skills.sh", color: "#34d399", icon: "\u{2734}" },
  ClawHub: { label: "ClawHub", color: "#f472b6", icon: "\u{2726}" },
  "browse.sh": { label: "browse.sh", color: "#22d3ee", icon: "\u{29BF}" },
  OpenAI: { label: "OpenAI", color: "#10b981", icon: "\u{2737}" },
  HuggingFace: { label: "HuggingFace", color: "#fbbf24", icon: "\u{1F917}" },
  NVIDIA: { label: "NVIDIA", color: "#76b900", icon: "\u{25B6}" },
  VoltAgent: { label: "VoltAgent", color: "#facc15", icon: "\u{26A1}" },
  GitHub: { label: "GitHub", color: "#94a3b8", icon: "\u{2756}" },
  "Well-Known": { label: "Well-Known", color: "#818cf8", icon: "\u{2756}" },
  gstack: { label: "gstack", color: "#fb923c", icon: "\u{2756}" },
  MiniMax: { label: "MiniMax", color: "#f87171", icon: "\u{2756}" },
  "Weights & Biases": { label: "Weights & Biases", color: "#facc15", icon: "\u{2756}" },
  Qdrant: { label: "Qdrant", color: "#f43f5e", icon: "\u{2756}" },
  Pinecone: { label: "Pinecone", color: "#a3e635", icon: "\u{2756}" },
  Notion: { label: "Notion", color: "#e5e7eb", icon: "\u{25A0}" },
  Airtable: { label: "Airtable", color: "#fcb400", icon: "\u{25A6}" },
  Shopify: { label: "Shopify", color: "#95bf47", icon: "\u{25B2}" },
};

export const SOURCE_ORDER = [
  "all", "built-in", "optional", "Anthropic", "OpenAI", "HuggingFace", "NVIDIA",
  "skills.sh", "ClawHub", "browse.sh", "LobeHub", "VoltAgent", "Well-Known", "GitHub",
  "gstack", "MiniMax", "Weights & Biases", "Qdrant", "Pinecone", "Notion", "Airtable", "Shopify",
];

export const OFFICIAL_SKILL_SOURCES = ["built-in", "bundled", "optional", "official"];

export const isOfficialSkillSource = (source) => OFFICIAL_SKILL_SOURCES.includes(source ?? "");

export const pickFeatured = (entries, read, now = Date.now()) => {
  const WEEK_MS = 7 * 24 * 60 * 60 * 1000;
  const FRESH_MS = 30 * 24 * 60 * 60 * 1000;
  const rank = (v) => (typeof v.featured === "number" && v.featured > 0 ? v.featured : 0);
  const added = (v) => Date.parse(v.addedAt ?? "") || 0;
  const rows = entries.map((entry) => ({ entry, value: read(entry) }));
  const curated = rows.filter((r) => rank(r.value)).sort((a, b) => rank(a.value) - rank(b.value));
  if (curated.length) {
    const start = Math.floor(now / WEEK_MS) % curated.length;
    return [...curated.slice(start), ...curated.slice(0, start)][0].entry;
  }
  const official = rows.filter((r) => r.value.official);
  const fresh = official
    .filter((r) => r.value.pictured && added(r.value) > now - FRESH_MS)
    .sort((a, b) => added(b.value) - added(a.value));
  const pick =
    fresh[0] ??
    official.sort(
      (a, b) => Number(b.value.pictured) - Number(a.value.pictured) || (b.value.stars ?? 0) - (a.value.stars ?? 0),
    )[0];
  return pick?.entry;
};

export const skillCatalogInstallIdentifier = (skill) => {
  if (skill.installIdentifier) return skill.installIdentifier;
  const identifier = skill.identifier;
  if (!identifier) {
    return skill.source === "built-in" ? null : skill.source === "optional" ? `official/${skill.name}` : skill.name;
  }
  return skill.source.toLowerCase() === "clawhub" && !identifier.startsWith("clawhub/")
    ? `clawhub/${identifier}`
    : identifier;
};

export const skillCatalogInstallUrl = (skill) => {
  const identifier = skillCatalogInstallIdentifier(skill);
  return identifier ? `hermes://skill/install?${new URLSearchParams({ identifier })}` : null;
};

export const installCommand = (skill) =>
  skill.installCmd || `hermes skills install ${skillCatalogInstallIdentifier(skill) || skill.name}`;

export const skillDocsHref = (skill) => (skill.docsPath ? `/user-guide/skills/${skill.docsPath}` : null);

export const localImage = (src) =>
  src && src.startsWith(SOURCE_SITE_IMAGE_PREFIX) ? `/img/catalog/${src.slice(SOURCE_SITE_IMAGE_PREFIX.length)}` : src;

export const formatRelativeTime = (iso) => {
  if (!iso) return null;
  const then = new Date(iso).getTime();
  if (!Number.isFinite(then)) return null;
  const diffMs = Date.now() - then;
  if (diffMs < 0) return "just now";
  const mins = Math.floor(diffMs / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins} minute${mins === 1 ? "" : "s"} ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours} hour${hours === 1 ? "" : "s"} ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days} day${days === 1 ? "" : "s"} ago`;
  const months = Math.floor(days / 30);
  return `${months} month${months === 1 ? "" : "s"} ago`;
};

export const buildSearchHaystack = (s) =>
  [s.name, s.description, s.overview, s.categoryLabel, s.author, ...(s.tags || [])]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();

export const highlightMatch = (text, query) => {
  if (!query || !text) return text;
  const idx = text.toLowerCase().indexOf(query.toLowerCase());
  if (idx === -1) return text;
  return (
    <>
      {text.slice(0, idx)}
      <mark className="hh-highlight">{text.slice(idx, idx + query.length)}</mark>
      {text.slice(idx + query.length)}
    </>
  );
};

export const CopyButton = ({ text }) => {
  const [copied, setCopied] = useState(false);
  const onCopy = (e) => {
    e.stopPropagation();
    navigator.clipboard?.writeText(text).then(
      () => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      },
      () => {},
    );
  };
  return (
    <button type="button" className="hh-copy-btn" onClick={onCopy} title="Copy install command" aria-label="Copy install command">
      {copied ? "Copied" : "Copy"}
    </button>
  );
};

export const CatalogFeatured = ({ label, author, title, description, tags, href, cta }) => {
  const external = /^https?:\/\//.test(href);
  const chips = tags.filter(Boolean);
  return (
    <section aria-label={label} className="hh-featured">
      <p className="hh-featured-label hermes-caption">{label}</p>
      <div className="hh-featured-card">
        <div className="hh-featured-copy">
          {author && <p className="hh-featured-author">{author}</p>}
          <h2 className="hh-featured-title hermes-display">{title}</h2>
          {description && <p className="hh-featured-desc">{description}</p>}
          {chips.length > 0 && (
            <div className="hh-tag-row">
              {chips.map((tag) => (
                <span className="hh-chip" key={tag}>{tag}</span>
              ))}
            </div>
          )}
        </div>
        <a
          className="hermes-btn hh-btn-solid"
          href={href}
          {...(external ? { target: "_blank", rel: "noopener noreferrer" } : {})}
        >
          {cta} {external ? "↗" : "→"}
        </a>
      </div>
    </section>
  );
};

export const SkillCard = ({ skill, query, expanded, onToggle, onCategoryClick, onTagClick, onPick }) => {
  const src = SOURCE_CONFIG[skill.source] || SOURCE_CONFIG.optional;
  const icon = CATEGORY_ICONS[skill.category] || "\u{1F4E6}";
  const installUrl = skillCatalogInstallUrl(skill);
  const docsHref = skillDocsHref(skill);
  const cmd = installCommand(skill);
  return (
    <div
      className={`hh-card${expanded ? " hh-card-expanded" : ""}`}
      onClick={onToggle}
      style={{ "--hh-accent": src.color }}
    >
      {skill.image && <img src={localImage(skill.image)} alt="" loading="lazy" decoding="async" className="hh-card-image" />}
      <div className="hh-card-inner">
        <div className="hh-card-top">
          <span className="hh-card-icon" aria-hidden="true">{icon}</span>
          <div className="hh-card-title-group">
            <h3 className="hh-card-title">{highlightMatch(skill.name, query)}</h3>
            <span className="hh-source-pill">
              <span className="hh-dot" aria-hidden="true" />
              {src.label}
            </span>
          </div>
        </div>

        <p className={`hh-card-desc${expanded ? " hh-card-desc-full" : ""}`}>
          {highlightMatch(skill.description || "No description available.", query)}
        </p>

        <div className="hh-card-meta">
          <button
            type="button"
            className="hh-chip hh-chip-button"
            onClick={(e) => {
              e.stopPropagation();
              onCategoryClick(skill.category);
            }}
            title={`Filter by ${skill.categoryLabel}`}
          >
            {skill.categoryLabel || skill.category}
          </button>
          {skill.platforms?.map((p) => (
            <span key={p} className="hh-chip hh-chip-muted">
              {p === "macos" ? "macOS" : p === "linux" ? "Linux" : p}
            </span>
          ))}
        </div>

        {!onPick && installUrl && (
          <a className="hh-pick-btn" href={installUrl} onClick={(e) => e.stopPropagation()}>
            Install in Hermes
          </a>
        )}

        {expanded && (
          <div className="hh-card-detail">
            {skill.overview && (
              <div className="hh-detail-block">
                <span className="hh-detail-label hermes-caption">Overview</span>
                <p className="hh-overview">{skill.overview}</p>
              </div>
            )}
            {skill.envVars?.length || skill.commands?.length ? (
              <div className="hh-detail-block">
                <span className="hh-detail-label hermes-caption">Prerequisites</span>
                {skill.envVars?.length ? (
                  <div className="hh-prereq-row">
                    <span className="hh-prereq-kind hermes-caption">env</span>
                    <span className="hh-prereq-list">
                      {skill.envVars.map((v) => <code key={v}>{v}</code>)}
                    </span>
                  </div>
                ) : null}
                {skill.commands?.length ? (
                  <div className="hh-prereq-row">
                    <span className="hh-prereq-kind hermes-caption">cmd</span>
                    <span className="hh-prereq-list">
                      {skill.commands.map((c) => <code key={c}>{c}</code>)}
                    </span>
                  </div>
                ) : null}
              </div>
            ) : null}
            {skill.tags?.length > 0 && (
              <div className="hh-tag-row">
                {skill.tags.map((tag) => (
                  <button
                    type="button"
                    key={tag}
                    className="hh-chip hh-chip-button"
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
            {[
              ["Author", skill.author],
              ["Version", skill.version],
              ["License", skill.license],
            ]
              .filter(([, v]) => v)
              .map(([k, v]) => (
                <div className="hh-author-row" key={k}>
                  <span className="hh-author-label hermes-caption">{k}</span>
                  <span>{v}</span>
                </div>
              ))}
            <div className="hh-install-hint">
              <code>{cmd}</code>
              <CopyButton text={cmd} />
            </div>
            {onPick ? (
              <button
                type="button"
                className="hh-pick-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  onPick(skill);
                }}
              >
                + Add to this Agent
              </button>
            ) : null}
            <div className="hh-card-links">
              {docsHref ? (
                <a className="hh-docs-link" href={docsHref} onClick={(e) => e.stopPropagation()}>
                  View full documentation →
                </a>
              ) : skill.sourceUrl ? (
                <a
                  className="hh-docs-link"
                  href={skill.sourceUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={(e) => e.stopPropagation()}
                >
                  View source ↗
                </a>
              ) : null}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export const SkillsHub = () => {
  // ?embed=picker: Hermes desktop iframes this page as a skill picker. Chrome is
  // hidden and each card posts the pick to the parent; the host installs it.
  const [pickerMode, setPickerMode] = useState(false);
  useEffect(() => {
    setPickerMode(new URLSearchParams(window.location.search).get("embed") === "picker");
  }, []);

  const pickSkill = useCallback((skill) => {
    if (window.parent === window) return;
    window.parent.postMessage(
      {
        type: "hermes-skill-pick",
        name: skill.name,
        identifier: skillCatalogInstallIdentifier(skill) || skill.name,
        installCmd: installCommand(skill),
        source: skill.source,
      },
      "*",
    );
  }, []);

  const [data, setData] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [reloadKey, setReloadKey] = useState(0);
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [sourceFilter, setSourceFilter] = useState("all");
  const [categoryFilter, setCategoryFilter] = useState("all");
  const [expandedCard, setExpandedCard] = useState(null);
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [mobileFiltersActive, setMobileFiltersActive] = useState(false);
  const searchRef = useRef(null);
  const gridRef = useRef(null);
  const filterToggleRef = useRef(null);
  const filterDialogRef = useRef(null);
  const filterCloseRef = useRef(null);

  // Read the body as a stream and abort only when no bytes arrive for the stall
  // window, so slow-but-alive downloads of the large catalog are never cut off.
  useEffect(() => {
    const ctrl = new AbortController();
    let stallTimer = null;
    let cancelled = false;
    const armStallTimer = () => {
      if (stallTimer) clearTimeout(stallTimer);
      stallTimer = setTimeout(() => ctrl.abort(), CATALOG_STALL_TIMEOUT_MS);
    };
    const readJson = async (url) => {
      const res = await fetch(url, { signal: ctrl.signal });
      if (!res.ok) throw new Error(`HTTP ${res.status} from ${url}`);
      if (!res.body) return res.json();
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let text = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        armStallTimer();
        text += decoder.decode(value, { stream: true });
      }
      return JSON.parse(text);
    };
    (async () => {
      try {
        armStallTimer();
        const [sk, mt] = await Promise.all([readJson(SKILLS_URL), readJson(META_URL).catch(() => ({}))]);
        if (cancelled) return;
        const skillsArr = Array.isArray(sk) ? sk : [];
        for (const s of skillsArr) s._search = buildSearchHaystack(s);
        setData({ skills: skillsArr, meta: mt || {} });
      } catch (err) {
        if (cancelled) return;
        if (ctrl.signal.aborted) {
          setLoadError(`catalog fetch stalled — no data received for ${CATALOG_STALL_TIMEOUT_MS / 1000}s`);
          return;
        }
        setLoadError(err instanceof Error ? err.message : String(err));
      } finally {
        if (stallTimer) clearTimeout(stallTimer);
      }
    })();
    return () => {
      cancelled = true;
      ctrl.abort();
      if (stallTimer) clearTimeout(stallTimer);
    };
  }, [reloadKey]);

  const retryCatalogLoad = () => {
    setLoadError(null);
    setReloadKey((k) => k + 1);
  };

  useEffect(() => {
    const t = setTimeout(() => setDebouncedSearch(search), 150);
    return () => clearTimeout(t);
  }, [search]);

  const allSkills = data?.skills ?? [];
  const meta = data?.meta ?? {};

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
        setExpandedCard(null);
        setSidebarOpen(false);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  useEffect(() => {
    const media = window.matchMedia(MOBILE_FILTER_QUERY);
    const sync = () => {
      setMobileFiltersActive(media.matches);
      if (!media.matches) setSidebarOpen(false);
    };
    sync();
    media.addEventListener("change", sync);
    return () => media.removeEventListener("change", sync);
  }, []);

  useEffect(() => {
    if (!sidebarOpen || !mobileFiltersActive) return;
    const dialog = filterDialogRef.current;
    const previouslyFocused =
      document.activeElement instanceof HTMLElement ? document.activeElement : filterToggleRef.current;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    filterCloseRef.current?.focus();
    const trapFocus = (e) => {
      if (!dialog) return;
      if (e.key === "Escape") {
        e.preventDefault();
        e.stopPropagation();
        setSidebarOpen(false);
        return;
      }
      if (e.key !== "Tab") {
        if (!dialog.contains(e.target)) {
          e.preventDefault();
          e.stopPropagation();
        }
        return;
      }
      const focusable = Array.from(
        dialog.querySelectorAll(
          "button:not([disabled]), select:not([disabled]), input:not([disabled]), [href], [tabindex]:not([tabindex='-1'])",
        ),
      ).filter((el) => el.offsetParent !== null);
      if (focusable.length === 0) {
        e.preventDefault();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", trapFocus, true);
    return () => {
      document.removeEventListener("keydown", trapFocus, true);
      document.body.style.overflow = previousOverflow;
      previouslyFocused?.focus();
    };
  }, [sidebarOpen, mobileFiltersActive]);

  const sourceCounts = useMemo(() => {
    const counts = {};
    for (const s of allSkills) counts[s.source] = (counts[s.source] || 0) + 1;
    return counts;
  }, [allSkills]);

  const sources = useMemo(() => SOURCE_ORDER.filter((s) => s === "all" || sourceCounts[s]), [sourceCounts]);

  const categoryEntries = useMemo(() => {
    const pool = sourceFilter === "all" ? allSkills : allSkills.filter((s) => s.source === sourceFilter);
    const map = new Map();
    for (const s of pool) {
      const key = s.category || "uncategorized";
      const existing = map.get(key);
      if (existing) existing.count++;
      else map.set(key, { label: s.categoryLabel || s.category || "Uncategorized", count: 1 });
    }
    return Array.from(map.entries())
      .sort((a, b) => b[1].count - a[1].count)
      .map(([key, { label, count }]) => ({ key, label, count }));
  }, [sourceFilter, allSkills]);

  const filtered = useMemo(() => {
    const q = debouncedSearch.toLowerCase().trim();
    return allSkills.filter((s) => {
      if (sourceFilter !== "all" && s.source !== sourceFilter) return false;
      if (categoryFilter !== "all" && s.category !== categoryFilter) return false;
      if (q) return (s._search || "").includes(q);
      return true;
    });
  }, [debouncedSearch, sourceFilter, categoryFilter, allSkills]);

  useEffect(() => {
    setVisibleCount(PAGE_SIZE);
    setExpandedCard(null);
  }, [debouncedSearch, sourceFilter, categoryFilter]);

  const visible = filtered.slice(0, visibleCount);
  const hasMore = visibleCount < filtered.length;

  const featured = useMemo(
    () =>
      pickFeatured(allSkills, (s) => ({
        featured: s.featured,
        official: isOfficialSkillSource(s.source),
        pictured: Boolean(s.image),
      })),
    [allSkills],
  );
  const browsing = !search && sourceFilter === "all" && categoryFilter === "all" && !pickerMode;

  const stats = useMemo(() => {
    const builtIn = sourceCounts["built-in"] || 0;
    const optional = sourceCounts.optional || 0;
    return [
      { value: builtIn, label: "Built-in" },
      { value: optional, label: "Optional" },
      { value: allSkills.length - builtIn - optional, label: "Community" },
      { value: new Set(allSkills.map((s) => s.category)).size, label: "Categories" },
    ];
  }, [allSkills, sourceCounts]);

  const handleSourceChange = (src) => {
    setSourceFilter(src);
    setCategoryFilter("all");
  };
  const handleCategoryClick = (cat) => {
    setCategoryFilter(cat);
    gridRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    setSidebarOpen(false);
  };
  const handleTagClick = (tag) => {
    setSearch(tag);
    searchRef.current?.focus();
  };
  const clearAll = () => {
    setSearch("");
    setSourceFilter("all");
    setCategoryFilter("all");
  };

  const sourceOptions = sources.map((src) => (
    <option key={src} value={src}>
      {src === "all" ? "All sources" : SOURCE_CONFIG[src]?.label || src}
    </option>
  ));
  const refreshedAt = meta.indexGeneratedAt || meta.extractedAt;

  return (
    <div className={`hh-page${pickerMode ? " hh-picker" : ""}`}>
      <section className="hh-hero">
        <div className="hh-container">
          <p className="hermes-caption hh-hero-eyebrow">Hermes Agent</p>
          <h1 className="hermes-display hh-hero-title">Skills Hub</h1>
          <div className="hh-cross-nav" role="navigation" aria-label="Catalog pages">
            <span className="hermes-btn hermes-btn-paper" aria-current="page">Skills</span>
            <a className="hermes-btn hermes-btn-ghost" href={PLUGINS_URL}>Plugins</a>
          </div>
          <p className="hh-hero-sub">
            Discover, search, and install from{" "}
            <strong>{data ? allSkills.length.toLocaleString() : "…"}</strong> skills across {sources.length - 1} registries
            {loadError && <span className="hh-hero-error"> · failed to load catalog ({loadError})</span>}
          </p>
          {refreshedAt && (
            <p className="hh-hero-sub hh-hero-small">
              Catalog refreshed <span title={refreshedAt}>{formatRelativeTime(refreshedAt) || "recently"}</span> · auto-rebuilt twice daily
            </p>
          )}
          <p className="hh-hero-sub hh-hero-small">
            <a href="https://portal.nousresearch.com/terms" target="_blank" rel="noopener noreferrer">Terms</a>
            {" • "}
            <a href="https://portal.nousresearch.com/privacy" target="_blank" rel="noopener noreferrer">Privacy Policy</a>
          </p>
          <div className="hh-stats">
            {stats.map((s) => (
              <div className="hh-stat" key={s.label}>
                <span className="hermes-display hh-stat-value">{s.value.toLocaleString()}</span>
                <span className="hermes-caption hh-stat-label">{s.label}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      <div className="hh-controls">
        <div className="hh-controls-top">
          <div className="hh-search-wrap">
            <svg className="hh-search-icon" viewBox="0 0 20 20" fill="currentColor" width="18" height="18" aria-hidden="true">
              <path fillRule="evenodd" d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z" clipRule="evenodd" />
            </svg>
            <input
              ref={searchRef}
              type="text"
              placeholder="Search skills"
              aria-label="Search skills"
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
            ref={filterToggleRef}
            type="button"
            className="hh-filter-toggle"
            aria-expanded={sidebarOpen}
            aria-controls="skills-directory-filters"
            onClick={() => setSidebarOpen((open) => !open)}
          >
            Filters
            {(sourceFilter !== "all" || categoryFilter !== "all") && (
              <span className="hh-active-count">{Number(sourceFilter !== "all") + Number(categoryFilter !== "all")}</span>
            )}
          </button>

          <label className="hh-compact-select">
            <span className="hermes-caption">Source</span>
            <select value={sourceFilter} onChange={(e) => handleSourceChange(e.target.value)}>
              {sourceOptions}
            </select>
          </label>

          <label className="hh-compact-select">
            <span className="hermes-caption">Category</span>
            <select value={categoryFilter} onChange={(e) => setCategoryFilter(e.target.value)}>
              <option value="all">All categories</option>
              {categoryEntries.map((cat) => (
                <option key={cat.key} value={cat.key}>{cat.label}</option>
              ))}
            </select>
          </label>
        </div>

        <div className="hh-source-pills">
          {sources.map((src) => {
            const active = sourceFilter === src;
            const conf = SOURCE_CONFIG[src];
            const count = src === "all" ? allSkills.length : sourceCounts[src] || 0;
            return (
              <button
                type="button"
                key={src}
                className={`hh-src-pill${active ? " hh-src-pill-active" : ""}`}
                aria-pressed={active}
                onClick={() => handleSourceChange(src)}
                style={conf ? { "--hh-accent": conf.color } : undefined}
              >
                {conf && <span className="hh-dot" aria-hidden="true" />}
                {src === "all" ? "All" : conf?.label || src}
                <span className="hh-src-count">{count.toLocaleString()}</span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="hh-layout">
        <aside
          id="skills-directory-filters"
          ref={filterDialogRef}
          role={mobileFiltersActive && sidebarOpen ? "dialog" : undefined}
          aria-modal={mobileFiltersActive && sidebarOpen ? true : undefined}
          aria-labelledby="skills-directory-filters-title"
          className={`hh-sidebar${sidebarOpen ? " hh-sidebar-open" : ""}`}
        >
          <div className="hh-sidebar-header">
            <h2 id="skills-directory-filters-title" className="hermes-caption hh-sidebar-title">Categories</h2>
            {categoryFilter !== "all" && (
              <button type="button" className="hh-text-btn" onClick={() => setCategoryFilter("all")}>
                Clear
              </button>
            )}
            <button
              ref={filterCloseRef}
              type="button"
              className="hh-sidebar-close"
              aria-label="Close skill filters"
              onClick={() => setSidebarOpen(false)}
            >
              <span aria-hidden="true">×</span>
            </button>
          </div>
          <label className="hh-sidebar-source">
            <span className="hermes-caption">Source</span>
            <select value={sourceFilter} onChange={(e) => handleSourceChange(e.target.value)}>
              {sourceOptions}
            </select>
          </label>
          <div className="hh-cat-list" role="navigation" aria-label="Skill categories">
            <button
              type="button"
              className={`hh-cat-item${categoryFilter === "all" ? " hh-cat-item-active" : ""}`}
              onClick={() => {
                setCategoryFilter("all");
                setSidebarOpen(false);
              }}
            >
              <span className="hh-cat-icon" aria-hidden="true">{"\u{1F4CB}"}</span>
              <span className="hh-cat-label">All Skills</span>
              <span className="hh-cat-count">{filtered.length.toLocaleString()}</span>
            </button>
            {categoryEntries.map((cat) => (
              <button
                type="button"
                key={cat.key}
                className={`hh-cat-item${categoryFilter === cat.key ? " hh-cat-item-active" : ""}`}
                onClick={() => handleCategoryClick(cat.key)}
              >
                <span className="hh-cat-icon" aria-hidden="true">{CATEGORY_ICONS[cat.key] || "\u{1F4E6}"}</span>
                <span className="hh-cat-label">{cat.label}</span>
                <span className="hh-cat-count">{cat.count.toLocaleString()}</span>
              </button>
            ))}
          </div>
        </aside>

        <div className="hh-main" ref={gridRef}>
          {(search || sourceFilter !== "all" || categoryFilter !== "all") && (
            <div className="hh-filter-summary">
              <span className="hh-filter-count">
                {filtered.length.toLocaleString()} result{filtered.length !== 1 ? "s" : ""}
              </span>
              {search && (
                <span className="hh-chip">
                  “{search}”
                  <button type="button" onClick={() => setSearch("")} aria-label="Remove search filter">×</button>
                </span>
              )}
              {sourceFilter !== "all" && (
                <span className="hh-chip">
                  {SOURCE_CONFIG[sourceFilter]?.label || sourceFilter}
                  <button type="button" onClick={() => setSourceFilter("all")} aria-label="Remove source filter">×</button>
                </span>
              )}
              {categoryFilter !== "all" && (
                <span className="hh-chip">
                  {categoryEntries.find((c) => c.key === categoryFilter)?.label || categoryFilter}
                  <button type="button" onClick={() => setCategoryFilter("all")} aria-label="Remove category filter">×</button>
                </span>
              )}
              <button type="button" className="hh-text-btn" onClick={clearAll}>
                Clear all
              </button>
            </div>
          )}

          {!data && !loadError ? (
            <div className="hh-empty" aria-live="polite">
              <div className="hh-spinner" />
              <h3 className="hermes-display hh-empty-title">Loading the catalog…</h3>
              <p className="hh-empty-desc">Fetching 88k+ skills across every registry. One moment.</p>
            </div>
          ) : loadError ? (
            <div className="hh-empty" role="alert">
              <h3 className="hermes-display hh-empty-title">Couldn't load the catalog</h3>
              <p className="hh-empty-desc">
                {loadError}. The catalog is served from a public CDN — restricted networks may need a working proxy.
                Check your connection and retry.
              </p>
              <button type="button" className="hermes-btn hh-btn-solid" onClick={retryCatalogLoad}>
                Retry
              </button>
            </div>
          ) : visible.length > 0 ? (
            <>
              {featured && browsing && (
                <CatalogFeatured
                  author={featured.author}
                  cta="View skill"
                  description={featured.description}
                  href={skillDocsHref(featured) || featured.sourceUrl || "#"}
                  label="Featured skill"
                  tags={[
                    SOURCE_CONFIG[featured.source]?.label || featured.source,
                    featured.categoryLabel,
                    featured.version && `v${featured.version}`,
                  ]}
                  title={featured.name}
                />
              )}
              <div className="hh-grid">
                {visible.map((skill, i) => {
                  const key = `${skill.source}-${skill.name}-${i}`;
                  return (
                    <SkillCard
                      key={key}
                      skill={skill}
                      query={search}
                      expanded={expandedCard === key}
                      onToggle={() => setExpandedCard(expandedCard === key ? null : key)}
                      onCategoryClick={handleCategoryClick}
                      onTagClick={handleTagClick}
                      onPick={pickerMode ? pickSkill : undefined}
                    />
                  );
                })}
              </div>
              {hasMore && (
                <div className="hh-load-more">
                  <button type="button" className="hermes-btn hh-btn-outline" onClick={() => setVisibleCount((v) => v + PAGE_SIZE)}>
                    Show more ({(filtered.length - visibleCount).toLocaleString()} remaining)
                  </button>
                </div>
              )}
            </>
          ) : (
            <div className="hh-empty">
              <h3 className="hermes-display hh-empty-title">No skills found</h3>
              <p className="hh-empty-desc">Try a different search term or clear your filters.</p>
              <button type="button" className="hermes-btn hh-btn-solid" onClick={clearAll}>
                Reset all filters
              </button>
            </div>
          )}
        </div>
      </div>

      {sidebarOpen && <div className="hh-backdrop" onClick={() => setSidebarOpen(false)} />}
    </div>
  );
};
