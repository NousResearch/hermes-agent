import { useEffect, useLayoutEffect, useState, useMemo, useCallback } from "react";
import {
  Package,
  Search,
  Wrench,
  X,
  Cpu,
  Globe,
  Shield,
  ShieldCheck,
  ShieldAlert,
  ShieldQuestion,
  Eye,
  Paintbrush,
  Brain,
  Blocks,
  Code,
  Zap,
  Filter,
  Download,
  RefreshCw,
  FileText,
  ExternalLink,
  CheckCircle2,
  AlertTriangle,
  Sparkles,
  Loader2,
} from "lucide-react";
import { api } from "@/lib/api";
import type {
  SkillInfo,
  ToolsetInfo,
  SkillHubResult,
  SkillHubSource,
  SkillHubInstalledEntry,
  SkillHubPreview,
  SkillHubScan,
} from "@/lib/api";
import { ToolsetConfigDrawer } from "@/components/ToolsetConfigDrawer";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { ListItem } from "@nous-research/ui/ui/components/list-item";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Switch } from "@nous-research/ui/ui/components/switch";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@nous-research/ui/ui/components/dialog";
import { cn } from "@/lib/utils";
import { Input } from "@nous-research/ui/ui/components/input";
import { useI18n } from "@/i18n";
import { usePageHeader } from "@/contexts/usePageHeader";
import { PluginSlot } from "@/plugins";

/* ------------------------------------------------------------------ */
/*  Types & helpers                                                    */
/* ------------------------------------------------------------------ */

const CATEGORY_LABELS: Record<string, string> = {
  mlops: "MLOps",
  "mlops/cloud": "MLOps / Cloud",
  "mlops/evaluation": "MLOps / Evaluation",
  "mlops/inference": "MLOps / Inference",
  "mlops/models": "MLOps / Models",
  "mlops/training": "MLOps / Training",
  "mlops/vector-databases": "MLOps / Vector DBs",
  mcp: "MCP",
  "red-teaming": "Red Teaming",
  ocr: "OCR",
  p5js: "p5.js",
  ai: "AI",
  ux: "UX",
  ui: "UI",
};

function prettyCategory(
  raw: string | null | undefined,
  generalLabel: string,
): string {
  if (!raw) return generalLabel;
  if (CATEGORY_LABELS[raw]) return CATEGORY_LABELS[raw];
  return raw
    .split(/[-_/]/)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

const TOOLSET_ICONS: Record<
  string,
  React.ComponentType<{ className?: string }>
> = {
  computer: Cpu,
  web: Globe,
  security: Shield,
  vision: Eye,
  design: Paintbrush,
  ai: Brain,
  integration: Blocks,
  code: Code,
  automation: Zap,
};

function toolsetIcon(
  name: string,
): React.ComponentType<{ className?: string }> {
  const lower = name.toLowerCase();
  for (const [key, icon] of Object.entries(TOOLSET_ICONS)) {
    if (lower.includes(key)) return icon;
  }
  return Wrench;
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function SkillsPage() {
  const [skills, setSkills] = useState<SkillInfo[]>([]);
  const [toolsets, setToolsets] = useState<ToolsetInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [view, setView] = useState<"skills" | "toolsets" | "hub">("skills");
  const [activeCategory, setActiveCategory] = useState<string | null>(null);
  const [togglingSkills, setTogglingSkills] = useState<Set<string>>(new Set());
  const [configToolset, setConfigToolset] = useState<ToolsetInfo | null>(null);
  const { toast, showToast } = useToast();
  const { t } = useI18n();
  const { setAfterTitle, setEnd } = usePageHeader();

  useEffect(() => {
    Promise.all([api.getSkills(), api.getToolsets()])
      .then(([s, tsets]) => {
        setSkills(s);
        setToolsets(tsets);
      })
      .catch(() => showToast(t.common.loading, "error"))
      .finally(() => setLoading(false));
  }, []);

  /* ---- Toggle skill ---- */
  const handleToggleSkill = async (skill: SkillInfo) => {
    setTogglingSkills((prev) => new Set(prev).add(skill.name));
    try {
      await api.toggleSkill(skill.name, !skill.enabled);
      setSkills((prev) =>
        prev.map((s) =>
          s.name === skill.name ? { ...s, enabled: !s.enabled } : s,
        ),
      );
      showToast(
        `${skill.name} ${skill.enabled ? t.common.disabled : t.common.enabled}`,
        "success",
      );
    } catch {
      showToast(`${t.common.failedToToggle} ${skill.name}`, "error");
    } finally {
      setTogglingSkills((prev) => {
        const next = new Set(prev);
        next.delete(skill.name);
        return next;
      });
    }
  };

  /* ---- Refresh toolsets after a config change ---- */
  const refreshToolsets = async () => {
    try {
      const tsets = await api.getToolsets();
      setToolsets(tsets);
    } catch {
      /* non-fatal: the drawer already toasted on the failing write */
    }
  };

  /* ---- Derived data ---- */
  const lowerSearch = search.toLowerCase();
  const isSearching = search.trim().length > 0;

  const searchMatchedSkills = useMemo(() => {
    if (!isSearching) return [];
    return skills.filter(
      (s) =>
        s.name.toLowerCase().includes(lowerSearch) ||
        s.description.toLowerCase().includes(lowerSearch) ||
        (s.category ?? "").toLowerCase().includes(lowerSearch),
    );
  }, [skills, isSearching, lowerSearch]);

  const activeSkills = useMemo(() => {
    if (isSearching) return [];
    if (!activeCategory)
      return [...skills].sort((a, b) => a.name.localeCompare(b.name));
    return skills
      .filter((s) =>
        activeCategory === "__none__"
          ? !s.category
          : s.category === activeCategory,
      )
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [skills, activeCategory, isSearching]);

  const allCategories = useMemo(() => {
    const cats = new Map<string, number>();
    for (const s of skills) {
      const key = s.category || "__none__";
      cats.set(key, (cats.get(key) || 0) + 1);
    }
    return [...cats.entries()]
      .sort((a, b) => {
        if (a[0] === "__none__") return -1;
        if (b[0] === "__none__") return 1;
        return a[0].localeCompare(b[0]);
      })
      .map(([key, count]) => ({
        key,
        name: prettyCategory(key === "__none__" ? null : key, t.common.general),
        count,
      }));
  }, [skills, t]);

  const enabledCount = skills.filter((s) => s.enabled).length;

  useLayoutEffect(() => {
    if (loading) {
      setAfterTitle(null);
      setEnd(null);
      return;
    }
    setAfterTitle(
      <span className="whitespace-nowrap text-xs text-muted-foreground">
        {t.skills.enabledOf
          .replace("{enabled}", String(enabledCount))
          .replace("{total}", String(skills.length))}
      </span>,
    );
    setEnd(
      <div className="relative w-full min-w-0 sm:max-w-xs">
        <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
        <Input
          className="h-8 rounded-none pl-8 pr-7 text-xs"
          placeholder={t.common.search}
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        {search && (
          <Button
            ghost
            size="xs"
            className="absolute right-1.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
            onClick={() => setSearch("")}
            aria-label={t.common.clear}
          >
            <X />
          </Button>
        )}
      </div>,
    );
    return () => {
      setAfterTitle(null);
      setEnd(null);
    };
  }, [enabledCount, loading, search, setAfterTitle, setEnd, skills.length, t]);

  const filteredToolsets = useMemo(() => {
    return toolsets.filter(
      (ts) =>
        !search ||
        ts.name.toLowerCase().includes(lowerSearch) ||
        ts.label.toLowerCase().includes(lowerSearch) ||
        ts.description.toLowerCase().includes(lowerSearch),
    );
  }, [toolsets, search, lowerSearch]);

  /* ---- Loading ---- */
  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner className="text-2xl text-primary" />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <PluginSlot name="skills:top" />
      <Toast toast={toast} />

      <div className="flex flex-col sm:flex-row sm:items-start gap-4">
        <aside aria-label={t.skills.title} className="sm:w-56 sm:shrink-0">
          <div className="sm:sticky sm:top-0">
            <div className="flex flex-col rounded-none border border-border bg-muted/20">
              <div className="hidden sm:flex items-center gap-2 px-3 py-2 border-b border-border">
                <Filter className="h-3 w-3 text-text-tertiary" />
                <span className="font-mondwest text-display text-xs tracking-[0.12em] text-text-secondary">
                  {t.skills.filters}
                </span>
              </div>

              <div className="flex sm:flex-col gap-1 overflow-x-auto sm:overflow-x-visible scrollbar-none p-2">
                <PanelItem
                  icon={Package}
                  label={`${t.skills.all} (${skills.length})`}
                  active={view === "skills" && !isSearching}
                  onClick={() => {
                    setView("skills");
                    setActiveCategory(null);
                    setSearch("");
                  }}
                />
                <PanelItem
                  icon={Wrench}
                  label={`${t.skills.toolsets} (${toolsets.length})`}
                  active={view === "toolsets"}
                  onClick={() => {
                    setView("toolsets");
                    setSearch("");
                  }}
                />
                <PanelItem
                  icon={Search}
                  label="Browse hub"
                  active={view === "hub"}
                  onClick={() => {
                    setView("hub");
                    setSearch("");
                  }}
                />
              </div>

              {view === "skills" &&
                !isSearching &&
                allCategories.length > 0 && (
                  <div className="hidden sm:flex flex-col border-t border-border">
                    <div className="px-3 pt-2 pb-1 font-mondwest text-display text-xs tracking-[0.12em] text-text-tertiary">
                      {t.skills.categories}
                    </div>
                    <div className="flex flex-col p-2 pt-1 gap-px max-h-[calc(100vh-340px)] overflow-y-auto">
                      {allCategories.map(({ key, name, count }) => {
                        const isActive = activeCategory === key;

                        return (
                          <ListItem
                            key={key}
                            active={isActive}
                            onClick={() =>
                              setActiveCategory(isActive ? null : key)
                            }
                            className="rounded-none px-2 py-1 text-xs"
                          >
                            <span className="flex-1 truncate">{name}</span>
                            <span
                              className={`text-xs tabular-nums ${
                                isActive
                                  ? "text-text-secondary"
                                  : "text-text-tertiary"
                              }`}
                            >
                              {count}
                            </span>
                          </ListItem>
                        );
                      })}
                    </div>
                  </div>
                )}
            </div>
          </div>
        </aside>

        <div className="flex-1 min-w-0">
          {isSearching ? (
            <Card className="rounded-none">
              <CardHeader className="py-3 px-4">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Search className="h-4 w-4" />
                    {t.skills.title}
                  </CardTitle>
                  <Badge tone="secondary" className="text-xs">
                    {t.skills.resultCount
                      .replace("{count}", String(searchMatchedSkills.length))
                      .replace(
                        "{s}",
                        searchMatchedSkills.length !== 1 ? "s" : "",
                      )}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="px-4 pb-4">
                {searchMatchedSkills.length === 0 ? (
                  <p className="text-sm text-muted-foreground text-center py-8">
                    {t.skills.noSkillsMatch}
                  </p>
                ) : (
                  <div className="grid gap-1">
                    {searchMatchedSkills.map((skill) => (
                      <SkillRow
                        key={skill.name}
                        skill={skill}
                        toggling={togglingSkills.has(skill.name)}
                        onToggle={() => handleToggleSkill(skill)}
                        noDescriptionLabel={t.skills.noDescription}
                      />
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          ) : view === "skills" ? (
            /* Skills list */
            <Card className="rounded-none">
              <CardHeader className="py-3 px-4">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Package className="h-4 w-4" />
                    {activeCategory
                      ? prettyCategory(
                          activeCategory === "__none__" ? null : activeCategory,
                          t.common.general,
                        )
                      : t.skills.all}
                  </CardTitle>
                  <Badge tone="secondary" className="text-xs">
                    {t.skills.skillCount
                      .replace("{count}", String(activeSkills.length))
                      .replace("{s}", activeSkills.length !== 1 ? "s" : "")}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="px-4 pb-4">
                {activeSkills.length === 0 ? (
                  <p className="text-sm text-muted-foreground text-center py-8">
                    {skills.length === 0
                      ? t.skills.noSkills
                      : t.skills.noSkillsMatch}
                  </p>
                ) : (
                  <div className="grid gap-1">
                    {activeSkills.map((skill) => (
                      <SkillRow
                        key={skill.name}
                        skill={skill}
                        toggling={togglingSkills.has(skill.name)}
                        onToggle={() => handleToggleSkill(skill)}
                        noDescriptionLabel={t.skills.noDescription}
                      />
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          ) : view === "toolsets" ? (
            /* Toolsets grid */
            <>
              {filteredToolsets.length === 0 ? (
                <Card className="rounded-none">
                  <CardContent className="py-8 text-center text-sm text-muted-foreground">
                    {t.skills.noToolsetsMatch}
                  </CardContent>
                </Card>
              ) : (
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  {filteredToolsets.map((ts) => {
                    const TsIcon = toolsetIcon(ts.name);
                    const labelText = ts.label.trim() || ts.name;

                    return (
                      <Card key={ts.name} className="relative rounded-none">
                        <CardContent className="py-4">
                          <div className="flex items-start gap-3">
                            <TsIcon className="h-5 w-5 text-muted-foreground shrink-0 mt-0.5" />
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2 mb-1">
                                <span className="font-medium text-sm">
                                  {labelText}
                                </span>
                                <Badge
                                  tone={ts.enabled ? "success" : "outline"}
                                  className="text-xs"
                                >
                                  {ts.enabled
                                    ? t.common.active
                                    : t.common.inactive}
                                </Badge>
                              </div>
                              <p className="text-xs text-text-secondary mb-2">
                                {ts.description}
                              </p>
                              {ts.enabled && !ts.configured && (
                                <p className="text-xs text-amber-300 mb-2">
                                  {t.skills.setupNeeded}
                                </p>
                              )}
                              {ts.tools.length > 0 && (
                                <div className="flex flex-wrap gap-1">
                                  {ts.tools.map((tool) => (
                                    <Badge
                                      key={tool}
                                      tone="secondary"
                                      className="text-xs font-mono"
                                    >
                                      {tool}
                                    </Badge>
                                  ))}
                                </div>
                              )}
                              {ts.tools.length === 0 && (
                                <span className="text-xs text-text-tertiary">
                                  {ts.enabled
                                    ? t.skills.toolsetLabel.replace(
                                        "{name}",
                                        ts.name,
                                      )
                                    : t.skills.disabledForCli}
                                </span>
                              )}
                              <div className="mt-3">
                                <Button
                                  size="xs"
                                  outlined
                                  onClick={() => setConfigToolset(ts)}
                                >
                                  <Wrench className="h-3 w-3 mr-1" />
                                  Configure
                                </Button>
                              </div>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    );
                  })}
                </div>
              )}
            </>
          ) : (
            <HubBrowser showToast={showToast} />
          )}
        </div>
      </div>
      {configToolset && (
        <ToolsetConfigDrawer
          toolset={configToolset}
          onClose={() => setConfigToolset(null)}
          onChanged={() => void refreshToolsets()}
        />
      )}
      <PluginSlot name="skills:bottom" />
    </div>
  );
}

function SkillRow({
  skill,
  toggling,
  onToggle,
  noDescriptionLabel,
}: SkillRowProps) {
  return (
    <div className="group flex items-start gap-3 px-3 py-2.5 transition-colors hover:bg-muted/40">
      <div className="pt-0.5 shrink-0">
        <Switch
          checked={skill.enabled}
          onCheckedChange={onToggle}
          disabled={toggling}
        />
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-0.5">
          <span
            className={`font-mono-ui text-sm ${
              skill.enabled ? "text-foreground" : "text-muted-foreground"
            }`}
          >
            {skill.name}
          </span>
        </div>
        <p className="text-xs text-muted-foreground leading-relaxed line-clamp-2">
          {skill.description || noDescriptionLabel}
        </p>
      </div>
    </div>
  );
}

function PanelItem({ active, icon: Icon, label, onClick }: PanelItemProps) {
  return (
    <ListItem
      active={active}
      onClick={onClick}
      className={cn(
        "rounded-none whitespace-nowrap px-2.5 py-1.5",
        "font-mondwest text-[0.7rem] tracking-[0.08em] uppercase",
        active && "bg-foreground/90 text-background hover:text-background",
      )}
    >
      <Icon className="h-3.5 w-3.5 shrink-0" />
      <span className="flex-1 truncate">{label}</span>
    </ListItem>
  );
}

interface PanelItemProps {
  active: boolean;
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  onClick: () => void;
}

interface SkillRowProps {
  noDescriptionLabel: string;
  onToggle: () => void;
  skill: SkillInfo;
  toggling: boolean;
}

/* ------------------------------------------------------------------ */
/*  Hub browser — search the skill hub, preview, scan, install         */
/* ------------------------------------------------------------------ */

/** Map a trust level to a Badge tone + label + icon. */
function trustVisual(level: string): {
  tone: "success" | "secondary" | "warning" | "outline";
  label: string;
} {
  switch (level) {
    case "trusted":
      return { tone: "success", label: "trusted" };
    case "builtin":
      return { tone: "secondary", label: "builtin" };
    case "community":
      return { tone: "warning", label: "community" };
    default:
      return { tone: "outline", label: level || "unknown" };
  }
}

/** Map a scan verdict to tone + icon. */
function verdictVisual(verdict: string): {
  tone: "success" | "warning" | "destructive";
  Icon: React.ComponentType<{ className?: string }>;
  label: string;
} {
  switch (verdict) {
    case "safe":
      return { tone: "success", Icon: ShieldCheck, label: "Safe" };
    case "caution":
      return { tone: "warning", Icon: ShieldAlert, label: "Caution" };
    case "dangerous":
      return { tone: "destructive", Icon: ShieldAlert, label: "Dangerous" };
    default:
      return { tone: "warning", Icon: ShieldQuestion, label: verdict };
  }
}

const SEVERITY_TONE: Record<string, "destructive" | "warning" | "secondary" | "outline"> = {
  critical: "destructive",
  high: "destructive",
  medium: "warning",
  low: "secondary",
};

function HubBrowser({
  showToast,
}: {
  showToast: (msg: string, kind: "success" | "error") => void;
}) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SkillHubResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [searched, setSearched] = useState(false);
  const [sourceCounts, setSourceCounts] = useState<Record<string, number>>({});
  const [timedOut, setTimedOut] = useState<string[]>([]);
  const [searchMs, setSearchMs] = useState<number | null>(null);

  // Landing state: which hubs are wired up + featured skills.
  const [sources, setSources] = useState<SkillHubSource[]>([]);
  const [featured, setFeatured] = useState<SkillHubResult[]>([]);
  const [sourcesLoading, setSourcesLoading] = useState(true);

  // identifier -> installed entry (drives "Installed" badges).
  const [installed, setInstalled] = useState<Record<string, SkillHubInstalledEntry>>({});

  // Live action log for the most recent install/update.
  const [action, setAction] = useState<string | null>(null);
  const [actionLog, setActionLog] = useState<string[]>([]);
  const [actionRunning, setActionRunning] = useState(false);

  // Detail dialog (preview + scan for a single skill).
  const [detail, setDetail] = useState<SkillHubResult | null>(null);

  /* ---- Load connected hubs + featured skills on mount ---- */
  useEffect(() => {
    let cancelled = false;
    api
      .getSkillHubSources()
      .then((r) => {
        if (cancelled) return;
        setSources(r.sources);
        setFeatured(r.featured);
        setInstalled(r.installed);
      })
      .catch(() => {
        /* leave landing minimal on failure */
      })
      .finally(() => {
        if (!cancelled) setSourcesLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  /* ---- Search ---- */
  const runSearch = useCallback(async () => {
    const q = query.trim();
    if (!q) return;
    setSearching(true);
    setSearched(true);
    const t0 = performance.now();
    try {
      const r = await api.searchSkillsHub(q);
      setResults(r.results);
      setSourceCounts(r.source_counts || {});
      setTimedOut(r.timed_out || []);
      setInstalled((prev) => ({ ...prev, ...(r.installed || {}) }));
    } catch (e) {
      showToast(`Hub search failed: ${e}`, "error");
      setResults([]);
      setSourceCounts({});
      setTimedOut([]);
    } finally {
      setSearchMs(Math.round(performance.now() - t0));
      setSearching(false);
    }
  }, [query, showToast]);

  /* ---- Poll a spawned action's log until it exits ---- */
  useEffect(() => {
    if (!action) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const poll = async () => {
      try {
        const st = await api.getActionStatus(action, 200);
        if (cancelled) return;
        setActionLog(st.lines);
        setActionRunning(st.running);
        if (st.running) {
          timer = setTimeout(poll, 1200);
        } else {
          // Install finished — refresh installed-state so badges update.
          api
            .getSkillHubSources()
            .then((r) => !cancelled && setInstalled(r.installed))
            .catch(() => {});
        }
      } catch {
        if (!cancelled) setActionRunning(false);
      }
    };
    poll();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [action]);

  const install = useCallback(
    async (identifier: string) => {
      try {
        const res = await api.installSkillFromHub(identifier);
        showToast(`Installing ${identifier}…`, "success");
        setActionLog([]);
        setActionRunning(true);
        setAction(res.name);
        setDetail(null);
      } catch (e) {
        showToast(`Install failed: ${e}`, "error");
      }
    },
    [showToast],
  );

  const updateAll = useCallback(async () => {
    try {
      const res = await api.updateSkillsFromHub();
      showToast("Updating installed skills…", "success");
      setActionLog([]);
      setActionRunning(true);
      setAction(res.name);
    } catch (e) {
      showToast(`Update failed: ${e}`, "error");
    }
  }, [showToast]);

  const isInstalled = useCallback(
    (identifier: string) => Boolean(installed[identifier]),
    [installed],
  );

  const showLanding = !searched && !searching;

  return (
    <div className="flex flex-col gap-3">
      {/* ── Search bar ── */}
      <Card className="rounded-none">
        <CardContent className="py-4 flex flex-col gap-3">
          <div className="flex items-center gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
              <Input
                className="h-8 pl-8 text-sm"
                placeholder="Search the skill hub (GitHub, official, community)…"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") void runSearch();
                }}
              />
            </div>
            <Button
              size="sm"
              onClick={() => void runSearch()}
              disabled={searching || !query.trim()}
              prefix={searching ? <Spinner /> : <Search className="h-3.5 w-3.5" />}
            >
              Search
            </Button>
            <Button
              size="sm"
              outlined
              onClick={() => void updateAll()}
              prefix={<RefreshCw className="h-3.5 w-3.5" />}
            >
              Update all
            </Button>
          </div>

          {/* Connected hubs strip — proves the tab is wired up. */}
          <ConnectedHubs sources={sources} loading={sourcesLoading} />
        </CardContent>
      </Card>

      {/* ── Install/update action log ── */}
      {action && (
        <Card className="rounded-none">
          <CardContent className="py-3">
            <div className="flex items-center gap-2 mb-2">
              <Download className="h-3.5 w-3.5 text-muted-foreground" />
              <span className="font-mono text-xs">{action}</span>
              {actionRunning ? (
                <Badge tone="warning">running</Badge>
              ) : (
                <Badge tone="success">done</Badge>
              )}
              {!actionRunning && (
                <Button
                  ghost
                  size="xs"
                  className="ml-auto text-muted-foreground"
                  onClick={() => setAction(null)}
                  aria-label="Dismiss"
                >
                  <X className="h-3.5 w-3.5" />
                </Button>
              )}
            </div>
            <pre className="max-h-48 overflow-auto whitespace-pre-wrap break-words bg-background/50 border border-border p-2 text-xs font-mono text-muted-foreground">
              {actionLog.length ? actionLog.join("\n") : "Starting…"}
            </pre>
          </CardContent>
        </Card>
      )}

      {/* ── Landing: featured skills (before any search) ── */}
      {showLanding && (
        <>
          {sourcesLoading ? (
            <div className="flex items-center justify-center py-12">
              <Spinner className="text-xl text-primary" />
            </div>
          ) : featured.length > 0 ? (
            <div className="flex flex-col gap-2">
              <div className="flex items-center gap-2 px-1">
                <Sparkles className="h-3.5 w-3.5 text-primary" />
                <span className="font-mondwest text-display text-xs tracking-[0.12em] text-text-secondary uppercase">
                  Featured skills
                </span>
                <span className="text-xs text-text-tertiary">
                  from the Hermes index — search above for thousands more
                </span>
              </div>
              {featured.map((r) => (
                <HubResultCard
                  key={r.identifier}
                  result={r}
                  installed={isInstalled(r.identifier)}
                  onOpen={() => setDetail(r)}
                  onInstall={() => void install(r.identifier)}
                />
              ))}
            </div>
          ) : (
            <Card className="rounded-none">
              <CardContent className="py-10 text-center text-sm text-muted-foreground">
                Search the hub above to browse installable skills from the
                connected sources.
              </CardContent>
            </Card>
          )}
        </>
      )}

      {/* ── Searching spinner ── */}
      {searching && (
        <div className="flex items-center justify-center py-8">
          <Spinner className="text-xl text-primary" />
        </div>
      )}

      {/* ── Search results ── */}
      {!searching && searched && (
        <>
          <SearchMeta
            count={results.length}
            sourceCounts={sourceCounts}
            timedOut={timedOut}
            ms={searchMs}
          />
          {results.length === 0 ? (
            <Card className="rounded-none">
              <CardContent className="py-8 text-center text-sm text-muted-foreground">
                No matching skills found in the hub.
              </CardContent>
            </Card>
          ) : (
            results.map((r) => (
              <HubResultCard
                key={r.identifier}
                result={r}
                installed={isInstalled(r.identifier)}
                onOpen={() => setDetail(r)}
                onInstall={() => void install(r.identifier)}
              />
            ))
          )}
        </>
      )}

      {/* ── Detail dialog: preview + scan ── */}
      {detail && (
        <SkillDetailDialog
          result={detail}
          installed={isInstalled(detail.identifier)}
          onClose={() => setDetail(null)}
          onInstall={() => void install(detail.identifier)}
          showToast={showToast}
        />
      )}
    </div>
  );
}

/* ---- Connected hubs strip ---- */
function ConnectedHubs({
  sources,
  loading,
}: {
  sources: SkillHubSource[];
  loading: boolean;
}) {
  if (loading) {
    return (
      <p className="text-xs text-muted-foreground">Connecting to skill hubs…</p>
    );
  }
  if (sources.length === 0) {
    return (
      <p className="text-xs text-muted-foreground">
        Results come from the same sources as{" "}
        <span className="font-mono">hermes skills search</span>.
      </p>
    );
  }
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <span className="flex items-center gap-1 text-xs text-text-tertiary">
        <Globe className="h-3 w-3" />
        Connected hubs:
      </span>
      {sources.map((s) => {
        const down =
          (s.id === "hermes-index" && s.available === false) ||
          (s.id === "github" && s.rate_limited === true);
        return (
          <Badge
            key={s.id}
            tone={down ? "outline" : "secondary"}
            className={cn("text-xs", down && "opacity-60")}
            title={
              s.id === "github" && s.rate_limited
                ? "GitHub API rate-limited — set GITHUB_TOKEN to raise the limit"
                : s.id === "hermes-index" && s.available === false
                  ? "Centralized index unavailable — falling back to live sources"
                  : undefined
            }
          >
            {s.label}
            {s.id === "github" && s.rate_limited ? " (rate-limited)" : ""}
          </Badge>
        );
      })}
    </div>
  );
}

/* ---- Search result-count + per-source breakdown ---- */
function SearchMeta({
  count,
  sourceCounts,
  timedOut,
  ms,
}: {
  count: number;
  sourceCounts: Record<string, number>;
  timedOut: string[];
  ms: number | null;
}) {
  const entries = Object.entries(sourceCounts).filter(([, n]) => n > 0);
  return (
    <div className="flex flex-wrap items-center gap-2 px-1 text-xs text-text-tertiary">
      <Badge tone="secondary" className="text-xs">
        {count} result{count !== 1 ? "s" : ""}
      </Badge>
      {ms != null && <span>{(ms / 1000).toFixed(1)}s</span>}
      {entries.length > 0 && (
        <span className="flex flex-wrap items-center gap-1.5">
          {entries.map(([sid, n]) => (
            <span key={sid} className="font-mono">
              {sid}:{n}
            </span>
          ))}
        </span>
      )}
      {timedOut.length > 0 && (
        <span className="flex items-center gap-1 text-amber-400">
          <AlertTriangle className="h-3 w-3" />
          {timedOut.join(", ")} timed out
        </span>
      )}
    </div>
  );
}

/* ---- One result card ---- */
function HubResultCard({
  result,
  installed,
  onOpen,
  onInstall,
}: {
  result: SkillHubResult;
  installed: boolean;
  onOpen: () => void;
  onInstall: () => void;
}) {
  const trust = trustVisual(result.trust_level);
  return (
    <Card className="rounded-none transition-colors hover:bg-muted/30">
      <CardContent className="py-3 flex items-start gap-3">
        <button
          type="button"
          className="flex-1 min-w-0 text-left"
          onClick={onOpen}
          aria-label={`Open ${result.name}`}
        >
          <div className="flex flex-wrap items-center gap-2 mb-0.5">
            <span className="font-mono-ui text-sm hover:underline">
              {result.name}
            </span>
            <Badge tone={trust.tone} className="text-xs">
              {trust.label}
            </Badge>
            <Badge tone="secondary" className="text-xs">
              {result.source}
            </Badge>
            {installed && (
              <Badge tone="success" className="text-xs">
                installed
              </Badge>
            )}
          </div>
          <p className="text-xs text-text-secondary line-clamp-2">
            {result.description}
          </p>
          <div className="flex flex-wrap items-center gap-1 mt-1">
            {result.tags.slice(0, 5).map((tag) => (
              <span
                key={tag}
                className="text-[0.65rem] font-mono text-text-tertiary border border-border px-1 py-px"
              >
                {tag}
              </span>
            ))}
          </div>
          <p className="text-xs font-mono text-text-tertiary truncate mt-1">
            {result.identifier}
          </p>
        </button>
        <div className="flex shrink-0 flex-col gap-1.5">
          <Button
            size="sm"
            outlined
            onClick={onOpen}
            prefix={<FileText className="h-3.5 w-3.5" />}
          >
            Details
          </Button>
          {installed ? (
            <Button size="sm" ghost disabled prefix={<CheckCircle2 className="h-3.5 w-3.5" />}>
              Installed
            </Button>
          ) : (
            <Button
              size="sm"
              onClick={onInstall}
              prefix={<Download className="h-3.5 w-3.5" />}
            >
              Install
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

/* ---- Detail dialog: SKILL.md preview + on-demand security scan ---- */
function SkillDetailDialog({
  result,
  installed,
  onClose,
  onInstall,
  showToast,
}: {
  result: SkillHubResult;
  installed: boolean;
  onClose: () => void;
  onInstall: () => void;
  showToast: (msg: string, kind: "success" | "error") => void;
}) {
  const [tab, setTab] = useState<"readme" | "scan">("readme");
  const [preview, setPreview] = useState<SkillHubPreview | null>(null);
  const [previewLoading, setPreviewLoading] = useState(true);
  const [scan, setScan] = useState<SkillHubScan | null>(null);
  const [scanning, setScanning] = useState(false);
  const trust = trustVisual(result.trust_level);

  useEffect(() => {
    let cancelled = false;
    setPreviewLoading(true);
    api
      .previewSkillFromHub(result.identifier)
      .then((p) => !cancelled && setPreview(p))
      .catch((e) => {
        if (!cancelled) showToast(`Preview failed: ${e}`, "error");
      })
      .finally(() => !cancelled && setPreviewLoading(false));
    return () => {
      cancelled = true;
    };
  }, [result.identifier, showToast]);

  const runScan = useCallback(async () => {
    setScanning(true);
    setTab("scan");
    try {
      const s = await api.scanSkillFromHub(result.identifier);
      setScan(s);
    } catch (e) {
      showToast(`Scan failed: ${e}`, "error");
    } finally {
      setScanning(false);
    }
  }, [result.identifier, showToast]);

  return (
    <Dialog open onOpenChange={(o: boolean) => !o && onClose()}>
      <DialogContent className="max-w-3xl rounded-none">
        <DialogHeader>
          <DialogTitle className="flex flex-wrap items-center gap-2 text-sm">
            <Package className="h-4 w-4" />
            {result.name}
            <Badge tone={trust.tone} className="text-xs">
              {trust.label}
            </Badge>
            <Badge tone="secondary" className="text-xs">
              {result.source}
            </Badge>
            {installed && (
              <Badge tone="success" className="text-xs">
                installed
              </Badge>
            )}
          </DialogTitle>
          <DialogDescription className="sr-only">
            Preview the SKILL.md source and run a security scan for {result.name}{" "}
            before installing.
          </DialogDescription>
        </DialogHeader>

        <div className="mt-1 flex flex-col gap-1">
          <p className="text-xs text-text-secondary">{result.description}</p>
          <p className="text-xs font-mono text-text-tertiary truncate">
            {result.identifier}
          </p>
        </div>

        {/* Action row */}
        <div className="mt-3 flex flex-wrap items-center gap-2 border-y border-border py-2.5">
          <Button
            size="sm"
            outlined={tab !== "readme"}
            onClick={() => setTab("readme")}
            prefix={<FileText className="h-3.5 w-3.5" />}
          >
            Read SKILL.md
          </Button>
          <Button
            size="sm"
            outlined={tab !== "scan"}
            onClick={() => void runScan()}
            disabled={scanning}
            prefix={
              scanning ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Shield className="h-3.5 w-3.5" />
              )
            }
          >
            {scan ? "Re-scan" : "Security scan"}
          </Button>
          <div className="ml-auto flex items-center gap-3">
            {result.repo && (
              <a
                href={`https://github.com/${result.repo}`}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
              >
                <ExternalLink className="h-3.5 w-3.5" />
                {result.repo}
              </a>
            )}
            {installed ? (
              <Button size="sm" ghost disabled prefix={<CheckCircle2 className="h-3.5 w-3.5" />}>
                Installed
              </Button>
            ) : (
              <Button
                size="sm"
                onClick={onInstall}
                prefix={<Download className="h-3.5 w-3.5" />}
              >
                Install
              </Button>
            )}
          </div>
        </div>

        {/* Body */}
        <div className="mt-3 max-h-[55vh] overflow-auto">
          {tab === "readme" ? (
            previewLoading ? (
              <div className="flex items-center justify-center py-12">
                <Spinner className="text-xl text-primary" />
              </div>
            ) : preview ? (
              <div className="flex flex-col gap-2.5">
                {preview.tags.length > 0 && (
                  <div className="flex flex-wrap items-center gap-1">
                    {preview.tags.map((tag) => (
                      <span
                        key={tag}
                        className="text-[0.65rem] font-mono text-text-tertiary border border-border px-1 py-px"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                )}
                {preview.files.length > 0 && (
                  <div className="text-xs text-text-tertiary">
                    <span className="font-mondwest tracking-[0.1em] uppercase">
                      Files:{" "}
                    </span>
                    <span className="font-mono">{preview.files.join("  ")}</span>
                  </div>
                )}
                <pre className="whitespace-pre-wrap break-words bg-background/50 border border-border p-3 text-xs font-mono text-text-secondary leading-relaxed">
                  {(preview.skill_md || "").trim() || "(SKILL.md is empty)"}
                </pre>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground text-center py-10">
                Couldn't load the skill source.
              </p>
            )
          ) : (
            <ScanPanel scan={scan} scanning={scanning} />
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}

/* ---- Visual security-scan result ---- */
function ScanPanel({
  scan,
  scanning,
}: {
  scan: SkillHubScan | null;
  scanning: boolean;
}) {
  if (scanning && !scan) {
    return (
      <div className="flex flex-col items-center justify-center gap-2 py-12">
        <Loader2 className="h-6 w-6 animate-spin text-primary" />
        <span className="text-xs text-muted-foreground">
          Fetching, quarantining, and scanning…
        </span>
      </div>
    );
  }
  if (!scan) {
    return (
      <p className="text-sm text-muted-foreground text-center py-10">
        Run a security scan to inspect this skill for risky patterns before
        installing.
      </p>
    );
  }

  const v = verdictVisual(scan.verdict);
  const policyTone =
    scan.policy === "allow"
      ? "success"
      : scan.policy === "ask"
        ? "warning"
        : "destructive";
  const policyLabel =
    scan.policy === "allow"
      ? "Install allowed"
      : scan.policy === "ask"
        ? "Needs confirmation"
        : "Install blocked";

  return (
    <div className="flex flex-col gap-3">
      {/* Verdict header */}
      <div className="flex flex-wrap items-center gap-2 border border-border p-3">
        <v.Icon
          className={cn(
            "h-6 w-6",
            scan.verdict === "safe"
              ? "text-emerald-400"
              : scan.verdict === "dangerous"
                ? "text-red-400"
                : "text-amber-400",
          )}
        />
        <div className="flex flex-col">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium">Verdict: {v.label}</span>
            <Badge tone={v.tone} className="text-xs">
              {scan.verdict}
            </Badge>
          </div>
          <span className="text-xs text-text-tertiary">
            {scan.trust_level} source · {scan.findings.length} finding
            {scan.findings.length !== 1 ? "s" : ""}
          </span>
        </div>
        <Badge tone={policyTone} className="ml-auto text-xs">
          {policyLabel}
        </Badge>
      </div>

      {/* Severity tally */}
      <div className="flex flex-wrap items-center gap-1.5">
        {(["critical", "high", "medium", "low"] as const).map((sev) => {
          const n = scan.severity_counts[sev] || 0;
          if (n === 0) return null;
          return (
            <Badge key={sev} tone={SEVERITY_TONE[sev]} className="text-xs">
              {n} {sev}
            </Badge>
          );
        })}
        {scan.findings.length === 0 && (
          <span className="flex items-center gap-1 text-xs text-emerald-400">
            <CheckCircle2 className="h-3.5 w-3.5" />
            No risky patterns detected
          </span>
        )}
      </div>

      <p className="text-xs text-text-tertiary">{scan.policy_reason}</p>

      {/* Findings */}
      {scan.findings.length > 0 && (
        <div className="flex flex-col border border-border divide-y divide-border">
          {scan.findings.map((f, i) => (
            <div key={i} className="flex items-start gap-2 p-2">
              <Badge tone={SEVERITY_TONE[f.severity] || "outline"} className="text-xs shrink-0">
                {f.severity}
              </Badge>
              <div className="flex-1 min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-xs font-medium">{f.category}</span>
                  <span className="text-xs font-mono text-text-tertiary truncate">
                    {f.file}:{f.line}
                  </span>
                </div>
                <p className="text-xs text-text-secondary">{f.description}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
