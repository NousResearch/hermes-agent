import { useEffect, useState, useMemo, useCallback } from "react";
import {
  Package,
  Search,
  Wrench,
  ChevronRight,
  X,
  Cpu,
  Globe,
  Shield,
  Eye,
  Paintbrush,
  Brain,
  Blocks,
  Code,
  Zap,
  Download,
  Upload,
  RefreshCw,
  ShieldCheck,
  Plus,
  Globe2,
  Trash2,
  FileUp,
  Rocket,
} from "lucide-react";
import { H2 } from "@nous-research/ui";
import { api } from "@/lib/api";
import type {
  SkillInfo,
  ToolsetInfo,
  HubSkillInfo,
  SkillSearchResult,
  SkillUpdateCheck,
  SkillAuditResult,
  TapInfo,
  SkillSnapshot,
} from "@/lib/api";
import { useToast } from "@/hooks/useToast";
import { Toast } from "@/components/Toast";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Label } from "@/components/ui/label";
import { useI18n } from "@/i18n";

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

type SubView =
  | "all"
  | "browse"
  | "search"
  | "updates"
  | "audit"
  | "taps"
  | "snapshot"
  | "create"
  | "toolsets";

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function SkillsPage() {
  const [skills, setSkills] = useState<SkillInfo[]>([]);
  const [hubSkills, setHubSkills] = useState<HubSkillInfo[]>([]);
  const [toolsets, setToolsets] = useState<ToolsetInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [view, setView] = useState<"skills" | "toolsets">("skills");
  const [subView, setSubView] = useState<SubView>("all");
  const [activeCategory, setActiveCategory] = useState<string | null>(null);
  const [togglingSkills, setTogglingSkills] = useState<Set<string>>(new Set());
  const { toast, showToast } = useToast();
  const { t } = useI18n();

  // Sub-view state
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<SkillSearchResult[]>([]);
  const [browseResults, setSearchBrowseResults] = useState<SkillSearchResult[]>([]);
  const [updateChecks, setUpdateChecks] = useState<SkillUpdateCheck[]>([]);
  const [auditResults, setAuditResults] = useState<SkillAuditResult[]>([]);
  const [taps, setTapsList] = useState<TapInfo[]>([]);
  const [snapshotData, setSnapshotData] = useState<SkillSnapshot | null>(null);
  const [busy, setBusy] = useState(false);

  // Dialog state
  const [installId, setInstallId] = useState("");
  const [createName, setCreateName] = useState("");
  const [createDisplay, setCreateDisplay] = useState("");
  const [createDesc, setCreateDesc] = useState("");
  const [createCategory, setCreateCategory] = useState("");
  const [createTags, setCreateTags] = useState("");
  const [tapRepo, setTapRepo] = useState("");
  const [publishPath, setPublishPath] = useState("");
  const [publishRepo, setPublishRepo] = useState("");

  useEffect(() => {
    Promise.all([api.getSkills(), api.getToolsets(), api.getInstalledSkills()])
      .then(([s, tsets, hub]) => {
        setSkills(s);
        setToolsets(tsets);
        setHubSkills(hub);
      })
      .catch(() => showToast(t.common.loading, "error"))
      .finally(() => setLoading(false));
  }, []);

  const refreshHubSkills = useCallback(async () => {
    try {
      const hub = await api.getInstalledSkills();
      setHubSkills(hub);
    } catch { /* ignore */ }
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

  /* ---- Hub actions ---- */
  const handleSearch = useCallback(async () => {
    if (!searchQuery.trim()) return;
    setBusy(true);
    try {
      const results = await api.searchSkills(searchQuery);
      setSearchResults(results);
      setSubView("search");
    } catch (e) {
      showToast(`Search failed: ${(e as Error).message}`, "error");
    } finally {
      setBusy(false);
    }
  }, [searchQuery, showToast]);

  const handleBrowse = useCallback(async () => {
    setBusy(true);
    try {
      const results = await api.browseSkills(1, 50);
      setSearchBrowseResults(results.skills || []);
      setSubView("browse");
    } catch (e) {
      showToast(`Browse failed: ${(e as Error).message}`, "error");
    } finally {
      setBusy(false);
    }
  }, [showToast]);

  const handleInstall = useCallback(async () => {
    if (!installId.trim()) return;
    setBusy(true);
    try {
      const result = await api.installSkill(installId);
      showToast(`Installed: ${result.name}`, "success");
      setInstallId("");
      await refreshHubSkills();
    } catch (e) {
      showToast(`Install failed: ${(e as Error).message}`, "error");
    } finally {
      setBusy(false);
    }
  }, [installId, showToast, refreshHubSkills]);

  const handleUninstall = useCallback(async (name: string) => {
    setBusy(true);
    try {
      await api.uninstallSkill(name);
      showToast(`Uninstalled: ${name}`, "success");
      await refreshHubSkills();
    } catch (e) {
      showToast(`Uninstall failed: ${(e as Error).message}`, "error");
    } finally {
      setBusy(false);
    }
  }, [showToast, refreshHubSkills]);

  const handleCheckUpdates = useCallback(async () => {
    setBusy(true);
    try {
      const checks = await api.checkSkillUpdates();
      setUpdateChecks(checks);
      setSubView("updates");
    } catch (e) {
      showToast(`Check failed: ${(e as Error).message}`, "error");
    } finally {
      setBusy(false);
    }
  }, [showToast]);

  const handleUpdateAll = useCallback(async () => {
    setBusy(true);
    try {
      const result = await api.updateSkill();
      showToast(`Updated ${result.updated} skills`, "success");
    } catch (e) {
      showToast(`Update failed: ${(e as Error).message}`, "error");
    } finally {
      setBusy(false);
    }
  }, [showToast]);

  const handleAudit = useCallback(async () => {
    setBusy(true);
    try {
      const results = await api.auditSkills();
      setAuditResults(results);
      setSubView("audit");
    } catch (e) {
      showToast(`Audit failed: ${(e as Error).message}`, "error");
    } finally {
      setBusy(false);
    }
  }, [showToast]);

  const handleLoadTaps = useCallback(async () => {
    setBusy(true);
    try {
      const list = await api.listTaps();
      setTapsList(list);
      setSubView("taps");
    } catch (e) {
      showToast(`Load taps failed: ${(e as Error).message}`, "error");
    } finally {
      setBusy(false);
    }
  }, [showToast]);

  const handleAddTap = useCallback(async () => {
    if (!tapRepo.trim()) return;
    setBusy(true);
    try {
      await api.addTap(tapRepo);
      showToast(`Tap added: ${tapRepo}`, "success");
      setTapRepo("");
      const list = await api.listTaps();
      setTapsList(list);
    } catch (e) {
      showToast(`Add tap failed: ${(e as Error).message}`, "error");
    } finally {
      setBusy(false);
    }
  }, [tapRepo, showToast]);

  const handleRemoveTap = useCallback(async (repo: string) => {
    setBusy(true);
    try {
      await api.removeTap(repo);
      showToast(`Tap removed: ${repo}`, "success");
      setTapsList((prev) => prev.filter((t) => t.repo !== repo));
    } catch (e) {
      showToast(`Remove tap failed: ${(e as Error).message}`, "error");
    } finally {
      setBusy(false);
    }
  }, [showToast]);

  const handleExportSnapshot = useCallback(async () => {
    setBusy(true);
    try {
      const data = await api.exportSnapshot();
      setSnapshotData(data);
      setSubView("snapshot");
    } catch (e) {
      showToast(`Export failed: ${(e as Error).message}`, "error");
    } finally {
      setBusy(false);
    }
  }, [showToast]);

  const handleImportSnapshot = useCallback(async () => {
    if (!snapshotData) return;
    setBusy(true);
    try {
      const result = await api.importSnapshot(snapshotData, true);
      showToast(`Imported ${result.installed} skills`, "success");
      await refreshHubSkills();
    } catch (e) {
      showToast(`Import failed: ${(e as Error).message}`, "error");
    } finally {
      setBusy(false);
    }
  }, [snapshotData, showToast, refreshHubSkills]);

  const handleCreateSkill = useCallback(async () => {
    if (!createName.trim()) return;
    setBusy(true);
    try {
      const result = await api.createSkill(
        createName, createDisplay, createDesc, createCategory, createTags,
      );
      showToast(`Created: ${result.name}`, "success");
      setCreateName(""); setCreateDisplay(""); setCreateDesc("");
      setCreateCategory(""); setCreateTags("");
      await refreshHubSkills();
    } catch (e) {
      showToast(`Create failed: ${(e as Error).message}`, "error");
    } finally {
      setBusy(false);
    }
  }, [createName, createDisplay, createDesc, createCategory, createTags, showToast, refreshHubSkills]);

  const handlePublish = useCallback(async () => {
    if (!publishPath.trim() || !publishRepo.trim()) return;
    setBusy(true);
    try {
      const result = await api.publishSkill(publishPath, "github", publishRepo);
      showToast(`Published: ${result.name}`, "success");
      setPublishPath(""); setPublishRepo("");
    } catch (e) {
      showToast(`Publish failed: ${(e as Error).message}`, "error");
    } finally {
      setBusy(false);
    }
  }, [publishPath, publishRepo, showToast]);

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
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <Toast toast={toast} />

      {/* ═══════════════ Header ═══════════════ */}
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <Package className="h-5 w-5 text-muted-foreground" />
          <H2 variant="sm">{t.skills.title}</H2>
          <span className="text-xs text-muted-foreground">
            {t.skills.enabledOf
              .replace("{enabled}", String(enabledCount))
              .replace("{total}", String(skills.length))}
          </span>
        </div>
      </div>

      {/* ═══════════════ Sidebar + Content ═══════════════ */}
      <div
        className="flex flex-col sm:flex-row gap-4"
        style={{ minHeight: "calc(100vh - 180px)" }}
      >
        {/* ---- Sidebar ---- */}
        <div className="sm:w-52 sm:shrink-0">
          <div className="sm:sticky sm:top-[72px] flex flex-col gap-1">
            {/* Search */}
            <div className="relative mb-2 hidden sm:block">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
              <Input
                className="pl-8 h-8 text-xs"
                placeholder={t.common.search}
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
              {search && (
                <button
                  type="button"
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                  onClick={() => setSearch("")}
                >
                  <X className="h-3 w-3" />
                </button>
              )}
            </div>

            {/* Top-level nav */}
            <div className="flex sm:flex-col gap-1 overflow-x-auto sm:overflow-x-visible scrollbar-none pb-1 sm:pb-0">
              <button
                type="button"
                onClick={() => {
                  setView("skills");
                  setSubView("all");
                  setActiveCategory(null);
                  setSearch("");
                }}
                className={`group flex items-center gap-2 px-2.5 py-1.5 text-left text-xs transition-colors cursor-pointer ${
                  view === "skills" && subView === "all" && !isSearching
                    ? "bg-primary/10 text-primary font-medium"
                    : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                }`}
              >
                <Package className="h-3.5 w-3.5 shrink-0" />
                <span className="flex-1 truncate">
                  {t.skills.all} ({skills.length})
                </span>
                {view === "skills" && subView === "all" && !isSearching && (
                  <ChevronRight className="h-3 w-3 text-primary/50 shrink-0" />
                )}
              </button>

              {/* Skill categories (nested under All Skills) */}
              {view === "skills" && subView === "all" &&
                !isSearching &&
                allCategories.map(({ key, name, count }) => {
                  const isActive = activeCategory === key;
                  return (
                    <button
                      key={key}
                      type="button"
                      onClick={() =>
                        setActiveCategory(activeCategory === key ? null : key)
                      }
                      className={`group flex items-center gap-2 px-2.5 py-1 pl-7 text-left text-[11px] transition-colors cursor-pointer ${
                        isActive
                          ? "text-primary font-medium"
                          : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                      }`}
                    >
                      <span className="flex-1 truncate">{name}</span>
                      <span
                        className={`text-[10px] tabular-nums ${isActive ? "text-primary/60" : "text-muted-foreground/50"}`}
                      >
                        {count}
                      </span>
                    </button>
                  );
                })}

              {/* Hub nav items */}
              <Separator className="my-1 hidden sm:block" />

              <SidebarItem
                icon={<Search className="h-3.5 w-3.5 shrink-0" />}
                label="Search Hub"
                active={view === "skills" && subView === "search"}
                onClick={() => { setView("skills"); setSubView("search"); setSearch(""); }}
              />
              <SidebarItem
                icon={<Download className="h-3.5 w-3.5 shrink-0" />}
                label="Browse"
                active={view === "skills" && subView === "browse"}
                onClick={() => { setView("skills"); setSubView("browse"); handleBrowse(); }}
              />
              <SidebarItem
                icon={<RefreshCw className="h-3.5 w-3.5 shrink-0" />}
                label="Updates"
                active={view === "skills" && subView === "updates"}
                onClick={() => { setView("skills"); setSubView("updates"); handleCheckUpdates(); }}
              />
              <SidebarItem
                icon={<ShieldCheck className="h-3.5 w-3.5 shrink-0" />}
                label="Audit"
                active={view === "skills" && subView === "audit"}
                onClick={() => { setView("skills"); setSubView("audit"); handleAudit(); }}
              />
              <SidebarItem
                icon={<Globe2 className="h-3.5 w-3.5 shrink-0" />}
                label="Skill Sources"
                active={view === "skills" && subView === "taps"}
                onClick={() => { setView("skills"); setSubView("taps"); handleLoadTaps(); }}
              />
              <SidebarItem
                icon={<FileUp className="h-3.5 w-3.5 shrink-0" />}
                label="Snapshot"
                active={view === "skills" && subView === "snapshot"}
                onClick={() => { setView("skills"); setSubView("snapshot"); handleExportSnapshot(); }}
              />
              <SidebarItem
                icon={<Plus className="h-3.5 w-3.5 shrink-0" />}
                label="Create"
                active={view === "skills" && subView === "create"}
                onClick={() => { setView("skills"); setSubView("create"); setSearch(""); }}
              />

              <Separator className="my-1 hidden sm:block" />

              <button
                type="button"
                onClick={() => {
                  setView("toolsets");
                  setSearch("");
                }}
                className={`group flex items-center gap-2 px-2.5 py-1.5 text-left text-xs transition-colors cursor-pointer ${
                  view === "toolsets"
                    ? "bg-primary/10 text-primary font-medium"
                    : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                }`}
              >
                <Wrench className="h-3.5 w-3.5 shrink-0" />
                <span className="flex-1 truncate">
                  {t.skills.toolsets} ({toolsets.length})
                </span>
                {view === "toolsets" && (
                  <ChevronRight className="h-3 w-3 text-primary/50 shrink-0" />
                )}
              </button>
            </div>
          </div>
        </div>

        {/* ---- Content ---- */}
        <div className="flex-1 min-w-0">
          {isSearching ? (
            /* Local search results */
            <Card>
              <CardHeader className="py-3 px-4">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Search className="h-4 w-4" />
                    {t.skills.title}
                  </CardTitle>
                  <Badge variant="secondary" className="text-[10px]">
                    {searchMatchedSkills.length} result{searchMatchedSkills.length !== 1 ? "s" : ""}
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
          ) : view === "toolsets" ? (
            /* Toolsets grid */
            <>
              {filteredToolsets.length === 0 ? (
                <Card>
                  <CardContent className="py-8 text-center text-sm text-muted-foreground">
                    {t.skills.noToolsetsMatch}
                  </CardContent>
                </Card>
              ) : (
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  {filteredToolsets.map((ts) => {
                    const TsIcon = toolsetIcon(ts.name);
                    const labelText =
                      ts.label.replace(/^[\p{Emoji}\s]+/u, "").trim() ||
                      ts.name;
                    return (
                      <Card key={ts.name} className="relative">
                        <CardContent className="py-4">
                          <div className="flex items-start gap-3">
                            <TsIcon className="h-5 w-5 text-muted-foreground shrink-0 mt-0.5" />
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2 mb-1">
                                <span className="font-medium text-sm">{labelText}</span>
                                <Badge variant={ts.enabled ? "success" : "outline"} className="text-[10px]">
                                  {ts.enabled ? t.common.active : t.common.inactive}
                                </Badge>
                              </div>
                              <p className="text-xs text-muted-foreground mb-2">{ts.description}</p>
                              {ts.enabled && !ts.configured && (
                                <p className="text-[10px] text-amber-300/80 mb-2">{t.skills.setupNeeded}</p>
                              )}
                              {ts.tools.length > 0 && (
                                <div className="flex flex-wrap gap-1">
                                  {ts.tools.map((tool) => (
                                    <Badge key={tool} variant="secondary" className="text-[10px] font-mono">
                                      {tool}
                                    </Badge>
                                  ))}
                                </div>
                              )}
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    );
                  })}
                </div>
              )}
            </>
          ) : subView === "search" ? (
            /* Hub search */
            <Card>
              <CardHeader className="py-3 px-4">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Search className="h-4 w-4" /> Search Hub
                </CardTitle>
              </CardHeader>
              <CardContent className="px-4 pb-4 space-y-4">
                <div className="flex gap-2">
                  <Input
                    placeholder="Search skills in registry..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                    className="flex-1"
                  />
                  <Button onClick={handleSearch} disabled={busy || !searchQuery.trim()} size="sm">
                    <Search className="h-4 w-4 mr-1" /> Search
                  </Button>
                </div>
                {searchResults.length > 0 && (
                  <div className="grid gap-2">
                    {searchResults.map((r) => (
                      <div key={r.identifier} className="flex items-center justify-between px-3 py-2 hover:bg-muted/40 transition-colors">
                        <div>
                          <span className="font-mono-ui text-sm">{r.name}</span>
                          <p className="text-xs text-muted-foreground">{r.description || "No description"}</p>
                          <div className="flex gap-1 mt-1">
                            <Badge variant="secondary" className="text-[10px]">{r.source}</Badge>
                            <Badge variant="outline" className="text-[10px]">{r.trust}</Badge>
                          </div>
                        </div>
                        <Button size="sm" variant="outline" onClick={() => { setInstallId(r.identifier); }}>
                          <Download className="h-3 w-3 mr-1" /> Install
                        </Button>
                      </div>
                    ))}
                  </div>
                )}
                {searchResults.length === 0 && searchQuery && !busy && (
                  <p className="text-sm text-muted-foreground text-center py-4">No results. Try a different query.</p>
                )}
              </CardContent>
            </Card>
          ) : subView === "browse" ? (
            <Card>
              <CardHeader className="py-3 px-4">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Download className="h-4 w-4" /> Browse Available Skills
                  </CardTitle>
                  <Button size="sm" variant="outline" onClick={handleBrowse} disabled={busy}>
                    <RefreshCw className={`h-3 w-3 mr-1 ${busy ? "animate-spin" : ""}`} /> Refresh
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="px-4 pb-4">
                {browseResults.length === 0 && !busy && (
                  <p className="text-sm text-muted-foreground text-center py-4">No skills available.</p>
                )}
                <div className="grid gap-2">
                  {browseResults.map((r) => (
                    <div key={r.identifier} className="flex items-center justify-between px-3 py-2 hover:bg-muted/40 transition-colors">
                      <div>
                        <span className="font-mono-ui text-sm">{r.name}</span>
                        <p className="text-xs text-muted-foreground">{r.description || "No description"}</p>
                      </div>
                      <Button size="sm" variant="outline" onClick={() => setInstallId(r.identifier)}>
                        <Download className="h-3 w-3 mr-1" /> Install
                      </Button>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          ) : subView === "updates" ? (
            <Card>
              <CardHeader className="py-3 px-4">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <RefreshCw className="h-4 w-4" /> Skill Updates
                  </CardTitle>
                  <Button size="sm" variant="outline" onClick={handleUpdateAll} disabled={busy}>
                    <Upload className="h-3 w-3 mr-1" /> Update All
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="px-4 pb-4">
                {updateChecks.length === 0 && !busy && (
                  <p className="text-sm text-muted-foreground text-center py-4">No hub-installed skills to check.</p>
                )}
                <div className="grid gap-2">
                  {updateChecks.map((u) => (
                    <div key={u.name} className="flex items-center justify-between px-3 py-2 hover:bg-muted/40 transition-colors">
                      <div>
                        <span className="font-mono-ui text-sm">{u.name}</span>
                        <Badge variant={u.status === "update_available" ? "destructive" : "secondary"} className="text-[10px] ml-2">
                          {u.status}
                        </Badge>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          ) : subView === "audit" ? (
            <Card>
              <CardHeader className="py-3 px-4">
                <CardTitle className="text-sm flex items-center gap-2">
                  <ShieldCheck className="h-4 w-4" /> Security Audit
                </CardTitle>
              </CardHeader>
              <CardContent className="px-4 pb-4">
                {auditResults.length === 0 && !busy && (
                  <p className="text-sm text-muted-foreground text-center py-4">No hub-installed skills to audit.</p>
                )}
                <div className="grid gap-2">
                  {auditResults.map((a) => (
                    <div key={a.name} className="px-3 py-2 hover:bg-muted/40 transition-colors">
                      <div className="flex items-center gap-2">
                        <span className="font-mono-ui text-sm">{a.name}</span>
                        <Badge
                          variant={a.verdict === "safe" ? "success" : a.verdict === "caution" ? "secondary" : "destructive"}
                          className="text-[10px]"
                        >
                          {a.verdict}
                        </Badge>
                        <span className="text-[10px] text-muted-foreground">{a.findings_count} findings</span>
                      </div>
                      {a.findings.length > 0 && (
                        <div className="mt-1 ml-4 space-y-0.5">
                          {a.findings.map((f, i) => (
                            <p key={i} className="text-[11px] text-muted-foreground">
                              <Badge variant="outline" className="text-[9px] mr-1">{f.severity}</Badge>
                              {f.description}
                            </p>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          ) : subView === "taps" ? (
            <Card>
              <CardHeader className="py-3 px-4">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Globe2 className="h-4 w-4" /> Skill Sources (Taps)
                </CardTitle>
              </CardHeader>
              <CardContent className="px-4 pb-4 space-y-4">
                <div className="flex gap-2">
                  <Input
                    placeholder="GitHub repo URL (e.g. user/skills-repo)"
                    value={tapRepo}
                    onChange={(e) => setTapRepo(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleAddTap()}
                    className="flex-1"
                  />
                  <Button onClick={handleAddTap} disabled={busy || !tapRepo.trim()} size="sm">
                    <Plus className="h-3 w-3 mr-1" /> Add
                  </Button>
                </div>
                {taps.length === 0 && !busy && (
                  <p className="text-sm text-muted-foreground text-center py-4">No custom skill sources configured.</p>
                )}
                <div className="grid gap-2">
                  {taps.map((tap) => (
                    <div key={tap.repo} className="flex items-center justify-between px-3 py-2 hover:bg-muted/40 transition-colors">
                      <div>
                        <span className="font-mono-ui text-sm">{tap.repo}</span>
                        <span className="text-[10px] text-muted-foreground ml-2">/{tap.path}</span>
                      </div>
                      <Button size="sm" variant="ghost" onClick={() => handleRemoveTap(tap.repo)}>
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          ) : subView === "snapshot" ? (
            <Card>
              <CardHeader className="py-3 px-4">
                <CardTitle className="text-sm flex items-center gap-2">
                  <FileUp className="h-4 w-4" /> Snapshot Export / Import
                </CardTitle>
              </CardHeader>
              <CardContent className="px-4 pb-4 space-y-4">
                <div className="flex gap-2">
                  <Button onClick={handleExportSnapshot} disabled={busy} size="sm">
                    <Download className="h-3 w-3 mr-1" /> Export
                  </Button>
                  <Button onClick={handleImportSnapshot} disabled={busy || !snapshotData} size="sm" variant="outline">
                    <Upload className="h-3 w-3 mr-1" /> Import
                  </Button>
                </div>
                {snapshotData && (
                  <div className="relative">
                    <pre className="bg-muted/30 rounded p-3 text-[11px] font-mono overflow-auto max-h-64">
                      {JSON.stringify(snapshotData, null, 2)}
                    </pre>
                  </div>
                )}
              </CardContent>
            </Card>
          ) : subView === "create" ? (
            <Card>
              <CardHeader className="py-3 px-4">
                <CardTitle className="text-sm flex items-center gap-2">
                  <Plus className="h-4 w-4" /> Create Skill
                </CardTitle>
              </CardHeader>
              <CardContent className="px-4 pb-4 space-y-3">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <Label className="text-xs mb-1">Skill Name *</Label>
                    <Input placeholder="my-skill" value={createName} onChange={(e) => setCreateName(e.target.value)} />
                  </div>
                  <div>
                    <Label className="text-xs mb-1">Display Name</Label>
                    <Input placeholder="My Skill" value={createDisplay} onChange={(e) => setCreateDisplay(e.target.value)} />
                  </div>
                </div>
                <div>
                  <Label className="text-xs mb-1">Description</Label>
                  <Input placeholder="What this skill does..." value={createDesc} onChange={(e) => setCreateDesc(e.target.value)} />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <Label className="text-xs mb-1">Category</Label>
                    <Input placeholder="mlops" value={createCategory} onChange={(e) => setCreateCategory(e.target.value)} />
                  </div>
                  <div>
                    <Label className="text-xs mb-1">Tags (comma-separated)</Label>
                    <Input placeholder="python, ml" value={createTags} onChange={(e) => setCreateTags(e.target.value)} />
                  </div>
                </div>
                <Separator />
                <div className="flex items-center gap-3">
                  <Button onClick={handleCreateSkill} disabled={busy || !createName.trim()} size="sm">
                    <Plus className="h-3 w-3 mr-1" /> Create
                  </Button>
                  <Separator orientation="vertical" className="h-6" />
                  <Input
                    placeholder="Skill path to publish..."
                    value={publishPath}
                    onChange={(e) => setPublishPath(e.target.value)}
                    className="flex-1"
                  />
                  <Input
                    placeholder="GitHub repo"
                    value={publishRepo}
                    onChange={(e) => setPublishRepo(e.target.value)}
                    className="w-48"
                  />
                  <Button onClick={handlePublish} disabled={busy || !publishPath.trim() || !publishRepo.trim()} size="sm" variant="outline">
                    <Rocket className="h-3 w-3 mr-1" /> Publish
                  </Button>
                </div>
              </CardContent>
            </Card>
          ) : (
            /* Default: Skills list (with install bar) */
            <Card>
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
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary" className="text-[10px]">
                      {activeSkills.length} skill{activeSkills.length !== 1 ? "s" : ""}
                    </Badge>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="px-4 pb-4">
                {/* Quick install bar */}
                <div className="flex gap-2 mb-3">
                  <Input
                    placeholder="Install skill (e.g. nous-research/skill-name)..."
                    value={installId}
                    onChange={(e) => setInstallId(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleInstall()}
                    className="flex-1 h-8 text-xs"
                  />
                  <Button onClick={handleInstall} disabled={busy || !installId.trim()} size="sm">
                    <Download className="h-3 w-3 mr-1" /> Install
                  </Button>
                </div>

                {activeSkills.length === 0 ? (
                  <p className="text-sm text-muted-foreground text-center py-8">
                    {skills.length === 0 ? t.skills.noSkills : t.skills.noSkillsMatch}
                  </p>
                ) : (
                  <div className="grid gap-1">
                    {activeSkills.map((skill) => {
                      const hub = hubSkills.find((h) => h.name === skill.name);
                      return (
                        <SkillRow
                          key={skill.name}
                          skill={skill}
                          toggling={togglingSkills.has(skill.name)}
                          onToggle={() => handleToggleSkill(skill)}
                          noDescriptionLabel={t.skills.noDescription}
                          hubType={hub?.type}
                          isHubInstalled={hub?.type === "hub"}
                          onUninstall={() => hub && handleUninstall(skill.name)}
                        />
                      );
                    })}
                  </div>
                )}
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Sub-components                                                     */
/* ------------------------------------------------------------------ */

function SidebarItem({
  icon,
  label,
  active,
  onClick,
}: {
  icon: React.ReactNode;
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`group flex items-center gap-2 px-2.5 py-1.5 text-left text-xs transition-colors cursor-pointer ${
        active
          ? "bg-primary/10 text-primary font-medium"
          : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
      }`}
    >
      {icon}
      <span className="flex-1 truncate">{label}</span>
      {active && <ChevronRight className="h-3 w-3 text-primary/50 shrink-0" />}
    </button>
  );
}

function SkillRow({
  skill,
  toggling,
  onToggle,
  noDescriptionLabel,
  hubType,
  isHubInstalled,
  onUninstall,
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
          {hubType && (
            <Badge variant={hubType === "hub" ? "default" : "outline"} className="text-[9px]">
              {hubType}
            </Badge>
          )}
        </div>
        <p className="text-xs text-muted-foreground leading-relaxed line-clamp-2">
          {skill.description || noDescriptionLabel}
        </p>
      </div>
      {isHubInstalled && onUninstall && (
        <Button
          size="sm"
          variant="ghost"
          className="opacity-0 group-hover:opacity-100 transition-opacity shrink-0"
          onClick={onUninstall}
        >
          <Trash2 className="h-3 w-3" />
        </Button>
      )}
    </div>
  );
}

interface SkillRowProps {
  skill: SkillInfo;
  toggling: boolean;
  onToggle: () => void;
  noDescriptionLabel: string;
  hubType?: string;
  isHubInstalled?: boolean;
  onUninstall?: () => void;
}
