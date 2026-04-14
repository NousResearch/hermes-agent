import { useEffect, useState, useMemo } from "react";
import {
  Blocks,
  Bot,
  BrainCircuit,
  ChevronRight,
  Code,
  Database,
  FileCode,
  FileSearch,
  Globe,
  Image,
  LayoutDashboard,
  Monitor,
  Package,
  Paintbrush,
  Search,
  Server,
  Shield,
  Sparkles,
  Terminal,
  Wrench,
  X,
} from "lucide-react";
import type { ComponentType } from "react";
import { api } from "@/lib/api";
import type { SkillInfo, ToolsetInfo } from "@/lib/api";
import { useToast } from "@/hooks/useToast";
import { Toast } from "@/components/Toast";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";

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

function prettyCategory(raw: string | null | undefined): string {
  if (!raw) return "General";
  if (CATEGORY_LABELS[raw]) return CATEGORY_LABELS[raw];
  return raw
    .split(/[-_/]/)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

const TOOLSET_ICONS: Record<string, ComponentType<{ className?: string }>> = {
  terminal: Terminal,
  shell: Terminal,
  browser: Globe,
  web: Globe,
  code: Code,
  coding: Code,
  python: FileCode,
  files: FileSearch,
  file: FileSearch,
  search: Search,
  image: Image,
  vision: Image,
  memory: BrainCircuit,
  database: Database,
  db: Database,
  mcp: Blocks,
  ai: Sparkles,
  agent: Bot,
  security: Shield,
  server: Server,
  deploy: Server,
  ui: Paintbrush,
  ux: LayoutDashboard,
  display: Monitor,
};

function toolsetIcon(name: string, label: string): ComponentType<{ className?: string }> {
  const lower = name.toLowerCase();
  if (TOOLSET_ICONS[lower]) return TOOLSET_ICONS[lower];
  for (const [key, icon] of Object.entries(TOOLSET_ICONS)) {
    if (lower.includes(key) || label.toLowerCase().includes(key)) return icon;
  }
  return Wrench;
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function SkillsPage() {
  const [view, setView] = useState<"skills" | "toolsets">("skills");
  const [skills, setSkills] = useState<SkillInfo[]>([]);
  const [toolsets, setToolsets] = useState<ToolsetInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [activeCategory, setActiveCategory] = useState<string | null>(null);
  const [togglingSkills, setTogglingSkills] = useState<Set<string>>(new Set());
  const { toast, showToast } = useToast();

  useEffect(() => {
    Promise.all([api.getSkills(), api.getToolsets()])
      .then(([s, t]) => {
        setSkills(s);
        setToolsets(t);
      })
      .catch(() => showToast("Failed to load skills/toolsets", "error"))
      .finally(() => setLoading(false));
  }, []);

  /* ---- Toggle skill ---- */
  const handleToggleSkill = async (skill: SkillInfo) => {
    setTogglingSkills((prev) => new Set(prev).add(skill.name));
    try {
      await api.toggleSkill(skill.name, !skill.enabled);
      setSkills((prev) =>
        prev.map((s) =>
          s.name === skill.name ? { ...s, enabled: !s.enabled } : s
        )
      );
      showToast(
        `${skill.name} ${skill.enabled ? "disabled" : "enabled"}`,
        "success"
      );
    } catch {
      showToast(`Failed to toggle ${skill.name}`, "error");
    } finally {
      setTogglingSkills((prev) => {
        const next = new Set(prev);
        next.delete(skill.name);
        return next;
      });
    }
  };

  /* ---- Derived data ---- */
  const lowerSearch = search.toLowerCase();

  const filteredSkills = useMemo(() => {
    return skills.filter((s) => {
      const matchesSearch =
        !search ||
        s.name.toLowerCase().includes(lowerSearch) ||
        s.description.toLowerCase().includes(lowerSearch) ||
        (s.category ?? "").toLowerCase().includes(lowerSearch);
      const matchesCategory =
        !activeCategory ||
        (activeCategory === "__none__" ? !s.category : s.category === activeCategory);
      return matchesSearch && matchesCategory;
    });
  }, [skills, search, lowerSearch, activeCategory]);

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
      .map(([key, count]) => ({ key, name: prettyCategory(key === "__none__" ? null : key), count }));
  }, [skills]);

  const enabledCount = skills.filter((s) => s.enabled).length;

  const filteredToolsets = useMemo(() => {
    return toolsets.filter(
      (t) =>
        !search ||
        t.name.toLowerCase().includes(lowerSearch) ||
        t.label.toLowerCase().includes(lowerSearch) ||
        t.description.toLowerCase().includes(lowerSearch)
    );
  }, [toolsets, search, lowerSearch]);

  const isSearching = search.trim().length > 0;

  const activeToolsetCount = toolsets.filter((t) => t.enabled).length;

  const searchMatchedSkills = useMemo(() => {
    if (!isSearching) return [];
    return skills.filter(
      (s) =>
        s.name.toLowerCase().includes(lowerSearch) ||
        s.description.toLowerCase().includes(lowerSearch) ||
        (s.category ?? "").toLowerCase().includes(lowerSearch),
    );
  }, [isSearching, skills, lowerSearch]);

  const activeSkills = useMemo(() => {
    if (isSearching) return [];
    return [...filteredSkills].sort((a, b) => a.name.localeCompare(b.name));
  }, [isSearching, filteredSkills]);

  /* ---- Loading ---- */
  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
      </div>
    );
  }

  const activeCategoryName = activeCategory
    ? prettyCategory(activeCategory === "__none__" ? null : activeCategory)
    : "All Skills";

  const renderSkillList = (list: SkillInfo[]) => (
    <div className="grid gap-1">
      {list.map((skill) => (
        <div
          key={skill.name}
          className="group flex items-start gap-3 px-3 py-2.5 transition-colors hover:bg-muted/40"
        >
          <div className="pt-0.5 shrink-0">
            <Switch
              checked={skill.enabled}
              onCheckedChange={() => handleToggleSkill(skill)}
              disabled={togglingSkills.has(skill.name)}
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
              {skill.description || "No description available."}
            </p>
          </div>
        </div>
      ))}
    </div>
  );

  return (
    <div className="flex flex-col gap-4">
      <Toast toast={toast} />

      {/* ═══════════════ Header ═══════════════ */}
      <div className="flex items-center gap-3">
        {view === "skills" ? (
          <Package className="h-4 w-4 text-muted-foreground" />
        ) : (
          <Wrench className="h-4 w-4 text-muted-foreground" />
        )}
        <span className="text-xs text-muted-foreground">
          {view === "skills"
            ? `${enabledCount}/${skills.length} skills enabled`
            : `${activeToolsetCount}/${toolsets.length} toolsets active`}
        </span>
      </div>

      {/* ═══════════════ Sidebar + Content ═══════════════ */}
      <div className="flex flex-col sm:flex-row gap-4" style={{ minHeight: "calc(100vh - 180px)" }}>
        {/* ---- Sidebar ---- */}
        <div className="sm:w-52 sm:shrink-0">
          <div className="sm:sticky sm:top-[72px] flex flex-col gap-1">
            {/* Search */}
            <div className="relative mb-2 hidden sm:block">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
              <Input
                className="pl-8 h-8 text-xs"
                placeholder="Search..."
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

            {/* Nav items */}
            <div className="flex sm:flex-col gap-1 overflow-x-auto sm:overflow-x-visible scrollbar-none pb-1 sm:pb-0">
              {/* Skills top-level */}
              <button
                type="button"
                onClick={() => {
                  setView("skills");
                  setActiveCategory(null);
                  setSearch("");
                }}
                className={`group flex items-center gap-2 px-2.5 py-1.5 text-left text-xs transition-colors cursor-pointer ${
                  view === "skills" && !activeCategory && !isSearching
                    ? "bg-primary/10 text-primary font-medium"
                    : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                }`}
              >
                <Package className="h-4 w-4 shrink-0" />
                <span className="flex-1 truncate">All Skills</span>
                <span className={`text-[10px] tabular-nums ${
                  view === "skills" && !activeCategory && !isSearching
                    ? "text-primary/60"
                    : "text-muted-foreground/50"
                }`}>
                  {skills.length}
                </span>
                {view === "skills" && !activeCategory && !isSearching && (
                  <ChevronRight className="h-3 w-3 text-primary/50 shrink-0" />
                )}
              </button>

              {/* Skill category sub-items */}
              {allCategories.map(({ key, name, count }) => {
                const isActive = view === "skills" && activeCategory === key && !isSearching;
                return (
                  <button
                    key={key}
                    type="button"
                    onClick={() => {
                      setView("skills");
                      setActiveCategory(key);
                      setSearch("");
                    }}
                    className={`group flex items-center gap-2 sm:pl-6 px-2.5 py-1.5 text-left text-xs transition-colors cursor-pointer ${
                      isActive
                        ? "bg-primary/10 text-primary font-medium"
                        : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                    }`}
                  >
                    <span className="flex-1 truncate">{name}</span>
                    <span className={`text-[10px] tabular-nums ${
                      isActive ? "text-primary/60" : "text-muted-foreground/50"
                    }`}>
                      {count}
                    </span>
                    {isActive && (
                      <ChevronRight className="h-3 w-3 text-primary/50 shrink-0" />
                    )}
                  </button>
                );
              })}

              {/* Divider */}
              <div className="hidden sm:block border-t border-border my-1" />

              {/* Toolsets top-level */}
              <button
                type="button"
                onClick={() => {
                  setView("toolsets");
                  setSearch("");
                }}
                className={`group flex items-center gap-2 px-2.5 py-1.5 text-left text-xs transition-colors cursor-pointer ${
                  view === "toolsets" && !isSearching
                    ? "bg-primary/10 text-primary font-medium"
                    : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                }`}
              >
                <Wrench className="h-4 w-4 shrink-0" />
                <span className="flex-1 truncate">Toolsets</span>
                <span className={`text-[10px] tabular-nums ${
                  view === "toolsets" && !isSearching
                    ? "text-primary/60"
                    : "text-muted-foreground/50"
                }`}>
                  {toolsets.length}
                </span>
                {view === "toolsets" && !isSearching && (
                  <ChevronRight className="h-3 w-3 text-primary/50 shrink-0" />
                )}
              </button>
            </div>
          </div>
        </div>

        {/* ---- Content ---- */}
        <div className="flex-1 min-w-0">
          {/* Search results (across both skills and toolsets) */}
          {isSearching ? (
            <Card>
              <CardHeader className="py-3 px-4">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Search className="h-4 w-4" />
                    Search Results
                  </CardTitle>
                  <Badge variant="secondary" className="text-[10px]">
                    {searchMatchedSkills.length} skill{searchMatchedSkills.length !== 1 ? "s" : ""}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="px-4 pb-4">
                {searchMatchedSkills.length === 0 ? (
                  <p className="text-sm text-muted-foreground text-center py-8">
                    No skills match &ldquo;<span className="text-foreground">{search}</span>&rdquo;
                  </p>
                ) : (
                  renderSkillList(searchMatchedSkills)
                )}
              </CardContent>
            </Card>

          ) : view === "skills" ? (
            /* ---- Skills view ---- */
            <Card>
              <CardHeader className="py-3 px-4">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Package className="h-4 w-4" />
                    {activeCategoryName}
                  </CardTitle>
                  <Badge variant="secondary" className="text-[10px]">
                    {activeSkills.length} skill{activeSkills.length !== 1 ? "s" : ""}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="px-4 pb-4">
                {activeSkills.length === 0 ? (
                  <p className="text-sm text-muted-foreground text-center py-8">
                    {skills.length === 0
                      ? "No skills found. Skills are loaded from ~/.hermes/skills/"
                      : "No skills in this category."}
                  </p>
                ) : (
                  renderSkillList(activeSkills)
                )}
              </CardContent>
            </Card>

          ) : (
            /* ---- Toolsets view ---- */
            <>
              {filteredToolsets.length === 0 ? (
                <Card>
                  <CardContent className="py-8 text-center text-sm text-muted-foreground">
                    No toolsets found.
                  </CardContent>
                </Card>
              ) : (
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  {filteredToolsets.map((ts) => {
                    const labelText = ts.label.replace(/^[\p{Emoji}\s]+/u, "").trim() || ts.name;
                    const TsIcon = toolsetIcon(ts.name, ts.label);

                    return (
                      <Card key={ts.name}>
                        <CardContent className="py-4">
                          <div className="flex items-start gap-3">
                            <TsIcon className="h-5 w-5 shrink-0 mt-0.5 text-muted-foreground" />
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2 mb-1">
                                <span className="font-medium text-sm">{labelText}</span>
                                <Badge
                                  variant={ts.enabled ? "success" : "outline"}
                                  className="text-[10px]"
                                >
                                  {ts.enabled ? "active" : "inactive"}
                                </Badge>
                              </div>
                              <p className="text-xs text-muted-foreground mb-2">
                                {ts.description}
                              </p>
                              {ts.enabled && !ts.configured && (
                                <p className="text-[10px] text-amber-300/80 mb-2">
                                  Setup needed
                                </p>
                              )}
                              {ts.tools.length > 0 && (
                                <div className="flex flex-wrap gap-1">
                                  {ts.tools.map((tool) => (
                                    <Badge
                                      key={tool}
                                      variant="secondary"
                                      className="text-[10px] font-mono"
                                    >
                                      {tool}
                                    </Badge>
                                  ))}
                                </div>
                              )}
                              {ts.tools.length === 0 && (
                                <span className="text-[10px] text-muted-foreground/60">
                                  {ts.enabled ? `${ts.name} toolset` : "Disabled for CLI"}
                                </span>
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
          )}
        </div>
      </div>
    </div>
  );
}
