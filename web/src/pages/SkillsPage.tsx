import { useEffect, useLayoutEffect, useState, useMemo } from "react";
import {
  ChevronDown,
  ChevronRight,
  Package,
  PowerOff,
  Search,
  Wrench,
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
  Filter,
} from "lucide-react";
import { api } from "@/lib/api";
import type { SkillInfo, ToolsetInfo } from "@/lib/api";
import { useToast } from "@/hooks/useToast";
import { Toast } from "@/components/Toast";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { ListItem } from "@nous-research/ui/ui/components/list-item";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Switch } from "@nous-research/ui/ui/components/switch";
import { cn } from "@/lib/utils";
import { Input } from "@/components/ui/input";
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
  const [view, setView] = useState<"skills" | "toolsets">("skills");
  const [activeCategory, setActiveCategory] = useState<string | null>(null);
  const [collapsedCategories, setCollapsedCategories] = useState<Set<string>>(
    new Set(),
  );
  const [togglingSkills, setTogglingSkills] = useState<Set<string>>(new Set());
  const [bulkTogglingCategories, setBulkTogglingCategories] = useState<
    Set<string>
  >(new Set());
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
  }, [showToast, t.common.loading]);

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

  const handleDisableSkillGroup = async (
    groupKey: string,
    groupName: string,
    groupSkills: SkillInfo[],
  ) => {
    const enabledSkills = groupSkills.filter((skill) => skill.enabled);
    if (enabledSkills.length === 0) return;

    setBulkTogglingCategories((prev) => new Set(prev).add(groupKey));
    setTogglingSkills((prev) => {
      const next = new Set(prev);
      for (const skill of enabledSkills) next.add(skill.name);
      return next;
    });

    try {
      const results = await Promise.allSettled(
        enabledSkills.map(async (skill) => {
          await api.toggleSkill(skill.name, false);
          return skill.name;
        }),
      );
      const disabledNames = new Set(
        results
          .filter((result): result is PromiseFulfilledResult<string> =>
            result.status === "fulfilled",
          )
          .map((result) => result.value),
      );
      const failedCount = results.length - disabledNames.size;

      if (disabledNames.size > 0) {
        setSkills((prev) =>
          prev.map((skill) =>
            disabledNames.has(skill.name) ? { ...skill, enabled: false } : skill,
          ),
        );
      }

      if (failedCount > 0) {
        showToast(
          `Disabled ${disabledNames.size} skill${
            disabledNames.size === 1 ? "" : "s"
          } in ${groupName}; ${failedCount} failed`,
          "error",
        );
      } else {
        showToast(
          `Disabled ${disabledNames.size} skill${
            disabledNames.size === 1 ? "" : "s"
          } in ${groupName}`,
          "success",
        );
      }
    } finally {
      setBulkTogglingCategories((prev) => {
        const next = new Set(prev);
        next.delete(groupKey);
        return next;
      });
      setTogglingSkills((prev) => {
        const next = new Set(prev);
        for (const skill of enabledSkills) next.delete(skill.name);
        return next;
      });
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

  const skillGroups = useMemo(() => {
    const groups = new Map<string, SkillInfo[]>();
    for (const skill of activeSkills) {
      const key = skill.category || "__none__";
      const group = groups.get(key) ?? [];
      group.push(skill);
      groups.set(key, group);
    }

    return [...groups.entries()]
      .sort((a, b) => {
        if (a[0] === "__none__") return -1;
        if (b[0] === "__none__") return 1;
        return a[0].localeCompare(b[0]);
      })
      .map(([key, groupSkills]) => ({
        key,
        name: prettyCategory(key === "__none__" ? null : key, t.common.general),
        skills: groupSkills.sort((a, b) => a.name.localeCompare(b.name)),
      }));
  }, [activeSkills, t]);

  const visibleGroupKeys = useMemo(
    () => skillGroups.map((group) => group.key),
    [skillGroups],
  );

  const handleCollapseAllSkillGroups = () => {
    setCollapsedCategories((prev) => {
      const next = new Set(prev);
      for (const key of visibleGroupKeys) next.add(key);
      return next;
    });
  };

  const handleExpandAllSkillGroups = () => {
    setCollapsedCategories((prev) => {
      const next = new Set(prev);
      for (const key of visibleGroupKeys) next.delete(key);
      return next;
    });
  };

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
          className="h-8 pl-8 pr-7 text-xs"
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
            <div
              className={`
                flex flex-col
                border border-border bg-muted/20
              `}
            >
              <div className="hidden sm:flex items-center gap-2 px-3 py-2 border-b border-border">
                <Filter className="h-3 w-3 text-muted-foreground" />
                <span className="font-mondwest text-[0.65rem] tracking-[0.12em] uppercase text-muted-foreground">
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
              </div>

              {view === "skills" &&
                !isSearching &&
                allCategories.length > 0 && (
                  <div className="hidden sm:flex flex-col border-t border-border">
                    <div className="px-3 pt-2 pb-1 font-mondwest text-[0.6rem] tracking-[0.12em] uppercase text-muted-foreground/70">
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
                            className="rounded-sm px-2 py-1 text-[11px]"
                          >
                            <span className="flex-1 truncate">{name}</span>
                            <span
                              className={`text-[10px] tabular-nums ${
                                isActive
                                  ? "text-foreground/60"
                                  : "text-muted-foreground/50"
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
            <Card>
              <CardHeader className="py-3 px-4">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Search className="h-4 w-4" />
                    {t.skills.title}
                  </CardTitle>
                  <Badge tone="secondary" className="text-[10px]">
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
                  <div className="flex flex-wrap items-center justify-end gap-2">
                    {!activeCategory && skillGroups.length > 1 && (
                      <>
                        <Button
                          ghost
                          size="sm"
                          onClick={handleCollapseAllSkillGroups}
                        >
                          {t.common.collapse} {t.skills.all}
                        </Button>
                        <Button
                          ghost
                          size="sm"
                          onClick={handleExpandAllSkillGroups}
                        >
                          {t.common.expand} {t.skills.all}
                        </Button>
                      </>
                    )}
                    <Badge tone="secondary" className="text-[10px]">
                      {t.skills.skillCount
                        .replace("{count}", String(activeSkills.length))
                        .replace("{s}", activeSkills.length !== 1 ? "s" : "")}
                    </Badge>
                  </div>
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
                  <div className="grid gap-3">
                    {skillGroups.map((group) => {
                      const collapsed = collapsedCategories.has(group.key);
                      return (
                        <SkillGroupCard
                          key={group.key}
                          collapsed={collapsed}
                          disabling={bulkTogglingCategories.has(group.key)}
                          group={group}
                          noDescriptionLabel={t.skills.noDescription}
                          onDisable={() =>
                            handleDisableSkillGroup(
                              group.key,
                              group.name,
                              group.skills,
                            )
                          }
                          onToggleCollapsed={() => {
                            setCollapsedCategories((prev) => {
                              const next = new Set(prev);
                              if (next.has(group.key)) next.delete(group.key);
                              else next.add(group.key);
                              return next;
                            });
                          }}
                          onToggleSkill={handleToggleSkill}
                          togglingSkills={togglingSkills}
                        />
                      );
                    })}
                  </div>
                )}
              </CardContent>
            </Card>
          ) : (
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
                                <span className="font-medium text-sm">
                                  {labelText}
                                </span>
                                <Badge
                                  tone={ts.enabled ? "success" : "outline"}
                                  className="text-[10px]"
                                >
                                  {ts.enabled
                                    ? t.common.active
                                    : t.common.inactive}
                                </Badge>
                              </div>
                              <p className="text-xs text-muted-foreground mb-2">
                                {ts.description}
                              </p>
                              {ts.enabled && !ts.configured && (
                                <p className="text-[10px] text-amber-300/80 mb-2">
                                  {t.skills.setupNeeded}
                                </p>
                              )}
                              {ts.tools.length > 0 && (
                                <div className="flex flex-wrap gap-1">
                                  {ts.tools.map((tool) => (
                                    <Badge
                                      key={tool}
                                      tone="secondary"
                                      className="text-[10px] font-mono"
                                    >
                                      {tool}
                                    </Badge>
                                  ))}
                                </div>
                              )}
                              {ts.tools.length === 0 && (
                                <span className="text-[10px] text-muted-foreground/60">
                                  {ts.enabled
                                    ? t.skills.toolsetLabel.replace(
                                        "{name}",
                                        ts.name,
                                      )
                                    : t.skills.disabledForCli}
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
      <PluginSlot name="skills:bottom" />
    </div>
  );
}

function SkillGroupCard({
  collapsed,
  disabling,
  group,
  noDescriptionLabel,
  onDisable,
  onToggleCollapsed,
  onToggleSkill,
  togglingSkills,
}: SkillGroupCardProps) {
  const enabledCount = group.skills.filter((skill) => skill.enabled).length;
  const totalCount = group.skills.length;
  const Chevron = collapsed ? ChevronRight : ChevronDown;

  return (
    <div className="rounded-none border border-border bg-background/40">
      <div className="flex flex-wrap items-center gap-2 border-b border-border px-3 py-2">
        <button
          type="button"
          onClick={onToggleCollapsed}
          aria-expanded={!collapsed}
          className="flex min-w-0 flex-1 items-center gap-2 text-left"
        >
          <Chevron className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
          <span className="truncate font-mondwest text-display text-xs tracking-[0.12em] text-text-secondary uppercase">
            {group.name}
          </span>
          <Badge tone="secondary" className="text-xs tabular-nums">
            {enabledCount}/{totalCount} enabled
          </Badge>
        </button>
        <Button
          ghost
          size="sm"
          disabled={disabling || enabledCount === 0}
          onClick={onDisable}
        >
          <PowerOff className="h-3.5 w-3.5" />
          Turn off
        </Button>
      </div>
      {!collapsed && (
        <div className="grid gap-1">
          {group.skills.map((skill) => (
            <SkillRow
              key={skill.name}
              skill={skill}
              toggling={togglingSkills.has(skill.name)}
              onToggle={() => onToggleSkill(skill)}
              noDescriptionLabel={noDescriptionLabel}
            />
          ))}
        </div>
      )}
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
        "rounded-sm whitespace-nowrap px-2.5 py-1.5",
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

interface SkillGroup {
  key: string;
  name: string;
  skills: SkillInfo[];
}

interface SkillGroupCardProps {
  collapsed: boolean;
  disabling: boolean;
  group: SkillGroup;
  noDescriptionLabel: string;
  onDisable: () => void;
  onToggleCollapsed: () => void;
  onToggleSkill: (skill: SkillInfo) => void;
  togglingSkills: Set<string>;
}
