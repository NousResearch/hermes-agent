import { useEffect, useLayoutEffect, useRef, useState, useMemo } from "react";
import {
  Code,
  Download,
  FormInput,
  RotateCcw,
  Search,
  Upload,
  X,
  Settings2,
  FileText,
  Settings,
  Bot,
  Monitor,
  Palette,
  Users,
  Brain,
  Package,
  Lock,
  Globe,
  Mic,
  Volume2,
  Ear,
  ClipboardList,
  MessageCircle,
  Wrench,
  FileQuestion,
  Filter,
  Cloud,
  Sparkles,
  LayoutDashboard,
  BookOpen,
  Route,
  History,
  Shield,
  FileOutput,
  RefreshCw,
  Trash2,
  Copy,
  Pencil,
  CircleHelp,
} from "lucide-react";
import { api } from "@/lib/api";
import { getNestedValue, setNestedValue } from "@/lib/nested";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { AutoField } from "@/components/AutoField";
import { Button } from "@nous-research/ui/ui/components/button";
import { ListItem } from "@nous-research/ui/ui/components/list-item";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { ConfirmDialog } from "@nous-research/ui/ui/components/confirm-dialog";
import { Input } from "@nous-research/ui/ui/components/input";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { useI18n } from "@/i18n";
import { usePageHeader } from "@/contexts/usePageHeader";
import { PluginSlot } from "@/plugins";

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

const CATEGORY_ICONS: Record<
  string,
  React.ComponentType<{ className?: string }>
> = {
  general: Settings,
  agent: Bot,
  terminal: Monitor,
  display: Palette,
  delegation: Users,
  memory: Brain,
  compression: Package,
  security: Lock,
  browser: Globe,
  voice: Mic,
  tts: Volume2,
  stt: Ear,
  logging: ClipboardList,
  discord: MessageCircle,
  auxiliary: Wrench,
  bedrock: Cloud,
  curator: Sparkles,
  kanban: LayoutDashboard,
  model_catalog: BookOpen,
  openrouter: Route,
  sessions: History,
  tool_loop_guardrails: Shield,
  tool_output: FileOutput,
  updates: RefreshCw,
};

function CategoryIcon({
  category,
  className,
}: {
  category: string;
  className?: string;
}) {
  const Icon = CATEGORY_ICONS[category] ?? FileQuestion;
  return <Icon className={className ?? "h-4 w-4"} />;
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

type AllowlistEntry = {
  pattern: string;
  kind: "manual" | "danger_category";
};

type AllowlistTestResult = {
  command: string;
  matched: boolean;
  matched_pattern: string | null;
  matched_kind: "manual" | "danger_category" | null;
  blocked_by_shell_operator: boolean;
};

export default function ConfigPage() {
  const [config, setConfig] = useState<Record<string, unknown> | null>(null);
  const [schema, setSchema] = useState<Record<
    string,
    Record<string, unknown>
  > | null>(null);
  const [categoryOrder, setCategoryOrder] = useState<string[]>([]);
  const [defaults, setDefaults] = useState<Record<string, unknown> | null>(
    null,
  );
  const [saving, setSaving] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [yamlMode, setYamlMode] = useState(false);
  const [yamlText, setYamlText] = useState("");
  const [yamlLoading, setYamlLoading] = useState(false);
  const [yamlSaving, setYamlSaving] = useState(false);
  const [configPath, setConfigPath] = useState<string | null>(null);
  const [activeCategory, setActiveCategory] = useState<string>("");
  const [confirmReset, setConfirmReset] = useState(false);
  const [allowlistEntries, setAllowlistEntries] = useState<AllowlistEntry[]>([]);
  const [allowlistDraft, setAllowlistDraft] = useState("");
  const [allowlistLoading, setAllowlistLoading] = useState(true);
  const [allowlistBusy, setAllowlistBusy] = useState(false);
  const [allowlistError, setAllowlistError] = useState<string | null>(null);
  const [allowlistEditTarget, setAllowlistEditTarget] = useState<string | null>(null);
  const [allowlistEditDraft, setAllowlistEditDraft] = useState("");
  const [allowlistTestCommand, setAllowlistTestCommand] = useState("");
  const [allowlistTestResult, setAllowlistTestResult] = useState<AllowlistTestResult | null>(null);
  const [allowlistTestLoading, setAllowlistTestLoading] = useState(false);
  const [allowlistRestartingGateway, setAllowlistRestartingGateway] = useState(false);
  const [allowlistDangerExpanded, setAllowlistDangerExpanded] = useState(false);
  const [allowlistConfirm, setAllowlistConfirm] = useState<
    { type: "delete"; pattern: string } | { type: "clear" } | null
  >(null);
  const { toast, showToast } = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { t } = useI18n();
  const { setEnd } = usePageHeader();

  useLayoutEffect(() => {
    if (!config || !schema) {
      setEnd(null);
      return;
    }
    setEnd(
      <div className="relative w-full min-w-0 sm:max-w-xs">
        <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
        <Input
          className="h-8 pl-8 pr-7 text-xs"
          placeholder={t.common.search}
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
        {searchQuery && (
          <Button
            ghost
            size="xs"
            className="absolute right-1.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
            onClick={() => setSearchQuery("")}
            aria-label={t.common.clear}
          >
            <X />
          </Button>
        )}
      </div>,
    );
    return () => setEnd(null);
  }, [config, schema, searchQuery, setEnd, t.common.clear, t.common.search]);

  function prettyCategoryName(cat: string): string {
    const key = cat as keyof typeof t.config.categories;
    if (t.config.categories[key]) return t.config.categories[key];
    return cat.charAt(0).toUpperCase() + cat.slice(1);
  }

  const loadCommandAllowlist = async (showErrorToast = false) => {
    setAllowlistLoading(true);
    setAllowlistError(null);
    try {
      const resp = await api.getCommandAllowlist();
      setAllowlistEntries(resp.entries ?? []);
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      setAllowlistError(message);
      if (showErrorToast) {
        showToast(`Failed to load always-allow patterns: ${message}`, "error");
      }
    } finally {
      setAllowlistLoading(false);
    }
  };

  useEffect(() => {
    api
      .getConfig()
      .then(setConfig)
      .catch(() => {});
    api
      .getSchema()
      .then((resp) => {
        setSchema(resp.fields as Record<string, Record<string, unknown>>);
        setCategoryOrder(resp.category_order ?? []);
      })
      .catch(() => {});
    api
      .getDefaults()
      .then(setDefaults)
      .catch(() => {});
    // getConfigRaw is profile-scoped (fetchJSON appends ?profile=), so its
    // `path` reflects the switched profile's config.yaml. /api/status's
    // config_path is machine-global (the dashboard's own profile) — wrong
    // header under the global profile switcher, so it's only a fallback.
    api
      .getConfigRaw()
      .then((resp) => {
        if (resp.path) setConfigPath(resp.path);
      })
      .catch(() => {});
    api
      .getStatus()
      .then((resp) => setConfigPath((prev) => prev ?? resp.config_path))
      .catch(() => {});
    loadCommandAllowlist();
  }, []);

  // Set active category when categories load
  useEffect(() => {
    if (categoryOrder.length > 0 && !activeCategory) {
      setActiveCategory(categoryOrder[0]);
    }
  }, [categoryOrder, activeCategory]);

  // Load YAML when switching to YAML mode
  useEffect(() => {
    if (yamlMode) {
      setYamlLoading(true);
      api
        .getConfigRaw()
        .then((resp) => setYamlText(resp.yaml))
        .catch(() => showToast(t.config.failedToLoadRaw, "error"))
        .finally(() => setYamlLoading(false));
    }
  }, [yamlMode]);

  /* ---- Categories ---- */
  const categories = useMemo(() => {
    if (!schema) return [];
    const allCats = [
      ...new Set(
        Object.values(schema).map((s) => String(s.category ?? "general")),
      ),
    ];
    const ordered = categoryOrder.filter((c) => allCats.includes(c));
    const extra = allCats.filter((c) => !categoryOrder.includes(c)).sort();
    return [...ordered, ...extra];
  }, [schema, categoryOrder]);

  /* ---- Category field counts ---- */
  const categoryCounts = useMemo(() => {
    if (!schema) return {};
    const counts: Record<string, number> = {};
    for (const s of Object.values(schema)) {
      const cat = String(s.category ?? "general");
      counts[cat] = (counts[cat] || 0) + 1;
    }
    return counts;
  }, [schema]);

  /* ---- Search ---- */
  const isSearching = searchQuery.trim().length > 0;
  const lowerSearch = searchQuery.toLowerCase();

  const searchMatchedFields = useMemo(() => {
    if (!isSearching || !schema) return [];
    return Object.entries(schema).filter(([key, s]) => {
      const label = key.split(".").pop() ?? key;
      const humanLabel = label.replace(/_/g, " ");
      return (
        key.toLowerCase().includes(lowerSearch) ||
        humanLabel.toLowerCase().includes(lowerSearch) ||
        String(s.category ?? "")
          .toLowerCase()
          .includes(lowerSearch) ||
        String(s.description ?? "")
          .toLowerCase()
          .includes(lowerSearch)
      );
    });
  }, [isSearching, lowerSearch, schema]);

  /* ---- Active tab fields ---- */
  const activeFields = useMemo(() => {
    if (!schema || isSearching) return [];
    return Object.entries(schema).filter(
      ([, s]) => String(s.category ?? "general") === activeCategory,
    );
  }, [schema, activeCategory, isSearching]);

  const exactTestCandidate = allowlistTestCommand.trim();
  const allowlistPatterns = useMemo(
    () => allowlistEntries.map((entry) => entry.pattern),
    [allowlistEntries],
  );
  const manualAllowlistEntries = useMemo(
    () => allowlistEntries.filter((entry) => entry.kind === "manual"),
    [allowlistEntries],
  );
  const dangerCategoryAllowlistEntries = useMemo(
    () => allowlistEntries.filter((entry) => entry.kind === "danger_category"),
    [allowlistEntries],
  );
  useEffect(() => {
    if (!exactTestCandidate) {
      setAllowlistTestResult(null);
      setAllowlistTestLoading(false);
      return;
    }
    let cancelled = false;
    setAllowlistTestLoading(true);
    const timer = window.setTimeout(() => {
      api
        .testCommandAllowlist(exactTestCandidate)
        .then((resp) => {
          if (!cancelled) setAllowlistTestResult(resp);
        })
        .catch(() => {
          if (!cancelled) setAllowlistTestResult(null);
        })
        .finally(() => {
          if (!cancelled) setAllowlistTestLoading(false);
        });
    }, 180);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [exactTestCandidate]);
  const dangerCategoryCollapseThreshold = 3;
  const shouldCollapseDangerCategories =
    dangerCategoryAllowlistEntries.length > dangerCategoryCollapseThreshold;

  /* ---- Handlers ---- */
  const handleSave = async () => {
    if (!config) return;
    setSaving(true);
    try {
      await api.saveConfig(config);
      showToast(t.config.configSaved, "success");
    } catch (e) {
      showToast(`${t.config.failedToSave}: ${e}`, "error");
    } finally {
      setSaving(false);
    }
  };

  const handleYamlSave = async () => {
    setYamlSaving(true);
    try {
      await api.saveConfigRaw(yamlText);
      showToast(t.config.yamlConfigSaved, "success");
      api
        .getConfig()
        .then(setConfig)
        .catch(() => {});
    } catch (e) {
      showToast(`${t.config.failedToSaveYaml}: ${e}`, "error");
    } finally {
      setYamlSaving(false);
    }
  };

  const handleReset = () => {
    if (!defaults || !config) return;
    // Scope the reset to what the user is currently looking at:
    //   - search mode → the matched fields
    //   - form mode   → the active category's fields
    // Resetting the whole config here was a footgun (issue reported by @ykmfb001):
    // the button sits next to the category tabs and users reasonably assumed
    // "reset this tab", not "wipe my entire config.yaml".
    const scopedFields = isSearching ? searchMatchedFields : activeFields;
    if (scopedFields.length === 0) return;
    setConfirmReset(true);
  };

  const executeReset = () => {
    if (!defaults || !config) return;
    setConfirmReset(false);
    const scopedFields = isSearching ? searchMatchedFields : activeFields;
    if (scopedFields.length === 0) return;
    const scopeLabel = isSearching
      ? t.config.searchResults
      : prettyCategoryName(activeCategory);
    let next: Record<string, unknown> = config;
    for (const [key] of scopedFields) {
      next = setNestedValue(next, key, getNestedValue(defaults, key));
    }
    setConfig(next);
    showToast(
      t.config.resetScopeToast.replace("{scope}", scopeLabel),
      "success",
    );
  };

  const handleExport = () => {
    if (!config) return;
    const blob = new Blob([JSON.stringify(config, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "hermes-config.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const imported = JSON.parse(reader.result as string);
        setConfig(imported);
        showToast(t.config.configImported, "success");
      } catch {
        showToast(t.config.invalidJson, "error");
      }
    };
    reader.readAsText(file);
  };

  const executeAllowlistAction = async () => {
    if (!allowlistConfirm) return;
    setAllowlistBusy(true);
    try {
      if (allowlistConfirm.type === "delete") {
        const resp = await api.deleteCommandAllowlistEntry(allowlistConfirm.pattern);
        setAllowlistEntries(resp.entries ?? []);
        if (allowlistEditTarget === allowlistConfirm.pattern) {
          cancelAllowlistEdit();
        }
        showToast("Always-allow entry removed.", "success");
      } else {
        const resp = await api.clearCommandAllowlist();
        setAllowlistEntries(resp.entries ?? []);
        cancelAllowlistEdit();
        showToast("Always-allow list cleared.", "success");
      }
      setAllowlistConfirm(null);
    } catch (e) {
      showToast(
        `Failed to update always-allow list: ${e instanceof Error ? e.message : String(e)}`,
        "error",
      );
    } finally {
      setAllowlistBusy(false);
    }
  };

  const handleAddAllowlistPattern = async () => {
    const pattern = allowlistDraft.trim();
    if (!pattern) {
      showToast("Pattern cannot be empty.", "error");
      return;
    }
    setAllowlistBusy(true);
    try {
      const resp = await api.addCommandAllowlistEntry(pattern);
      setAllowlistEntries(resp.entries ?? []);
      setAllowlistDraft("");
      showToast(
        resp.created ? "Always-allow entry added." : "Entry already exists.",
        "success",
      );
    } catch (e) {
      showToast(
        `Failed to add always-allow pattern: ${e instanceof Error ? e.message : String(e)}`,
        "error",
      );
    } finally {
      setAllowlistBusy(false);
    }
  };

  const startAllowlistEdit = (entry: AllowlistEntry) => {
    if (entry.kind === "danger_category") {
      showToast(
        "Danger-category entries come from approval prompts and should not be edited in place. Delete it if you want to remove it, or add a manual exact pattern separately.",
        "error",
      );
      return;
    }
    setAllowlistEditTarget(entry.pattern);
    setAllowlistEditDraft(entry.pattern);
  };

  const cancelAllowlistEdit = () => {
    setAllowlistEditTarget(null);
    setAllowlistEditDraft("");
  };

  const handleSaveAllowlistEdit = async () => {
    if (!allowlistEditTarget) return;
    const replacement = allowlistEditDraft.trim();
    if (!replacement) {
      showToast("Replacement pattern cannot be empty.", "error");
      return;
    }
    setAllowlistBusy(true);
    try {
      const resp = await api.updateCommandAllowlistEntry(allowlistEditTarget, replacement);
      setAllowlistEntries(resp.entries ?? []);
      cancelAllowlistEdit();
      showToast("Always-allow entry updated.", "success");
    } catch (e) {
      showToast(
        `Failed to update always-allow entry: ${e instanceof Error ? e.message : String(e)}`,
        "error",
      );
    } finally {
      setAllowlistBusy(false);
    }
  };

  const handleCopyToClipboard = async (value: string, label = "Copied to clipboard.") => {
    try {
      await navigator.clipboard.writeText(value);
      showToast(label, "success");
    } catch (e) {
      showToast(
        `Could not copy to clipboard: ${e instanceof Error ? e.message : String(e)}`,
        "error",
      );
    }
  };

  const handleRestartGateway = async () => {
    setAllowlistRestartingGateway(true);
    try {
      await api.restartGateway();
      showToast("Gateway restart started.", "success");
    } catch (e) {
      showToast(
        `Failed to restart gateway: ${e instanceof Error ? e.message : String(e)}`,
        "error",
      );
    } finally {
      setAllowlistRestartingGateway(false);
    }
  };

  /* ---- Loading ---- */
  if (!config || !schema) {
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner className="text-2xl text-primary" />
      </div>
    );
  }

  /* ---- Render field list (shared between search & normal) ---- */
  const renderFields = (
    fields: [string, Record<string, unknown>][],
    showCategory = false,
  ) => {
    let lastSection = "";
    let lastCat = "";
    return fields.map(([key, s]) => {
      const parts = key.split(".");
      const section = parts.length > 1 ? parts[0] : "";
      const cat = String(s.category ?? "general");
      const showCatBadge = showCategory && cat !== lastCat;
      const showSection =
        !showCategory &&
        section &&
        section !== lastSection &&
        section !== activeCategory;
      lastSection = section;
      lastCat = cat;

      return (
        <div key={key}>
          {showCatBadge && (
            <div className="flex items-center gap-2 pt-4 pb-2 first:pt-0">
              <CategoryIcon
                category={cat}
                className="h-4 w-4 text-muted-foreground"
              />
              <span className="font-mondwest text-display text-xs font-semibold tracking-wider text-muted-foreground">
                {prettyCategoryName(cat)}
              </span>
              <div className="flex-1 border-t border-border" />
            </div>
          )}
          {showSection && (
            <div className="flex items-center gap-2 pt-4 pb-2 first:pt-0">
              <span className="font-mondwest text-display text-xs font-semibold tracking-wider text-muted-foreground">
                {section.replace(/_/g, " ")}
              </span>
              <div className="flex-1 border-t border-border" />
            </div>
          )}
          <div className="py-1">
            <AutoField
              schemaKey={key}
              schema={s}
              value={getNestedValue(config, key)}
              onChange={(v) => setConfig(setNestedValue(config, key, v))}
            />
          </div>
        </div>
      );
    });
  };

  const renderAllowlistEntryRows = (
    entries: AllowlistEntry[],
    options?: { compactDangerCategory?: boolean },
  ) => (
    <div className="grid gap-2">
      {entries.map((entry) => {
        const pattern = entry.pattern;
        const isEditing = allowlistEditTarget === pattern;
        const isDangerCategory = entry.kind === "danger_category";
        const compactDangerCategory = isDangerCategory && options?.compactDangerCategory;
        const badgeLabel = isDangerCategory ? "Danger category" : "Manual";
        const badgeClassName = isDangerCategory
          ? "border-amber-400/80 bg-amber-400/18 text-amber-100 shadow-[inset_0_0_0_1px_rgba(251,191,36,0.18)]"
          : "border-sky-400/70 bg-sky-500/12 text-sky-100 shadow-[inset_0_0_0_1px_rgba(56,189,248,0.14)]";
        const dangerTooltip =
          "This is a broad approval key saved from an approval prompt. It can cover future commands that match the same dangerous-command category, not just the one original command text.";
        const rowClassName = compactDangerCategory
          ? "grid gap-2 border border-amber-500/35 bg-amber-500/4 px-2.5 py-2"
          : isDangerCategory
            ? "grid gap-3 border border-amber-500/30 bg-amber-500/5 px-3 py-3"
            : "grid gap-3 border border-border/60 px-3 py-3";
        const valueBoxClassName = compactDangerCategory
          ? "grid gap-1.5 border border-amber-500/20 bg-background/20 px-2.5 py-2"
          : "grid gap-2 border border-border/60 bg-muted/10 px-3 py-3";
        const actionButtonClassName = compactDangerCategory
          ? "h-8 min-w-[88px] justify-center px-2.5 text-[11px]"
          : "h-9 min-w-[96px] justify-center";
        return (
          <div key={`${entry.kind}:${pattern}`} className={rowClassName}>
            {isEditing ? (
              <>
                <div className="flex items-center justify-between gap-3">
                  <Badge tone="secondary" className={`text-[11px] uppercase tracking-wide ${badgeClassName}`}>
                    {badgeLabel}
                  </Badge>
                  <p className="text-[11px] text-muted-foreground">Manual exact pattern edit</p>
                </div>
                <Input
                  value={allowlistEditDraft}
                  onChange={(e) => setAllowlistEditDraft(e.target.value)}
                  className="text-xs"
                  disabled={allowlistBusy}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      void handleSaveAllowlistEdit();
                    }
                    if (e.key === "Escape") {
                      e.preventDefault();
                      cancelAllowlistEdit();
                    }
                  }}
                />
                <div className="flex flex-wrap justify-end gap-2">
                  <Button
                    size="sm"
                    outlined
                    onClick={cancelAllowlistEdit}
                    disabled={allowlistBusy}
                  >
                    Cancel
                  </Button>
                  <Button
                    size="sm"
                    onClick={() => void handleSaveAllowlistEdit()}
                    disabled={allowlistBusy || allowlistEditDraft.trim().length === 0}
                  >
                    Save changes
                  </Button>
                </div>
              </>
            ) : (
              <div
                className={`grid gap-2.5 ${
                  compactDangerCategory
                    ? "lg:grid-cols-[minmax(0,1fr)_auto] lg:items-start"
                    : "gap-3 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-stretch"
                }`}
              >
                <div className={valueBoxClassName}>
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <Badge
                        tone="secondary"
                        className={`text-[11px] uppercase tracking-wide ${badgeClassName}`}
                        title={isDangerCategory ? dangerTooltip : "Manual exact-match entry saved as-is."}
                      >
                        {badgeLabel}
                      </Badge>
                      {isDangerCategory ? (
                        <span
                          className="inline-flex h-5 w-5 items-center justify-center rounded border border-amber-400/35 bg-amber-400/10 text-amber-200"
                          title={dangerTooltip}
                          aria-label={`What does danger category ${pattern} cover?`}
                        >
                          <CircleHelp className="h-3.5 w-3.5" />
                        </span>
                      ) : null}
                    </div>
                  </div>
                  <div className={`flex min-w-0 items-center font-mono text-foreground ${compactDangerCategory ? "min-h-7 text-[11px]" : "min-h-9 text-xs"}`}>
                    <span className="min-w-0 break-all">{pattern}</span>
                  </div>
                  {isDangerCategory ? (
                    <p className={compactDangerCategory ? "text-[10.5px] leading-5 text-amber-200/95" : "text-[11px] text-amber-300/90"}>
                      Covers future commands that hit this same dangerous-command category. Delete it to remove the broad approval, or add a separate manual exact pattern if needed.
                    </p>
                  ) : (
                    <p className="text-[11px] text-muted-foreground">
                      Exact-match manual entry saved as-is.
                    </p>
                  )}
                </div>
                <div className={`flex flex-wrap items-stretch justify-end gap-2 ${compactDangerCategory ? "lg:shrink-0 lg:self-start" : "lg:shrink-0"}`}>
                  <Button
                    outlined
                    size="sm"
                    className={actionButtonClassName}
                    prefix={<Copy className="h-3.5 w-3.5" />}
                    onClick={() => void handleCopyToClipboard(pattern, "Entry copied.")}
                    disabled={allowlistBusy}
                  >
                    Copy
                  </Button>
                  <Button
                    outlined
                    size="sm"
                    className={actionButtonClassName}
                    prefix={<Pencil className="h-3.5 w-3.5" />}
                    onClick={() => startAllowlistEdit(entry)}
                    disabled={allowlistBusy || isDangerCategory}
                    title={
                      isDangerCategory
                        ? "Danger-category approvals should not be edited in place. Delete it or add a manual exact pattern."
                        : "Edit entry"
                    }
                    aria-label={
                      isDangerCategory
                        ? `Editing disabled for danger category ${pattern}`
                        : `Edit entry ${pattern}`
                    }
                  >
                    {isDangerCategory ? "Manual only" : "Edit"}
                  </Button>
                  <Button
                    outlined
                    size="sm"
                    className={`${actionButtonClassName} text-destructive`}
                    prefix={<Trash2 className="h-3.5 w-3.5" />}
                    title="Delete entry"
                    aria-label={`Delete entry ${pattern}`}
                    onClick={() => setAllowlistConfirm({ type: "delete", pattern })}
                    disabled={allowlistBusy}
                  >
                    Delete
                  </Button>
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );

  const renderDangerCategoryChipList = (entries: AllowlistEntry[]) => (
    <div className="grid gap-2">
      <div className="flex flex-wrap gap-2">
        {entries.map((entry) => (
          <span
            key={`danger-chip:${entry.pattern}`}
            className="inline-flex max-w-full items-center rounded-full border border-amber-400/35 bg-amber-400/10 px-2.5 py-1 font-mono text-[11px] leading-5 text-amber-100"
            title={`Broad approval category: ${entry.pattern}. ${"This can auto-approve future commands that match the same dangerous-command class."}`}
          >
            <span className="truncate">{entry.pattern}</span>
          </span>
        ))}
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-muted-foreground">
        <span>
          Showing compact category chips because there are {dangerCategoryAllowlistEntries.length} danger approvals.
          Expand to edit or delete individual entries.
        </span>
        <Button
          size="sm"
          outlined
          className="h-8 min-w-[132px] justify-center px-2.5 text-[11px]"
          onClick={() => setAllowlistDangerExpanded(true)}
          disabled={allowlistBusy}
        >
          Expand controls
        </Button>
      </div>
    </div>
  );

  const renderAlwaysAllowCard = () => (
    <Card>
      <CardHeader className="py-3 px-4">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div className="space-y-1">
            <CardTitle className="text-sm flex items-center gap-2">
              <Shield className="h-4 w-4" />
              Always Allow
            </CardTitle>
            <p className="text-xs text-muted-foreground">
              Persistent dangerous-command approvals and manual exact-match entries saved in <code>command_allowlist</code>.
              Approval-prompt saves are grouped below as danger categories like <code>recursive delete</code>. Removing an entry stops future auto-approval, but active Hermes sessions or gateway processes may need a restart or a fresh session to fully drop in-memory approvals.
            </p>
          </div>
          <div className="flex items-center gap-2 sm:shrink-0">
            <Badge tone={allowlistPatterns.length > 0 ? "warning" : "secondary"} className="text-xs">
              {allowlistPatterns.length} pattern{allowlistPatterns.length === 1 ? "" : "s"}
            </Badge>
            <Button
              ghost
              size="icon"
              onClick={() => loadCommandAllowlist(true)}
              title="Refresh always-allow list"
              aria-label="Refresh always-allow list"
              disabled={allowlistLoading || allowlistBusy}
            >
              <RefreshCw />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="grid gap-4 px-4 pb-4">
        <div className="grid gap-2">
          <p className="text-xs text-muted-foreground">
            Add a manual exact-match entry to the persistent allowlist. Use this for precise strings you want saved as-is.
          </p>
          <div className="grid gap-2 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-stretch">
            <Input
              value={allowlistDraft}
              onChange={(e) => setAllowlistDraft(e.target.value)}
              placeholder="e.g. docker ps --format {{.Names}}"
              className="text-xs"
              disabled={allowlistBusy}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  void handleAddAllowlistPattern();
                }
              }}
            />
            <Button
              size="sm"
              className="h-9 min-w-[140px] justify-center whitespace-nowrap"
              onClick={() => void handleAddAllowlistPattern()}
              disabled={allowlistBusy || allowlistDraft.trim().length === 0}
            >
              Add pattern
            </Button>
          </div>
        </div>

        <div className="grid gap-2 border border-border/60 bg-muted/10 px-4 py-3">
          <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-sm font-medium">Test runtime allow match</p>
              <p className="text-xs text-muted-foreground">
                Paste a full command to check the real runtime allowlist contract: exact strings, wildcard patterns, danger-category approvals, and shell-operator rejection.
              </p>
            </div>
            {exactTestCandidate ? (
              <Badge
                tone={
                  allowlistTestLoading
                    ? "secondary"
                    : allowlistTestResult?.matched
                      ? "success"
                      : "destructive"
                }
                className="text-xs"
              >
                {allowlistTestLoading
                  ? "Checking"
                  : allowlistTestResult?.matched
                    ? "Matched"
                    : "No match"}
              </Badge>
            ) : null}
          </div>
          <Input
            value={allowlistTestCommand}
            onChange={(e) => setAllowlistTestCommand(e.target.value)}
            placeholder="Paste the full command you want to test"
            className="text-xs"
            disabled={allowlistBusy}
          />
          {exactTestCandidate ? (
            allowlistTestLoading ? (
              <p className="text-xs text-muted-foreground">Checking runtime allowlist rules…</p>
            ) : allowlistTestResult?.blocked_by_shell_operator ? (
              <p className="text-xs text-amber-300">
                Rejected for allowlist shortcut: compound shell operators are not eligible for command allowlist matching.
              </p>
            ) : allowlistTestResult?.matched ? (
              <p className="text-xs text-emerald-400">
                Runtime match found: <code>{allowlistTestResult.matched_pattern}</code>
                {allowlistTestResult.matched_kind === "danger_category"
                  ? " (danger category approval)"
                  : ""}
              </p>
            ) : (
              <p className="text-xs text-muted-foreground">
                No saved allowlist rule matches this command under runtime rules.
              </p>
            )
          ) : (
            <p className="text-xs text-muted-foreground">
              This tester uses the backend matcher, not a frontend-only guess.
            </p>
          )}
        </div>

        {allowlistLoading ? (
          <div className="flex items-center justify-center py-8">
            <Spinner className="text-xl text-primary" />
          </div>
        ) : allowlistError ? (
          <p className="text-sm text-destructive">{allowlistError}</p>
        ) : allowlistEntries.length === 0 ? (
          <div className="border border-dashed border-border px-4 py-6 text-sm text-muted-foreground">
            No always-allow entries saved.
          </div>
        ) : (
          <div className="grid gap-4">
            <div className="grid gap-2 border border-border/60 bg-muted/10 px-4 py-3">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-medium">Manual exact patterns</p>
                  <p className="text-xs text-muted-foreground">
                    Exact strings you added yourself from this panel or by editing config directly.
                  </p>
                </div>
                <Badge tone={manualAllowlistEntries.length > 0 ? "secondary" : "secondary"} className="text-xs">
                  {manualAllowlistEntries.length}
                </Badge>
              </div>
              {manualAllowlistEntries.length > 0 ? (
                renderAllowlistEntryRows(manualAllowlistEntries)
              ) : (
                <div className="border border-dashed border-border px-4 py-4 text-xs text-muted-foreground">
                  No manual exact patterns saved.
                </div>
              )}
            </div>

            <div className="grid gap-2 border border-amber-500/20 bg-amber-500/5 px-3 py-2.5">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-medium">Approval-generated danger categories</p>
                  <p className="text-[11px] leading-5 text-muted-foreground">
                    Broader dangerous-command approval keys created from prompt approvals, not reconstructed full commands.
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  {shouldCollapseDangerCategories ? (
                    <Badge tone={allowlistDangerExpanded ? "warning" : "secondary"} className="text-[11px]">
                      {allowlistDangerExpanded ? "Expanded" : "Compact"}
                    </Badge>
                  ) : null}
                  <Badge tone={dangerCategoryAllowlistEntries.length > 0 ? "warning" : "secondary"} className="text-xs">
                    {dangerCategoryAllowlistEntries.length}
                  </Badge>
                </div>
              </div>
              {dangerCategoryAllowlistEntries.length > 0 ? (
                shouldCollapseDangerCategories && !allowlistDangerExpanded ? (
                  renderDangerCategoryChipList(dangerCategoryAllowlistEntries)
                ) : (
                  <div className="grid gap-2">
                    {shouldCollapseDangerCategories ? (
                      <div className="flex justify-end">
                        <Button
                          size="sm"
                          outlined
                          className="h-8 min-w-[132px] justify-center px-2.5 text-[11px]"
                          onClick={() => setAllowlistDangerExpanded(false)}
                          disabled={allowlistBusy}
                        >
                          Collapse to chips
                        </Button>
                      </div>
                    ) : null}
                    {renderAllowlistEntryRows(dangerCategoryAllowlistEntries, { compactDangerCategory: true })}
                  </div>
                )
              ) : (
                <div className="border border-dashed border-border px-3 py-3 text-[11px] text-muted-foreground">
                  No approval-generated danger categories saved.
                </div>
              )}
            </div>
          </div>
        )}

        <div className="border border-border/60 bg-muted/10 px-4 py-3">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-sm font-medium">Runtime refresh</p>
              <p className="text-xs text-muted-foreground">
                Config updates are saved immediately. Gateway chats may still hold old approvals in memory until restart. CLI/TUI sessions usually need a fresh session.
              </p>
            </div>
            <div className="flex flex-wrap gap-2 sm:justify-end">
              <Button
                size="sm"
                outlined
                className="h-9 min-w-[140px] justify-center"
                onClick={() => void handleCopyToClipboard("/restart", "Restart command copied.")}
                disabled={allowlistBusy}
              >
                Copy /restart
              </Button>
              <Button
                size="sm"
                className="h-9 min-w-[140px] justify-center"
                onClick={() => void handleRestartGateway()}
                disabled={allowlistRestartingGateway}
              >
                {allowlistRestartingGateway ? "Restarting..." : "Restart gateway"}
              </Button>
            </div>
          </div>
        </div>

        <div className="border border-destructive/30 bg-destructive/5 px-4 py-3">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-sm font-medium text-destructive">Danger zone</p>
              <p className="text-xs text-muted-foreground">
                Clear the full always-allow list. This removes every persistent dangerous-command approval pattern from config.
              </p>
            </div>
            <Button
              size="sm"
              outlined
              prefix={<Trash2 />}
              className="h-9 min-w-[140px] justify-center text-destructive"
              onClick={() => setAllowlistConfirm({ type: "clear" })}
              disabled={allowlistBusy || allowlistPatterns.length === 0}
            >
              Clear all
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );

  return (
    <div className="flex flex-col gap-4">
      <PluginSlot name="config:top" />
      <Toast toast={toast} />

      <div className="flex min-w-0 flex-col gap-3 sm:flex-row sm:items-center sm:justify-between sm:gap-4">
        <div className="flex min-w-0 items-center gap-2 sm:flex-1">
          <Settings2 className="h-4 w-4 shrink-0 text-muted-foreground" />
          <code className="min-w-0 flex-1 break-words text-xs text-muted-foreground bg-muted/50 px-2 py-0.5">
            {configPath ?? t.config.configPath}
          </code>
        </div>
        <div className="flex flex-wrap items-center gap-1.5 sm:shrink-0">
          <Button
            ghost
            size="icon"
            onClick={handleExport}
            title={t.config.exportConfig}
            aria-label={t.config.exportConfig}
          >
            <Download />
          </Button>
          <Button
            ghost
            size="icon"
            onClick={() => fileInputRef.current?.click()}
            title={t.config.importConfig}
            aria-label={t.config.importConfig}
          >
            <Upload />
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".json"
            className="hidden"
            onChange={handleImport}
          />
          {!yamlMode &&
            (() => {
              const resetScopeLabel = isSearching
                ? t.config.searchResults
                : prettyCategoryName(activeCategory);
              const resetTitle = t.config.resetScopeTooltip.replace(
                "{scope}",
                resetScopeLabel,
              );
              return (
                <Button
                  ghost
                  size="icon"
                  onClick={handleReset}
                  title={resetTitle}
                  aria-label={resetTitle}
                >
                  <RotateCcw />
                </Button>
              );
            })()}

          <div className="w-px h-5 bg-border mx-1" />

          <Button
            size="sm"
            outlined={!yamlMode}
            onClick={() => setYamlMode(!yamlMode)}
            prefix={yamlMode ? <FormInput /> : <Code />}
          >
            {yamlMode ? t.common.form : "YAML"}
          </Button>

          {yamlMode ? (
            <Button
              size="sm"
              className="uppercase"
              onClick={handleYamlSave}
              disabled={yamlSaving}
            >
              {yamlSaving ? t.common.saving : t.common.save}
            </Button>
          ) : (
            <Button
              size="sm"
              className="uppercase"
              onClick={handleSave}
              disabled={saving}
            >
              {saving ? t.common.saving : t.common.save}
            </Button>
          )}
        </div>
      </div>

      {yamlMode ? (
        <Card>
          <CardHeader className="py-3 px-4">
            <CardTitle className="text-sm flex items-center gap-2">
              <FileText className="h-4 w-4" />
              {t.config.rawYaml}
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            {yamlLoading ? (
              <div className="flex items-center justify-center py-12">
                <Spinner className="text-xl text-primary" />
              </div>
            ) : (
              <textarea
                className="flex min-h-[600px] w-full bg-transparent px-4 py-3 text-sm font-mono leading-relaxed placeholder:text-muted-foreground focus-visible:outline-none border-t border-border"
                value={yamlText}
                onChange={(e) => setYamlText(e.target.value)}
                spellCheck={false}
              />
            )}
          </CardContent>
        </Card>
      ) : (
        <div className="flex flex-col sm:flex-row gap-4">
          <aside aria-label={t.config.filters} className="sm:w-56 sm:shrink-0">
            <div className="sm:sticky sm:top-4">
              <div className="flex flex-col border border-border bg-muted/20">
                <div className="hidden sm:flex items-center gap-2 px-3 py-2 border-b border-border">
                  <Filter className="h-3 w-3 text-text-tertiary" />
                  <span className="font-mondwest text-display text-xs tracking-[0.12em] text-text-secondary">
                    {t.config.filters}
                  </span>
                </div>

                <div className="hidden sm:block px-3 pt-2 pb-1 font-mondwest text-display text-xs tracking-[0.12em] text-text-tertiary">
                  {t.config.sections}
                </div>

                <div className="flex sm:flex-col gap-1 sm:gap-px p-2 sm:pt-1 overflow-x-auto sm:overflow-x-visible scrollbar-none sm:max-h-[calc(100vh-260px)] sm:overflow-y-auto">
                  {categories.map((cat) => {
                    const isActive = !isSearching && activeCategory === cat;

                    return (
                      <ListItem
                        key={cat}
                        active={isActive}
                        onClick={() => {
                          setSearchQuery("");
                          setActiveCategory(cat);
                        }}
                        className="rounded-none whitespace-nowrap px-2 py-1 text-xs"
                      >
                        <CategoryIcon
                          category={cat}
                          className="h-3.5 w-3.5 shrink-0"
                        />
                        <span className="flex-1 truncate">
                          {prettyCategoryName(cat)}
                        </span>
                        <span
                          className={`text-xs tabular-nums ${
                            isActive
                              ? "text-text-secondary"
                              : "text-text-tertiary"
                          }`}
                        >
                          {categoryCounts[cat] || 0}
                        </span>
                      </ListItem>
                    );
                  })}
                </div>
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
                      {t.config.searchResults}
                    </CardTitle>
                    <Badge tone="secondary" className="text-xs">
                      {searchMatchedFields.length}{" "}
                      {t.config.fields.replace(
                        "{s}",
                        searchMatchedFields.length !== 1 ? "s" : "",
                      )}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="grid gap-2 px-4 pb-4">
                  {searchMatchedFields.length === 0 ? (
                    <p className="text-sm text-muted-foreground text-center py-8">
                      {t.config.noFieldsMatch.replace("{query}", searchQuery)}
                    </p>
                  ) : (
                    renderFields(searchMatchedFields, true)
                  )}
                </CardContent>
              </Card>
            ) : (
              /* Active category */
              <Card>
                <CardHeader className="py-3 px-4">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <CategoryIcon
                        category={activeCategory}
                        className="h-4 w-4"
                      />
                      {prettyCategoryName(activeCategory)}
                    </CardTitle>
                    <Badge tone="secondary" className="text-xs">
                      {activeFields.length}{" "}
                      {t.config.fields.replace(
                        "{s}",
                        activeFields.length !== 1 ? "s" : "",
                      )}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="grid gap-2 px-4 pb-4">
                  {renderFields(activeFields)}
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      )}
      {renderAlwaysAllowCard()}
      <PluginSlot name="config:bottom" />
      <ConfirmDialog
        open={confirmReset}
        onCancel={() => setConfirmReset(false)}
        onConfirm={executeReset}
        title={t.config.confirmResetScope.replace(
          "{scope}",
          isSearching
            ? t.config.searchResults
            : prettyCategoryName(activeCategory),
        )}
        description={`This will reset ${
          (isSearching ? searchMatchedFields : activeFields).length
        } field(s) to their default values.`}
        destructive
        confirmLabel={t.config.resetDefaults}
      />
      <ConfirmDialog
        open={!!allowlistConfirm}
        onCancel={() => setAllowlistConfirm(null)}
        onConfirm={executeAllowlistAction}
        title={
          allowlistConfirm?.type === "delete"
            ? "Delete always-allow pattern?"
            : "Clear all always-allow patterns?"
        }
        description={
          allowlistConfirm?.type === "delete"
            ? `This will remove the persistent approval pattern:\n${allowlistConfirm.pattern}`
            : "This will remove every persistent dangerous-command approval from command_allowlist."
        }
        destructive
        loading={allowlistBusy}
        confirmLabel={allowlistConfirm?.type === "delete" ? "Delete pattern" : "Clear all"}
      />
    </div>
  );
}
