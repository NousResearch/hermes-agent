import { useEffect, useRef, useState, useMemo } from "react";
import {
  Bot,
  ChevronRight,
  Code,
  Ear,
  Download,
  FileText,
  FormInput,
  Globe,
  Lock,
  MessageSquare,
  Mic,
  Monitor,
  Package,
  Palette,
  RotateCcw,
  Save,
  ScrollText,
  Search,
  Settings,
  Settings2,
  Upload,
  Users,
  Volume2,
  Wrench,
  X,
} from "lucide-react";
import type { ComponentType } from "react";
import { api } from "@/lib/api";
import { getNestedValue, setNestedValue } from "@/lib/nested";
import { useToast } from "@/hooks/useToast";
import { Toast } from "@/components/Toast";
import { AutoField } from "@/components/AutoField";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

const CATEGORY_ICONS: Record<string, ComponentType<{ className?: string }>> = {
  general: Settings,
  agent: Bot,
  terminal: Monitor,
  display: Palette,
  delegation: Users,
  memory: Package,
  compression: Package,
  security: Lock,
  browser: Globe,
  voice: Mic,
  tts: Volume2,
  stt: Ear,
  logging: ScrollText,
  discord: MessageSquare,
  auxiliary: Wrench,
};
const FallbackIcon = FileText;

function prettyCategoryName(cat: string): string {
  if (cat === "tts") return "Text-to-Speech";
  if (cat === "stt") return "Speech-to-Text";
  return cat.charAt(0).toUpperCase() + cat.slice(1);
}

function CategoryIcon({ cat, className }: { cat: string; className?: string }) {
  const Icon = CATEGORY_ICONS[cat] ?? FallbackIcon;
  return <Icon className={className} />;
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function ConfigPage() {
  const [config, setConfig] = useState<Record<string, unknown> | null>(null);
  const [schema, setSchema] = useState<Record<string, Record<string, unknown>> | null>(null);
  const [categoryOrder, setCategoryOrder] = useState<string[]>([]);
  const [defaults, setDefaults] = useState<Record<string, unknown> | null>(null);
  const [saving, setSaving] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [yamlMode, setYamlMode] = useState(false);
  const [yamlText, setYamlText] = useState("");
  const [yamlLoading, setYamlLoading] = useState(false);
  const [yamlSaving, setYamlSaving] = useState(false);
  const [activeCategory, setActiveCategory] = useState<string>("");
  const { toast, showToast } = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.getConfig().then(setConfig).catch(() => {});
    api
      .getSchema()
      .then((resp) => {
        setSchema(resp.fields as Record<string, Record<string, unknown>>);
        setCategoryOrder(resp.category_order ?? []);
      })
      .catch(() => {});
    api.getDefaults().then(setDefaults).catch(() => {});
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
        .catch(() => showToast("Failed to load raw config", "error"))
        .finally(() => setYamlLoading(false));
    }
  }, [yamlMode]);

  /* ---- Categories ---- */
  const categories = useMemo(() => {
    if (!schema) return [];
    const allCats = [...new Set(Object.values(schema).map((s) => String(s.category ?? "general")))];
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
        String(s.category ?? "").toLowerCase().includes(lowerSearch) ||
        String(s.description ?? "").toLowerCase().includes(lowerSearch)
      );
    });
  }, [isSearching, lowerSearch, schema]);

  /* ---- Active tab fields ---- */
  const activeFields = useMemo(() => {
    if (!schema || isSearching) return [];
    return Object.entries(schema).filter(
      ([, s]) => String(s.category ?? "general") === activeCategory
    );
  }, [schema, activeCategory, isSearching]);

  /* ---- Handlers ---- */
  const handleSave = async () => {
    if (!config) return;
    setSaving(true);
    try {
      await api.saveConfig(config);
      showToast("Configuration saved", "success");
    } catch (e) {
      showToast(`Failed to save: ${e}`, "error");
    } finally {
      setSaving(false);
    }
  };

  const handleYamlSave = async () => {
    setYamlSaving(true);
    try {
      await api.saveConfigRaw(yamlText);
      showToast("YAML config saved", "success");
      api.getConfig().then(setConfig).catch(() => {});
    } catch (e) {
      showToast(`Failed to save YAML: ${e}`, "error");
    } finally {
      setYamlSaving(false);
    }
  };

  const handleReset = () => {
    if (defaults) setConfig(structuredClone(defaults));
  };

  const handleExport = () => {
    if (!config) return;
    const blob = new Blob([JSON.stringify(config, null, 2)], { type: "application/json" });
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
        showToast("Config imported — review and save", "success");
      } catch {
        showToast("Invalid JSON file", "error");
      }
    };
    reader.readAsText(file);
  };

  /* ---- Loading ---- */
  if (!config || !schema) {
    return (
      <div className="flex items-center justify-center py-24">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
      </div>
    );
  }

  /* ---- Render field list (shared between search & normal) ---- */
  const renderFields = (fields: [string, Record<string, unknown>][], showCategory = false) => {
    let lastSection = "";
    let lastCat = "";
    return fields.map(([key, s]) => {
      const parts = key.split(".");
      const section = parts.length > 1 ? parts[0] : "";
      const cat = String(s.category ?? "general");
      const showCatBadge = showCategory && cat !== lastCat;
      const showSection = !showCategory && section && section !== lastSection && section !== activeCategory;
      lastSection = section;
      lastCat = cat;

      return (
        <div key={key}>
          {showCatBadge && (
            <div className="flex items-center gap-2 pt-4 pb-2 first:pt-0">
              <CategoryIcon cat={cat} className="h-4 w-4 text-muted-foreground" />
              <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                {prettyCategoryName(cat)}
              </span>
              <div className="flex-1 border-t border-border" />
            </div>
          )}
          {showSection && (
            <div className="flex items-center gap-2 pt-4 pb-2 first:pt-0">
              <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
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

  return (
    <div className="flex flex-col gap-4">
      <Toast toast={toast} />

      {/* ═══════════════ Header Bar ═══════════════ */}
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <Settings2 className="h-4 w-4 text-muted-foreground" />
          <code className="text-xs text-muted-foreground bg-muted/50 px-2 py-0.5">
            ~/.hermes/config.yaml
          </code>
        </div>
        <div className="flex items-center gap-1.5">
          <Button variant="ghost" size="sm" onClick={handleExport} title="Export config as JSON" aria-label="Export config">
            <Download className="h-3.5 w-3.5" />
          </Button>
          <Button variant="ghost" size="sm" onClick={() => fileInputRef.current?.click()} title="Import config from JSON" aria-label="Import config">
            <Upload className="h-3.5 w-3.5" />
          </Button>
          <input ref={fileInputRef} type="file" accept=".json" className="hidden" onChange={handleImport} />
          <Button variant="ghost" size="sm" onClick={handleReset} title="Reset to defaults" aria-label="Reset to defaults">
            <RotateCcw className="h-3.5 w-3.5" />
          </Button>

          <div className="w-px h-5 bg-border mx-1" />

          <Button
            variant={yamlMode ? "default" : "outline"}
            size="sm"
            onClick={() => setYamlMode(!yamlMode)}
            className="gap-1.5"
          >
            {yamlMode ? (
              <>
                <FormInput className="h-3.5 w-3.5" />
                Form
              </>
            ) : (
              <>
                <Code className="h-3.5 w-3.5" />
                YAML
              </>
            )}
          </Button>

          {yamlMode ? (
            <Button size="sm" onClick={handleYamlSave} disabled={yamlSaving} className="gap-1.5">
              <Save className="h-3.5 w-3.5" />
              {yamlSaving ? "Saving..." : "Save"}
            </Button>
          ) : (
            <Button size="sm" onClick={handleSave} disabled={saving} className="gap-1.5">
              <Save className="h-3.5 w-3.5" />
              {saving ? "Saving..." : "Save"}
            </Button>
          )}
        </div>
      </div>

      {/* ═══════════════ YAML Mode ═══════════════ */}
      {yamlMode ? (
        <Card>
          <CardHeader className="py-3 px-4">
            <CardTitle className="text-sm flex items-center gap-2">
              <FileText className="h-4 w-4" />
              Raw YAML Configuration
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            {yamlLoading ? (
              <div className="flex items-center justify-center py-12">
                <div className="h-5 w-5 animate-spin rounded-full border-2 border-primary border-t-transparent" />
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
        /* ═══════════════ Form Mode ═══════════════ */
        <div className="flex flex-col sm:flex-row gap-4" style={{ minHeight: "calc(100vh - 180px)" }}>
          {/* ---- Sidebar — horizontal scroll on mobile, fixed column on sm+ ---- */}
          <div className="sm:w-52 sm:shrink-0">
            <div className="sm:sticky sm:top-[72px] flex flex-col gap-1">
              {/* Search */}
              <div className="relative mb-2 hidden sm:block">
                <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
                <Input
                  className="pl-8 h-8 text-xs"
                  placeholder="Search..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
                {searchQuery && (
                  <button
                    type="button"
                    className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                    onClick={() => setSearchQuery("")}
                  >
                    <X className="h-3 w-3" />
                  </button>
                )}
              </div>

              {/* Category nav — horizontal scroll on mobile */}
              <div className="flex sm:flex-col gap-1 overflow-x-auto sm:overflow-x-visible scrollbar-none pb-1 sm:pb-0">
                {categories.map((cat) => {
                const isActive = !isSearching && activeCategory === cat;
                return (
                  <button
                    key={cat}
                    type="button"
                    onClick={() => {
                      setSearchQuery("");
                      setActiveCategory(cat);
                    }}
                    className={`group flex items-center gap-2 px-2.5 py-1.5 text-left text-xs transition-colors cursor-pointer ${
                      isActive
                        ? "bg-primary/10 text-primary font-medium"
                        : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                    }`}
                  >
                    <CategoryIcon cat={cat} className="h-4 w-4 shrink-0" />
                    <span className="flex-1 truncate">{prettyCategoryName(cat)}</span>
                    <span className={`text-[10px] tabular-nums ${isActive ? "text-primary/60" : "text-muted-foreground/50"}`}>
                      {categoryCounts[cat] || 0}
                    </span>
                    {isActive && (
                      <ChevronRight className="h-3 w-3 text-primary/50 shrink-0" />
                    )}
                  </button>
                );
              })}
              </div>
            </div>
          </div>

          {/* ---- Content ---- */}
          <div className="flex-1 min-w-0">
            {isSearching ? (
              /* Search results */
              <Card>
                <CardHeader className="py-3 px-4">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <Search className="h-4 w-4" />
                      Search Results
                    </CardTitle>
                    <Badge variant="secondary" className="text-[10px]">
                      {searchMatchedFields.length} field{searchMatchedFields.length !== 1 ? "s" : ""}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="grid gap-2 px-4 pb-4">
                  {searchMatchedFields.length === 0 ? (
                    <p className="text-sm text-muted-foreground text-center py-8">
                      No fields match "<span className="text-foreground">{searchQuery}</span>"
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
                      <CategoryIcon cat={activeCategory} className="h-4 w-4" />
                      {prettyCategoryName(activeCategory)}
                    </CardTitle>
                    <Badge variant="secondary" className="text-[10px]">
                      {activeFields.length} field{activeFields.length !== 1 ? "s" : ""}
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
    </div>
  );
}
