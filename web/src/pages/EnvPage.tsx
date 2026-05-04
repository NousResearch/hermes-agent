import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Eye,
  EyeOff,
  ExternalLink,
  KeyRound,
  MessageSquare,
  Plus,
  Pencil,
  Save,
  Settings,
  Trash2,
  X,
  Zap,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import { api } from "@/lib/api";
import type { CustomProviderInfo, EnvVarInfo } from "@/lib/api";
import { DeleteConfirmDialog } from "@/components/DeleteConfirmDialog";
import { Toast } from "@/components/Toast";
import { useConfirmDelete } from "@/hooks/useConfirmDelete";
import { useToast } from "@/hooks/useToast";
import { OAuthProvidersCard } from "@/components/OAuthProvidersCard";
import { Button } from "@nous-research/ui/ui/components/button";
import { ListItem } from "@nous-research/ui/ui/components/list-item";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useI18n } from "@/i18n";
import { PluginSlot } from "@/plugins";

/* ------------------------------------------------------------------ */
/*  Provider grouping                                                  */
/* ------------------------------------------------------------------ */

/** Map env-var key prefixes to a human-friendly provider name + ordering. */
const PROVIDER_GROUPS: { prefix: string; name: string; priority: number }[] = [
  // Nous Portal first
  { prefix: "NOUS_", name: "Nous Portal", priority: 0 },
  // Then alphabetical by display name
  { prefix: "ANTHROPIC_", name: "Anthropic", priority: 1 },
  { prefix: "DASHSCOPE_", name: "DashScope (Qwen)", priority: 2 },
  { prefix: "HERMES_QWEN_", name: "DashScope (Qwen)", priority: 2 },
  { prefix: "DEEPSEEK_", name: "DeepSeek", priority: 3 },
  { prefix: "GOOGLE_", name: "Gemini", priority: 4 },
  { prefix: "GEMINI_", name: "Gemini", priority: 4 },
  { prefix: "GLM_", name: "GLM / Z.AI", priority: 5 },
  { prefix: "ZAI_", name: "GLM / Z.AI", priority: 5 },
  { prefix: "Z_AI_", name: "GLM / Z.AI", priority: 5 },
  { prefix: "HF_", name: "Hugging Face", priority: 6 },
  { prefix: "KIMI_", name: "Kimi / Moonshot", priority: 7 },
  { prefix: "MINIMAX_CN_", name: "MiniMax (China)", priority: 9 },
  { prefix: "MINIMAX_", name: "MiniMax", priority: 8 },
  { prefix: "OPENCODE_GO_", name: "OpenCode Go", priority: 10 },
  { prefix: "OPENCODE_ZEN_", name: "OpenCode Zen", priority: 11 },
  { prefix: "OPENROUTER_", name: "OpenRouter", priority: 12 },
  { prefix: "XIAOMI_", name: "Xiaomi MiMo", priority: 13 },
];

function getProviderGroup(key: string): string {
  for (const g of PROVIDER_GROUPS) {
    if (key.startsWith(g.prefix)) return g.name;
  }
  return "Other";
}

function getProviderPriority(groupName: string): number {
  const entry = PROVIDER_GROUPS.find((g) => g.name === groupName);
  return entry?.priority ?? 99;
}

interface ProviderGroup {
  name: string;
  priority: number;
  entries: [string, EnvVarInfo][];
  hasAnySet: boolean;
}

const CATEGORY_META_ICONS: Record<string, typeof KeyRound> = {
  provider: Zap,
  tool: KeyRound,
  messaging: MessageSquare,
  setting: Settings,
};

/* ------------------------------------------------------------------ */
/*  EnvVarRow — single key edit row                                    */
/* ------------------------------------------------------------------ */

function EnvVarRow({
  varKey,
  info,
  edits,
  setEdits,
  revealed,
  saving,
  onSave,
  onClear,
  onReveal,
  onCancelEdit,
  clearDialogOpen = false,
  compact = false,
}: {
  varKey: string;
  info: EnvVarInfo;
  edits: Record<string, string>;
  setEdits: React.Dispatch<React.SetStateAction<Record<string, string>>>;
  revealed: Record<string, string>;
  saving: string | null;
  onSave: (key: string) => void;
  onClear: (key: string) => void;
  onReveal: (key: string) => void;
  onCancelEdit: (key: string) => void;
  clearDialogOpen?: boolean;
  compact?: boolean;
}) {
  const { t } = useI18n();
  const isEditing = edits[varKey] !== undefined;
  const isRevealed = !!revealed[varKey];
  const displayValue = isRevealed
    ? revealed[varKey]
    : (info.redacted_value ?? "---");

  // Compact inline row for unset, non-editing keys (used inside provider groups)
  if (compact && !info.is_set && !isEditing) {
    return (
      <div className="flex items-center justify-between gap-3 py-1.5 opacity-50 hover:opacity-100 transition-opacity">
        <div className="flex items-center gap-2 min-w-0">
          <span className="font-mono-ui text-[0.7rem] text-muted-foreground">
            {varKey}
          </span>
          <span className="text-[0.65rem] text-muted-foreground/60 truncate hidden sm:block">
            {info.description}
          </span>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {info.url && (
            <a
              href={info.url}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 text-[0.65rem] text-primary hover:underline"
            >
              {t.env.getKey} <ExternalLink className="h-2.5 w-2.5" />
            </a>
          )}
          <Button
            size="sm"
            outlined
            prefix={<Pencil />}
            onClick={() => setEdits((prev) => ({ ...prev, [varKey]: "" }))}
          >
            {t.common.set}
          </Button>
        </div>
      </div>
    );
  }

  // Non-compact unset row
  if (!info.is_set && !isEditing) {
    return (
      <div className="flex items-center justify-between gap-3 border border-border/50 px-4 py-2.5 opacity-60 hover:opacity-100 transition-opacity">
        <div className="flex items-center gap-3 min-w-0">
          <Label className="font-mono-ui text-[0.7rem] text-muted-foreground">
            {varKey}
          </Label>
          <span className="text-[0.65rem] text-muted-foreground/60 truncate hidden sm:block">
            {info.description}
          </span>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {info.url && (
            <a
              href={info.url}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 text-[0.65rem] text-primary hover:underline"
            >
              {t.env.getKey} <ExternalLink className="h-2.5 w-2.5" />
            </a>
          )}
          <Button
            size="sm"
            outlined
            prefix={<Pencil />}
            onClick={() => setEdits((prev) => ({ ...prev, [varKey]: "" }))}
          >
            {t.common.set}
          </Button>
        </div>
      </div>
    );
  }

  // Full expanded row for set keys or keys being edited
  return (
    <div className="grid gap-2 border border-border p-4">
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <div className="flex items-center gap-2">
          <Label className="font-mono-ui text-[0.7rem]">{varKey}</Label>
          <Badge tone={info.is_set ? "success" : "outline"}>
            {info.is_set ? t.common.set : t.env.notSet}
          </Badge>
        </div>
        {info.url && (
          <a
            href={info.url}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1 text-[0.65rem] text-primary hover:underline"
          >
            {t.env.getKey} <ExternalLink className="h-2.5 w-2.5" />
          </a>
        )}
      </div>

      <p className="text-xs text-muted-foreground">{info.description}</p>

      {info.tools.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {info.tools.map((tool) => (
            <Badge
              key={tool}
              tone="secondary"
              className="text-[0.6rem] py-0 px-1.5"
            >
              {tool}
            </Badge>
          ))}
        </div>
      )}

      {!isEditing && (
        <div className="flex items-center gap-2">
          <div
            className={`flex-1 border border-border px-3 py-2 font-mono-ui text-xs ${
              isRevealed
                ? "bg-background text-foreground select-all"
                : "bg-muted/30 text-muted-foreground"
            }`}
          >
            {info.is_set ? displayValue : "---"}
          </div>

          {info.is_set && (
            <Button
              ghost
              size="icon"
              onClick={() => onReveal(varKey)}
              title={isRevealed ? t.env.hideValue : t.env.showValue}
              aria-label={isRevealed ? `Hide ${varKey}` : `Reveal ${varKey}`}
            >
              {isRevealed ? <EyeOff /> : <Eye />}
            </Button>
          )}

          <Button
            size="sm"
            outlined
            prefix={<Pencil />}
            onClick={() => setEdits((prev) => ({ ...prev, [varKey]: "" }))}
          >
            {info.is_set ? t.common.replace : t.common.set}
          </Button>

          {info.is_set && (
            <Button
              size="sm"
              outlined
              destructive
              prefix={<Trash2 />}
              onClick={() => onClear(varKey)}
              disabled={saving === varKey || clearDialogOpen}
            >
              {saving === varKey ? "..." : t.common.clear}
            </Button>
          )}
        </div>
      )}

      {isEditing && (
        <div className="flex items-center gap-2">
          <Input
            autoFocus
            type="text"
            value={edits[varKey]}
            onChange={(e) =>
              setEdits((prev) => ({ ...prev, [varKey]: e.target.value }))
            }
            placeholder={
              info.is_set
                ? t.env.replaceCurrentValue.replace(
                    "{preview}",
                    info.redacted_value ?? "---",
                  )
                : t.env.enterValue
            }
            className="flex-1 font-mono-ui text-xs"
          />
          <Button
            size="sm"
            onClick={() => onSave(varKey)}
            prefix={<Save />}
            disabled={saving === varKey || !edits[varKey]}
          >
            {saving === varKey ? "..." : t.common.save}
          </Button>
          <Button
            size="sm"
            outlined
            prefix={<X />}
            onClick={() => onCancelEdit(varKey)}
          >
            {t.common.cancel}
          </Button>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  ProviderGroupCard — groups API key + base URL per provider         */
/* ------------------------------------------------------------------ */

function ProviderGroupCard({
  group,
  edits,
  setEdits,
  revealed,
  saving,
  onSave,
  onClear,
  onReveal,
  onCancelEdit,
  clearDialogOpen = false,
}: {
  group: ProviderGroup;
  edits: Record<string, string>;
  setEdits: React.Dispatch<React.SetStateAction<Record<string, string>>>;
  revealed: Record<string, string>;
  saving: string | null;
  onSave: (key: string) => void;
  onClear: (key: string) => void;
  onReveal: (key: string) => void;
  onCancelEdit: (key: string) => void;
  clearDialogOpen?: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const { t } = useI18n();

  // Separate API keys from base URLs and other settings
  const apiKeys = group.entries.filter(
    ([k]) => k.endsWith("_API_KEY") || k.endsWith("_TOKEN"),
  );
  const baseUrls = group.entries.filter(([k]) => k.endsWith("_BASE_URL"));
  const other = group.entries.filter(
    ([k]) =>
      !k.endsWith("_API_KEY") &&
      !k.endsWith("_TOKEN") &&
      !k.endsWith("_BASE_URL"),
  );
  const hasAnyConfigured = group.entries.some(([, info]) => info.is_set);
  const configuredCount = group.entries.filter(
    ([, info]) => info.is_set,
  ).length;

  // Get a representative URL for "Get key" link
  const keyUrl = apiKeys.find(([, info]) => info.url)?.[1]?.url ?? null;

  return (
    <div className="border border-border">
      {/* Header — always visible */}
      <ListItem
        onClick={() => setExpanded(!expanded)}
        aria-expanded={expanded}
        className="justify-between gap-3 px-4 py-3 hover:bg-primary/5"
      >
        <div className="flex items-center gap-3 min-w-0">
          {expanded ? (
            <ChevronDown className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
          )}
          <span className="font-semibold text-sm tracking-wide">
            {group.name === "Other" ? t.common.other : group.name}
          </span>
          {hasAnyConfigured && (
            <Badge tone="success" className="text-[0.6rem]">
              {configuredCount} {t.common.set.toLowerCase()}
            </Badge>
          )}
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {keyUrl && (
            <a
              href={keyUrl}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 text-[0.65rem] text-primary hover:underline"
              onClick={(e) => e.stopPropagation()}
            >
              {t.env.getKey} <ExternalLink className="h-2.5 w-2.5" />
            </a>
          )}
          <span className="text-[0.65rem] text-muted-foreground/60">
            {t.env.keysCount
              .replace("{count}", String(group.entries.length))
              .replace("{s}", group.entries.length !== 1 ? "s" : "")}
          </span>
        </div>
      </ListItem>

      {expanded && (
        <div className="border-t border-border px-4 py-3 grid gap-2">
          {apiKeys.map(([key, info]) => (
            <EnvVarRow
              key={key}
              varKey={key}
              info={info}
              compact
              edits={edits}
              setEdits={setEdits}
              revealed={revealed}
              saving={saving}
              onSave={onSave}
              onClear={onClear}
              onReveal={onReveal}
              onCancelEdit={onCancelEdit}
              clearDialogOpen={clearDialogOpen}
            />
          ))}

          {baseUrls.map(([key, info]) => (
            <EnvVarRow
              key={key}
              varKey={key}
              info={info}
              compact
              edits={edits}
              setEdits={setEdits}
              revealed={revealed}
              saving={saving}
              onSave={onSave}
              onClear={onClear}
              onReveal={onReveal}
              onCancelEdit={onCancelEdit}
              clearDialogOpen={clearDialogOpen}
            />
          ))}

          {other.map(([key, info]) => (
            <EnvVarRow
              key={key}
              varKey={key}
              info={info}
              compact
              edits={edits}
              setEdits={setEdits}
              revealed={revealed}
              saving={saving}
              onSave={onSave}
              onClear={onClear}
              onReveal={onReveal}
              onCancelEdit={onCancelEdit}
              clearDialogOpen={clearDialogOpen}
            />
          ))}
        </div>
      )}
    </div>
  );
}


/* ------------------------------------------------------------------ */
/*  CustomProviderCard — config-backed providers                       */
/* ------------------------------------------------------------------ */

function CustomProviderCard({ provider }: { provider: CustomProviderInfo }) {
  const [expanded, setExpanded] = useState(false);
  const models = provider.models ?? [];
  const loggedIn = !!provider.auth?.logged_in;
  const credentialCount = provider.auth?.credentials ?? 0;

  return (
    <div className="border border-border bg-primary/5">
      <ListItem
        onClick={() => setExpanded(!expanded)}
        aria-expanded={expanded}
        className="justify-between gap-3 px-4 py-3 hover:bg-primary/10"
      >
        <div className="flex items-center gap-3 min-w-0">
          {expanded ? (
            <ChevronDown className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
          )}
          <div className="min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="font-semibold text-sm tracking-wide truncate">
                {provider.name || provider.slug}
              </span>
              <Badge tone="secondary" className="text-[0.6rem]">
                custom
              </Badge>
              {provider.is_current && (
                <Badge tone="success" className="text-[0.6rem]">
                  current
                </Badge>
              )}
              {loggedIn && (
                <Badge tone="success" className="text-[0.6rem]">
                  authenticated{credentialCount > 1 ? ` ×${credentialCount}` : ""}
                </Badge>
              )}
            </div>
            <div className="font-mono-ui text-[0.65rem] text-muted-foreground truncate">
              {provider.slug}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <span className="text-[0.65rem] text-muted-foreground/60">
            {models.length} model{models.length !== 1 ? "s" : ""}
          </span>
        </div>
      </ListItem>

      {expanded && (
        <div className="border-t border-border px-4 py-3 grid gap-3">
          <div className="grid gap-1">
            <span className="text-[0.65rem] uppercase tracking-wide text-muted-foreground/70">
              Endpoint
            </span>
            <code className="font-mono-ui text-xs break-all bg-muted/30 border border-border px-2 py-1">
              {provider.base_url || provider.api_url || "---"}
            </code>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs">
            <div className="border border-border px-2 py-1.5">
              <div className="text-[0.65rem] text-muted-foreground/70 uppercase tracking-wide">Transport</div>
              <div className="font-mono-ui">{provider.transport || "openai_chat"}</div>
            </div>
            <div className="border border-border px-2 py-1.5">
              <div className="text-[0.65rem] text-muted-foreground/70 uppercase tracking-wide">Auth</div>
              <div>{loggedIn ? "Credential pool" : provider.key_env ? provider.key_env : "Not connected"}</div>
            </div>
            <div className="border border-border px-2 py-1.5">
              <div className="text-[0.65rem] text-muted-foreground/70 uppercase tracking-wide">Models</div>
              <div>{models.length || 0}</div>
            </div>
          </div>
          {models.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {models.map((model) => (
                <Badge key={model} tone="outline" className="text-[0.6rem] py-0 px-1.5">
                  {model}
                </Badge>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Main page                                                          */
/* ------------------------------------------------------------------ */

export default function EnvPage() {
  const [vars, setVars] = useState<Record<string, EnvVarInfo> | null>(null);
  const [customProviders, setCustomProviders] = useState<CustomProviderInfo[]>([]);
  const [edits, setEdits] = useState<Record<string, string>>({});
  const [revealed, setRevealed] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(true); // Show all providers by default
  const [showAddProvider, setShowAddProvider] = useState(false);
  const [addingProvider, setAddingProvider] = useState(false);
  const [providerForm, setProviderForm] = useState({
    provider_id: "",
    name: "",
    base_url: "",
    default_model: "",
    api_key: "",
    key_env: "",
    transport: "openai_chat",
  });
  const { toast, showToast } = useToast();
  const { t } = useI18n();

  const refreshCustomProviders = useCallback(() => {
    api
      .getCustomProviders()
      .then((resp) => setCustomProviders(resp.providers ?? []))
      .catch(() => {});
  }, []);

  useEffect(() => {
    api
      .getEnvVars()
      .then(setVars)
      .catch(() => {});
    refreshCustomProviders();
  }, [refreshCustomProviders]);

  const handleSave = async (key: string) => {
    const value = edits[key];
    if (!value) return;
    setSaving(key);
    try {
      await api.setEnvVar(key, value);
      setVars((prev) =>
        prev
          ? {
              ...prev,
              [key]: {
                ...prev[key],
                is_set: true,
                redacted_value: value.slice(0, 4) + "..." + value.slice(-4),
              },
            }
          : prev,
      );
      setEdits((prev) => {
        const n = { ...prev };
        delete n[key];
        return n;
      });
      setRevealed((prev) => {
        const n = { ...prev };
        delete n[key];
        return n;
      });
      showToast(`${key} ${t.common.save.toLowerCase()}d`, "success");
    } catch (e) {
      showToast(`${t.config.failedToSave} ${key}: ${e}`, "error");
    } finally {
      setSaving(null);
    }
  };

  const keyClear = useConfirmDelete({
    onDelete: useCallback(
      async (key: string) => {
        setSaving(key);
        try {
          await api.deleteEnvVar(key);
          setVars((prev) =>
            prev
              ? {
                  ...prev,
                  [key]: { ...prev[key], is_set: false, redacted_value: null },
                }
              : prev,
          );
          setEdits((prev) => {
            const n = { ...prev };
            delete n[key];
            return n;
          });
          setRevealed((prev) => {
            const n = { ...prev };
            delete n[key];
            return n;
          });
          showToast(`${key} ${t.common.removed}`, "success");
        } catch (e) {
          showToast(`${t.common.failedToRemove} ${key}: ${e}`, "error");
          throw e;
        } finally {
          setSaving(null);
        }
      },
      [showToast, t.common.removed, t.common.failedToRemove],
    ),
  });

  const handleReveal = async (key: string) => {
    if (revealed[key]) {
      setRevealed((prev) => {
        const n = { ...prev };
        delete n[key];
        return n;
      });
      return;
    }
    try {
      const resp = await api.revealEnvVar(key);
      setRevealed((prev) => ({ ...prev, [key]: resp.value }));
    } catch {
      showToast(`${t.common.failedToReveal} ${key}`, "error");
    }
  };

  const cancelEdit = (key: string) => {
    setEdits((prev) => {
      const n = { ...prev };
      delete n[key];
      return n;
    });
  };

  const handleAddProvider = async () => {
    if (!providerForm.provider_id.trim() || !providerForm.base_url.trim()) {
      showToast("Provider id and base URL are required", "error");
      return;
    }
    setAddingProvider(true);
    try {
      await api.createCustomProvider({
        provider_id: providerForm.provider_id.trim(),
        name: providerForm.name.trim() || providerForm.provider_id.trim(),
        base_url: providerForm.base_url.trim(),
        default_model: providerForm.default_model.trim(),
        api_key: providerForm.api_key.trim(),
        key_env: providerForm.key_env.trim(),
        transport: providerForm.transport.trim() || "openai_chat",
        discover_models: true,
      });
      showToast(`${providerForm.provider_id.trim()} added`, "success");
      setProviderForm({
        provider_id: "",
        name: "",
        base_url: "",
        default_model: "",
        api_key: "",
        key_env: "",
        transport: "openai_chat",
      });
      setShowAddProvider(false);
      refreshCustomProviders();
    } catch (e) {
      showToast(`Failed to add provider: ${e}`, "error");
    } finally {
      setAddingProvider(false);
    }
  };

  /* ---- Build provider groups ---- */
  const { providerGroups, nonProviderGrouped } = useMemo(() => {
    if (!vars) return { providerGroups: [], nonProviderGrouped: [] };

    const providerEntries = Object.entries(vars).filter(
      ([, info]) =>
        info.category === "provider" && (showAdvanced || !info.advanced),
    );

    // Group by provider
    const groupMap = new Map<string, [string, EnvVarInfo][]>();
    for (const entry of providerEntries) {
      const groupName = getProviderGroup(entry[0]);
      if (!groupMap.has(groupName)) groupMap.set(groupName, []);
      groupMap.get(groupName)!.push(entry);
    }

    const groups: ProviderGroup[] = Array.from(groupMap.entries())
      .map(([name, entries]) => ({
        name,
        priority: getProviderPriority(name),
        entries,
        hasAnySet: entries.some(([, info]) => info.is_set),
      }))
      .sort((a, b) => a.priority - b.priority);

    // Non-provider categories — use translated labels
    const CATEGORY_META_LABELS: Record<string, string> = {
      tool: t.app.nav.keys,
      messaging: t.common.messaging,
      setting: t.app.nav.config,
    };
    const otherCategories = ["tool", "messaging", "setting"];
    const nonProvider = otherCategories.map((cat) => {
      const entries = Object.entries(vars).filter(
        ([, info]) => info.category === cat && (showAdvanced || !info.advanced),
      );
      const setEntries = entries.filter(([, info]) => info.is_set);
      const unsetEntries = entries.filter(([, info]) => !info.is_set);
      return {
        label: CATEGORY_META_LABELS[cat] ?? cat,
        icon: CATEGORY_META_ICONS[cat] ?? KeyRound,
        category: cat,
        setEntries,
        unsetEntries,
        totalEntries: entries.length,
      };
    });

    return { providerGroups: groups, nonProviderGrouped: nonProvider };
  }, [vars, showAdvanced, t]);

  if (!vars) {
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner className="text-2xl text-primary" />
      </div>
    );
  }

  const totalProviders = providerGroups.length + customProviders.length;
  const configuredProviders =
    providerGroups.filter((g) => g.hasAnySet).length +
    customProviders.filter((p) => p.auth?.logged_in || (p.models?.length ?? 0) > 0 || p.key_env).length;

  const pendingClearKey = keyClear.pendingId;
  const pendingKeyDescription =
    pendingClearKey && vars ? vars[pendingClearKey]?.description : undefined;

  return (
    <div className="flex flex-col gap-6">
      <PluginSlot name="env:top" />
      <Toast toast={toast} />

      <DeleteConfirmDialog
        open={keyClear.isOpen}
        onCancel={keyClear.cancel}
        onConfirm={keyClear.confirm}
        title={t.env.confirmClearTitle}
        description={
          pendingClearKey
            ? `${pendingClearKey}${pendingKeyDescription ? ` — ${pendingKeyDescription}` : ""}. ${t.env.confirmClearMessage}`
            : t.env.confirmClearMessage
        }
        loading={keyClear.isDeleting}
      />

      <div className="flex items-center justify-between">
        <div className="flex flex-col gap-1">
          <p className="text-sm text-muted-foreground">
            {t.env.description} <code>~/.hermes/.env</code>
          </p>
          <p className="text-[0.7rem] text-muted-foreground/70">
            {t.env.changesNote}
          </p>
        </div>
        <Button
          size="sm"
          outlined
          onClick={() => setShowAdvanced(!showAdvanced)}
        >
          {showAdvanced ? t.env.hideAdvanced : t.env.showAdvanced}
        </Button>
      </div>

      <OAuthProvidersCard
        onError={(msg) => showToast(msg, "error")}
        onSuccess={(msg) => showToast(msg, "success")}
      />

      <Card>
        <CardHeader className="border-b border-border bg-card">
          <div className="flex items-start justify-between gap-3">
            <div className="grid gap-1">
              <div className="flex items-center gap-2">
                <Zap className="h-5 w-5 text-muted-foreground" />
                <CardTitle className="text-base">{t.env.llmProviders}</CardTitle>
              </div>
              <CardDescription>
                {t.env.providersConfigured
                  .replace("{configured}", String(configuredProviders))
                  .replace("{total}", String(totalProviders))}
              </CardDescription>
            </div>
            <Button
              size="sm"
              outlined
              prefix={<Plus />}
              onClick={() => setShowAddProvider((v) => !v)}
            >
              Add Provider
            </Button>
          </div>
        </CardHeader>

        <CardContent className="grid gap-0 p-0">
          {showAddProvider && (
            <div className="border-b border-border bg-muted/20 p-4 grid gap-3">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div className="grid gap-1.5">
                  <Label className="text-[0.7rem]">Provider ID</Label>
                  <Input
                    value={providerForm.provider_id}
                    onChange={(e) => setProviderForm((p) => ({ ...p, provider_id: e.target.value }))}
                    placeholder="example.provider"
                  />
                </div>
                <div className="grid gap-1.5">
                  <Label className="text-[0.7rem]">Display name</Label>
                  <Input
                    value={providerForm.name}
                    onChange={(e) => setProviderForm((p) => ({ ...p, name: e.target.value }))}
                    placeholder="Example Provider"
                  />
                </div>
              </div>
              <div className="grid gap-1.5">
                <Label className="text-[0.7rem]">Base URL</Label>
                <Input
                  value={providerForm.base_url}
                  onChange={(e) => setProviderForm((p) => ({ ...p, base_url: e.target.value }))}
                  placeholder="https://api.example.com/v1"
                />
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div className="grid gap-1.5">
                  <Label className="text-[0.7rem]">Default model</Label>
                  <Input
                    value={providerForm.default_model}
                    onChange={(e) => setProviderForm((p) => ({ ...p, default_model: e.target.value }))}
                    placeholder="model-id"
                  />
                </div>
                <div className="grid gap-1.5">
                  <Label className="text-[0.7rem]">Transport</Label>
                  <Input
                    value={providerForm.transport}
                    onChange={(e) => setProviderForm((p) => ({ ...p, transport: e.target.value }))}
                    placeholder="openai_chat"
                  />
                </div>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div className="grid gap-1.5">
                  <Label className="text-[0.7rem]">API key</Label>
                  <Input
                    type="password"
                    value={providerForm.api_key}
                    onChange={(e) => setProviderForm((p) => ({ ...p, api_key: e.target.value }))}
                    placeholder="stored in Hermes auth, not config.yaml"
                  />
                </div>
                <div className="grid gap-1.5">
                  <Label className="text-[0.7rem]">Key env var</Label>
                  <Input
                    value={providerForm.key_env}
                    onChange={(e) => setProviderForm((p) => ({ ...p, key_env: e.target.value }))}
                    placeholder="OPTIONAL_API_KEY"
                  />
                </div>
              </div>
              <div className="flex justify-end gap-2">
                <Button size="sm" outlined onClick={() => setShowAddProvider(false)} disabled={addingProvider}>
                  {t.common.cancel}
                </Button>
                <Button size="sm" prefix={<Save />} onClick={handleAddProvider} disabled={addingProvider}>
                  {addingProvider ? "..." : "Save Provider"}
                </Button>
              </div>
            </div>
          )}

          {customProviders.map((provider) => (
            <CustomProviderCard key={provider.slug} provider={provider} />
          ))}

          {providerGroups.map((group) => (
            <ProviderGroupCard
              key={group.name}
              group={group}
              edits={edits}
              setEdits={setEdits}
              revealed={revealed}
              saving={saving}
              onSave={handleSave}
              onClear={keyClear.requestDelete}
              onReveal={handleReveal}
              onCancelEdit={cancelEdit}
              clearDialogOpen={keyClear.isOpen}
            />
          ))}
        </CardContent>
      </Card>

      {nonProviderGrouped.map(
        ({
          label,
          icon: Icon,
          setEntries,
          unsetEntries,
          totalEntries,
          category,
        }) => {
          if (totalEntries === 0) return null;

          return (
            <Card key={category}>
              <CardHeader className="border-b border-border bg-card">
                <div className="flex items-center gap-2">
                  <Icon className="h-5 w-5 text-muted-foreground" />
                  <CardTitle className="text-base">{label}</CardTitle>
                </div>
                <CardDescription>
                  {setEntries.length} {t.common.of} {totalEntries}{" "}
                  {t.common.configured}
                </CardDescription>
              </CardHeader>

              <CardContent className="grid gap-3 pt-4">
                {setEntries.map(([key, info]) => (
                  <EnvVarRow
                    key={key}
                    varKey={key}
                    info={info}
                    edits={edits}
                    setEdits={setEdits}
                    revealed={revealed}
                    saving={saving}
                    onSave={handleSave}
                    onClear={keyClear.requestDelete}
                    onReveal={handleReveal}
                    onCancelEdit={cancelEdit}
                    clearDialogOpen={keyClear.isOpen}
                  />
                ))}

                {unsetEntries.length > 0 && (
                  <CollapsibleUnset
                    category={category}
                    unsetEntries={unsetEntries}
                    edits={edits}
                    setEdits={setEdits}
                    revealed={revealed}
                    saving={saving}
                    onSave={handleSave}
                    onClear={keyClear.requestDelete}
                    onReveal={handleReveal}
                    onCancelEdit={cancelEdit}
                    clearDialogOpen={keyClear.isOpen}
                  />
                )}
              </CardContent>
            </Card>
          );
        },
      )}
      <PluginSlot name="env:bottom" />
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  CollapsibleUnset — for non-provider categories                     */
/* ------------------------------------------------------------------ */

function CollapsibleUnset({
  category: _category,
  unsetEntries,
  edits,
  setEdits,
  revealed,
  saving,
  onSave,
  onClear,
  onReveal,
  onCancelEdit,
  clearDialogOpen = false,
}: {
  category: string;
  unsetEntries: [string, EnvVarInfo][];
  edits: Record<string, string>;
  setEdits: React.Dispatch<React.SetStateAction<Record<string, string>>>;
  revealed: Record<string, string>;
  saving: string | null;
  onSave: (key: string) => void;
  onClear: (key: string) => void;
  onReveal: (key: string) => void;
  onCancelEdit: (key: string) => void;
  clearDialogOpen?: boolean;
}) {
  const [collapsed, setCollapsed] = useState(true);
  const { t } = useI18n();

  return (
    <>
      <Button
        ghost
        size="sm"
        prefix={collapsed ? <ChevronRight /> : <ChevronDown />}
        onClick={() => setCollapsed(!collapsed)}
        aria-expanded={!collapsed}
        className="self-start mt-1 normal-case tracking-normal text-xs text-muted-foreground hover:text-foreground"
      >
        {t.env.notConfigured.replace("{count}", String(unsetEntries.length))}
      </Button>

      {!collapsed &&
        unsetEntries.map(([key, info]) => (
          <EnvVarRow
            key={key}
            varKey={key}
            info={info}
            edits={edits}
            setEdits={setEdits}
            revealed={revealed}
            saving={saving}
            onSave={onSave}
            onClear={onClear}
            onReveal={onReveal}
            onCancelEdit={onCancelEdit}
            clearDialogOpen={clearDialogOpen}
          />
        ))}
    </>
  );
}
