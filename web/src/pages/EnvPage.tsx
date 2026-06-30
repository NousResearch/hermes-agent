import { useCallback, useEffect, useLayoutEffect, useMemo, useState } from "react";
import {
  Eye,
  EyeOff,
  ExternalLink,
  KeyRound,
  MessageSquare,
  Pencil,
  Plus,
  Save,
  Settings,
  Trash2,
  X,
  Zap,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import { api } from "@/lib/api";
import type { EnvVarInfo } from "@/lib/api";
import { DeleteConfirmDialog } from "@/components/DeleteConfirmDialog";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { useConfirmDelete } from "@nous-research/ui/hooks/use-confirm-delete";
import { useToast } from "@nous-research/ui/hooks/use-toast";
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
} from "@nous-research/ui/ui/components/card";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Input } from "@nous-research/ui/ui/components/input";
import { Label } from "@nous-research/ui/ui/components/label";
import { useI18n } from "@/i18n";
import { usePageHeader } from "@/contexts/usePageHeader";
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
      <div className="flex items-center justify-between gap-3 py-1.5 min-w-0 overflow-hidden text-text-secondary hover:text-foreground transition-colors">
        <div className="flex items-center gap-2 min-w-0">
          <span className="font-mono-ui text-xs">
            {varKey}
          </span>
          <span className="text-xs text-text-tertiary truncate hidden sm:block">
            {info.description}
          </span>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {info.url && (
            <a
              href={info.url}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
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
      <div className="flex items-center justify-between gap-3 border border-border/50 px-4 py-2.5 min-w-0 overflow-hidden text-text-secondary hover:text-foreground transition-colors">
        <div className="flex items-center gap-3 min-w-0">
          <Label className="font-mono-ui text-xs">
            {varKey}
          </Label>
          <span className="text-xs text-text-tertiary truncate hidden sm:block">
            {info.description}
          </span>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {info.url && (
            <a
              href={info.url}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
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
    <div className="grid gap-2 border border-border p-4 min-w-0 overflow-hidden">
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <div className="flex items-center gap-2">
          <Label className="font-mono-ui text-xs">{varKey}</Label>
          <Badge tone={info.is_set ? "success" : "outline"}>
            {info.is_set ? t.common.set : t.env.notSet}
          </Badge>
        </div>
        {info.url && (
          <a
            href={info.url}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
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
              className="text-xs py-0 px-1.5"
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
            <Badge tone="success" className="text-xs">
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
              className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
              onClick={(e) => e.stopPropagation()}
            >
              {t.env.getKey} <ExternalLink className="h-2.5 w-2.5" />
            </a>
          )}
          <span className="text-xs text-text-tertiary">
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
/*  CustomKeysCard — user-added arbitrary env vars + add-key form      */
/* ------------------------------------------------------------------ */

// Mirror of the backend env-name guard (hermes_cli/config.py _ENV_VAR_NAME_RE).
const ENV_VAR_NAME_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;

function CustomKeysCard({
  entries,
  edits,
  setEdits,
  revealed,
  saving,
  onSave,
  onClear,
  onReveal,
  onCancelEdit,
  onAddKey,
  clearDialogOpen = false,
}: {
  entries: [string, EnvVarInfo][];
  edits: Record<string, string>;
  setEdits: React.Dispatch<React.SetStateAction<Record<string, string>>>;
  revealed: Record<string, string>;
  saving: string | null;
  onSave: (key: string) => void;
  onClear: (key: string) => void;
  onReveal: (key: string) => void;
  onCancelEdit: (key: string) => void;
  onAddKey: (key: string) => void;
  clearDialogOpen?: boolean;
}) {
  const { t } = useI18n();
  const [newKey, setNewKey] = useState("");
  const trimmed = newKey.trim().toUpperCase();
  const alreadyEditing = edits[trimmed] !== undefined;
  const nameValid = ENV_VAR_NAME_RE.test(trimmed);
  const showInvalid = trimmed.length > 0 && !nameValid;

  const rowProps = {
    edits,
    setEdits,
    revealed,
    saving,
    onSave,
    onClear,
    onReveal,
    onCancelEdit,
    clearDialogOpen,
  };

  const handleAdd = () => {
    if (!nameValid || alreadyEditing) return;
    onAddKey(trimmed);
    setNewKey("");
  };

  return (
    <Card id="section-custom">
      <CardHeader className="border-b border-border bg-card">
        <div className="flex items-center gap-2">
          <KeyRound className="h-5 w-5 text-muted-foreground" />
          <CardTitle className="text-base">{t.env.customTitle}</CardTitle>
        </div>
        <CardDescription>
          {t.env.customConfigured
            .replace("{count}", String(entries.length))
            .replace("{s}", entries.length !== 1 ? "s" : "")}
        </CardDescription>
        <CardDescription className="text-text-tertiary">
          {t.env.customHint}
        </CardDescription>
      </CardHeader>

      <CardContent className="grid gap-3 overflow-hidden pt-4">
        {entries.map(([key, info]) => (
          <EnvVarRow key={key} varKey={key} info={info} {...rowProps} />
        ))}

        {/* Add-key form */}
        <div className="grid gap-2 border border-dashed border-border p-4">
          <Label className="text-xs font-semibold tracking-wide">
            {t.env.addCustomKey}
          </Label>
          <div className="flex items-start gap-2">
            <div className="flex-1">
              <Input
                type="text"
                value={newKey}
                onChange={(e) => setNewKey(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleAdd();
                }}
                placeholder={t.env.customKeyNamePlaceholder}
                aria-label={t.env.customKeyName}
                className="w-full font-mono-ui text-xs"
              />
              {showInvalid && (
                <p className="mt-1 text-xs text-destructive">
                  {t.env.invalidKeyName}
                </p>
              )}
            </div>
            <Button
              size="sm"
              prefix={<Plus />}
              onClick={handleAdd}
              disabled={!nameValid || alreadyEditing}
            >
              {t.env.add}
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

/* ------------------------------------------------------------------ */
/*  Main page                                                          */
/* ------------------------------------------------------------------ */

export default function EnvPage() {
  const [vars, setVars] = useState<Record<string, EnvVarInfo> | null>(null);
  const [edits, setEdits] = useState<Record<string, string>>({});
  const [revealed, setRevealed] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(true); // Show all providers by default
  const { toast, showToast } = useToast();
  const { t } = useI18n();
  const { setAfterTitle } = usePageHeader();

  useEffect(() => {
    api
      .getEnvVars()
      .then(setVars)
      .catch(() => {});
  }, []);

  // Scroll-to sub-nav in the page header
  const sections = useMemo(() => {
    const items: { id: string; label: string }[] = [
      { id: "section-oauth", label: "OAuth" },
      { id: "section-providers", label: "Providers" },
    ];
    if (vars) {
      const categories = ["tool", "messaging", "setting"];
      const CATEGORY_LABELS: Record<string, string> = {
        tool: "Tools",
        messaging: t.common.gateway ?? "Gateway",
        setting: "Settings",
      };
      for (const cat of categories) {
        const hasEntries = Object.values(vars).some(
          (info) => info.category === cat && !info.channel_managed,
        );
        if (hasEntries) {
          items.push({ id: `section-${cat}`, label: CATEGORY_LABELS[cat] ?? cat });
        }
      }
      // Custom keys section is always present (it carries the add-key form).
      items.push({ id: "section-custom", label: t.env.customTitle });
    }
    return items;
  }, [vars, t]);

  useLayoutEffect(() => {
    if (!vars) {
      setAfterTitle(null);
      return;
    }
    const scrollTo = (id: string) => {
      document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
    };
    setAfterTitle(
      <nav
        className="flex shrink-0 flex-nowrap items-center gap-1"
        aria-label="Jump to section"
      >
        {sections.map((s) => (
          <button
            key={s.id}
            type="button"
            onClick={() => scrollTo(s.id)}
            className="shrink-0 cursor-pointer px-2 py-0.5 font-mondwest text-display text-xs tracking-wider text-text-secondary hover:text-foreground border border-border/50 hover:border-foreground/30 transition-colors"
          >
            {s.label}
          </button>
        ))}
      </nav>,
    );
    return () => {
      setAfterTitle(null);
    };
  }, [vars, sections, setAfterTitle]);

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

  // Add a custom key: register an unset row in local state and open it for
  // editing. The value isn't persisted until the user types one and saves
  // (reusing the normal handleSave → PUT /api/env path); on save the backend
  // surfaces it back as a custom row, so the new entry is durable.
  const handleAddKey = (key: string) => {
    setVars((prev) =>
      prev && prev[key]
        ? prev
        : {
            ...(prev ?? {}),
            [key]: {
              is_set: false,
              redacted_value: null,
              description: "",
              url: null,
              category: "custom",
              is_password: true,
              tools: [],
              advanced: false,
              custom: true,
            },
          },
    );
    setEdits((prev) => ({ ...prev, [key]: "" }));
  };

  /* ---- Build provider groups ---- */
  const { providerGroups, nonProviderGrouped, customEntries } = useMemo(() => {
    if (!vars)
      return {
        providerGroups: [],
        nonProviderGrouped: [],
        customEntries: [] as [string, EnvVarInfo][],
      };

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

    // Non-provider categories — use translated labels. Platform credentials
    // (channel_managed) are configured on the Channels page, so the messaging
    // category here is trimmed down to cross-cutting gateway / API / proxy
    // settings and relabelled accordingly.
    const CATEGORY_META_LABELS: Record<string, string> = {
      tool: t.app.nav.keys,
      messaging: t.common.gateway ?? "Gateway",
      setting: t.app.nav.config,
    };
    const CATEGORY_META_HINTS: Record<string, string | undefined> = {
      messaging:
        t.common.gatewayHint ??
        "Messaging platforms, the API server and webhooks are configured on the Channels page. These are gateway-wide settings (proxy/relay mode and the global allowlist).",
    };
    const otherCategories = ["tool", "messaging", "setting"];
    const nonProvider = otherCategories.map((cat) => {
      const entries = Object.entries(vars).filter(
        ([, info]) =>
          info.category === cat &&
          !info.channel_managed &&
          (showAdvanced || !info.advanced),
      );
      const setEntries = entries.filter(([, info]) => info.is_set);
      const unsetEntries = entries.filter(([, info]) => !info.is_set);
      return {
        label: CATEGORY_META_LABELS[cat] ?? cat,
        hint: CATEGORY_META_HINTS[cat],
        icon: CATEGORY_META_ICONS[cat] ?? KeyRound,
        category: cat,
        setEntries,
        unsetEntries,
        totalEntries: entries.length,
      };
    });

    // Custom keys: user-added vars the backend flagged as not in any catalog.
    // Sorted alphabetically; an in-flight (just-added, unsaved) row carries the
    // custom category locally so it shows here immediately.
    const customEntries = Object.entries(vars)
      .filter(([, info]) => info.category === "custom" && !info.channel_managed)
      .sort(([a], [b]) => a.localeCompare(b));

    return {
      providerGroups: groups,
      nonProviderGrouped: nonProvider,
      customEntries,
    };
  }, [vars, showAdvanced, t]);

  if (!vars) {
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner className="text-2xl text-primary" />
      </div>
    );
  }

  const totalProviders = providerGroups.length;
  const configuredProviders = providerGroups.filter((g) => g.hasAnySet).length;

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
          <p className="text-xs text-text-tertiary">
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

      <div id="section-oauth">
        <OAuthProvidersCard
          onError={(msg) => showToast(msg, "error")}
          onSuccess={(msg) => showToast(msg, "success")}
        />
      </div>

      <Card id="section-providers">
        <CardHeader className="border-b border-border bg-card">
          <div className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-muted-foreground" />
            <CardTitle className="text-base">{t.env.llmProviders}</CardTitle>
          </div>
          <CardDescription>
            {t.env.providersConfigured
              .replace("{configured}", String(configuredProviders))
              .replace("{total}", String(totalProviders))}
          </CardDescription>
        </CardHeader>

        <CardContent className="grid gap-0 p-0">
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

      {nonProviderGrouped.map((section) => {
        if (section.totalEntries === 0) return null;

        return (
          <EnvCategoryCard
            key={section.category}
            section={section}
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
        );
      })}
      <CustomKeysCard
        entries={customEntries}
        edits={edits}
        setEdits={setEdits}
        revealed={revealed}
        saving={saving}
        onSave={handleSave}
        onClear={keyClear.requestDelete}
        onReveal={handleReveal}
        onCancelEdit={cancelEdit}
        onAddKey={handleAddKey}
        clearDialogOpen={keyClear.isOpen}
      />
      <PluginSlot name="env:bottom" />
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  EnvCategoryCard — keys / messaging / settings sections             */
/* ------------------------------------------------------------------ */

function EnvCategoryCard({
  section,
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
  section: {
    category: string;
    hint?: string;
    icon: React.ComponentType<{ className?: string }>;
    label: string;
    setEntries: [string, EnvVarInfo][];
    totalEntries: number;
    unsetEntries: [string, EnvVarInfo][];
  };
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
  const noneConfigured = section.setEntries.length === 0;
  const [showAll, setShowAll] = useState(noneConfigured);
  const { t } = useI18n();
  const Icon = section.icon;
  const hasContent = section.setEntries.length > 0 || showAll;
  const rowProps = {
    edits,
    setEdits,
    revealed,
    saving,
    onSave,
    onClear,
    onReveal,
    onCancelEdit,
    clearDialogOpen,
  };

  return (
    <Card id={`section-${section.category}`}>
      <CardHeader
        className={`bg-card${hasContent ? " border-b border-border" : ""}`}
      >
        <div className="flex items-center justify-between gap-3">
          <div className="flex min-w-0 items-center gap-2">
            <Icon className="h-5 w-5 shrink-0 text-muted-foreground" />
            <CardTitle className="text-base">{section.label}</CardTitle>
          </div>

          {section.unsetEntries.length > 0 && (
            <button
              type="button"
              onClick={() => setShowAll((open) => !open)}
              aria-expanded={showAll}
              className="shrink-0 cursor-pointer border-0 bg-transparent p-0 font-mondwest text-xs tracking-[0.08em] text-text-secondary transition-colors hover:text-foreground"
            >
              {showAll ? t.env.showLess : t.env.showMore}
            </button>
          )}
        </div>

        <CardDescription>
          {section.setEntries.length} {t.common.of} {section.totalEntries}{" "}
          {t.common.configured}
        </CardDescription>

        {section.hint && (
          <CardDescription className="text-text-tertiary">
            {section.hint}
          </CardDescription>
        )}
      </CardHeader>

      {hasContent && (
        <CardContent className="grid gap-3 overflow-hidden pt-4">
          {section.setEntries.map(([key, info]) => (
            <EnvVarRow key={key} varKey={key} info={info} {...rowProps} />
          ))}

          {showAll &&
            section.unsetEntries.map(([key, info]) => (
              <EnvVarRow key={key} varKey={key} info={info} {...rowProps} />
            ))}
        </CardContent>
      )}
    </Card>
  );
}
