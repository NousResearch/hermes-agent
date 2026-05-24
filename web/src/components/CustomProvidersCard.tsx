import { useCallback, useEffect, useMemo, useState } from "react";
import { Plus, RefreshCw, Save, Trash2, X, Zap } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Checkbox } from "@nous-research/ui/ui/components/checkbox";
import { Select, SelectOption } from "@nous-research/ui/ui/components/select";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api } from "@/lib/api";
import type { CustomProviderInfo, CustomProviderPayload } from "@/lib/api";
import { useI18n } from "@/i18n";

const API_MODES = [
  "chat_completions",
  "codex_responses",
  "anthropic_messages",
  "bedrock_converse",
  "codex_app_server",
];

const DEFAULT_COPY = {
  title: "Custom Endpoints",
  description:
    "Manage user-defined model providers from config.yaml providers:. API keys are stored in .env through key_env.",
  configDescription:
    "These entries are the same providers: custom endpoints used by model routing. Manage secrets from Keys.",
  add: "Add endpoint",
  edit: "Edit",
  endpointKey: "Provider key",
  displayName: "Display name",
  baseUrl: "Base URL",
  keyEnv: "Key env",
  apiMode: "API mode",
  defaultModel: "Default model",
  models: "Models",
  modelsHelp: "One model per line or comma-separated.",
  contextLength: "Context length",
  discoverModels: "Discover models from /models",
  extraBody: "Extra request body JSON",
  apiKey: "API key",
  apiKeyHelp: "Optional. Saved to .env using key_env and never written to config.yaml.",
  clearStoredKey: "Clear stored API key",
  probe: "Probe models",
  noProviders: "No custom endpoints configured.",
  active: "Active",
  keySet: "key set",
  keyMissing: "no key",
  sourceConfig: "config key",
  sourceEnv: "env key",
  create: "Create endpoint",
  update: "Update endpoint",
  deleteTitle: "Delete custom endpoint?",
  deleteMessage: "This removes the providers: entry and its key_env value from .env.",
  saved: "Custom provider saved",
  removed: "Custom provider removed",
  probeLoaded: "Model list loaded",
};

interface ProviderForm {
  key: string;
  name: string;
  baseUrl: string;
  keyEnv: string;
  apiMode: string;
  defaultModel: string;
  modelsText: string;
  contextLength: string;
  discoverModels: boolean;
  extraBodyText: string;
  apiKey: string;
  clearApiKey: boolean;
}

function emptyForm(): ProviderForm {
  return {
    key: "",
    name: "",
    baseUrl: "",
    keyEnv: "",
    apiMode: "chat_completions",
    defaultModel: "",
    modelsText: "",
    contextLength: "",
    discoverModels: false,
    extraBodyText: "",
    apiKey: "",
    clearApiKey: false,
  };
}

function providerToForm(provider: CustomProviderInfo): ProviderForm {
  const extraBody =
    provider.extra_body && Object.keys(provider.extra_body).length > 0
      ? JSON.stringify(provider.extra_body, null, 2)
      : "";
  return {
    key: provider.key,
    name: provider.name || provider.key,
    baseUrl: provider.base_url,
    keyEnv: provider.key_env,
    apiMode: provider.api_mode || "chat_completions",
    defaultModel: provider.default_model || "",
    modelsText: provider.models.map((model) => model.name).join("\n"),
    contextLength: provider.context_length ? String(provider.context_length) : "",
    discoverModels: provider.discover_models,
    extraBodyText: extraBody,
    apiKey: "",
    clearApiKey: false,
  };
}

function parseModels(value: string): string[] {
  return value
    .split(/[\n,]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function isActiveProvider(activeProvider: string | null, key: string): boolean {
  const active = (activeProvider || "").toLowerCase();
  const normalized = key.toLowerCase();
  return active === normalized || active === `custom:${normalized}`;
}

export function CustomProvidersCard({
  embedded = false,
  context = "keys",
  onError,
  onSuccess,
}: {
  embedded?: boolean;
  context?: "keys" | "config";
  onError?: (message: string) => void;
  onSuccess?: (message: string) => void;
}) {
  const { t } = useI18n();
  const copy = t.env.customProviders ?? DEFAULT_COPY;
  const [providers, setProviders] = useState<CustomProviderInfo[]>([]);
  const [activeProvider, setActiveProvider] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [probing, setProbing] = useState(false);
  const [editingKey, setEditingKey] = useState<string | null>(null);
  const [formOpen, setFormOpen] = useState(false);
  const [form, setForm] = useState<ProviderForm>(() => emptyForm());
  const [deleteKey, setDeleteKey] = useState<string | null>(null);

  const loadProviders = useCallback(async () => {
    setLoading(true);
    try {
      const response = await api.getCustomProviders();
      setProviders(response.providers);
      setActiveProvider(response.active_provider);
    } catch (error) {
      onError?.(`${copy.title}: ${error}`);
    } finally {
      setLoading(false);
    }
  }, [copy.title, onError]);

  useEffect(() => {
    void loadProviders();
  }, [loadProviders]);

  const configuredCount = providers.filter((provider) => provider.api_key_set).length;
  const summary = useMemo(
    () => `${configuredCount} / ${providers.length} ${t.common.configured}`,
    [configuredCount, providers.length, t.common.configured],
  );

  const setField = <K extends keyof ProviderForm>(key: K, value: ProviderForm[K]) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const startCreate = () => {
    setEditingKey(null);
    setForm(emptyForm());
    setFormOpen(true);
  };

  const startEdit = (provider: CustomProviderInfo) => {
    setEditingKey(provider.key);
    setForm(providerToForm(provider));
    setFormOpen(true);
  };

  const buildPayload = (): CustomProviderPayload | null => {
    let extraBody: Record<string, unknown> | undefined;
    if (form.extraBodyText.trim()) {
      try {
        const parsed = JSON.parse(form.extraBodyText);
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
          throw new Error("extra_body must be a JSON object");
        }
        extraBody = parsed as Record<string, unknown>;
      } catch (error) {
        onError?.(`${copy.extraBody}: ${error}`);
        return null;
      }
    }

    const contextLength = Number(form.contextLength);
    return {
      key: form.key.trim(),
      name: form.name.trim(),
      base_url: form.baseUrl.trim(),
      key_env: form.keyEnv.trim(),
      api_mode: form.apiMode,
      default_model: form.defaultModel.trim(),
      models: parseModels(form.modelsText),
      context_length:
        form.contextLength.trim() && Number.isFinite(contextLength)
          ? contextLength
          : null,
      discover_models: form.discoverModels,
      extra_body: extraBody,
      api_key: form.apiKey,
      clear_api_key: form.clearApiKey,
    };
  };

  const saveProvider = async () => {
    const payload = buildPayload();
    if (!payload) return;
    setSaving(true);
    try {
      if (editingKey) {
        await api.updateCustomProvider(editingKey, payload);
      } else {
        await api.createCustomProvider(payload);
      }
      onSuccess?.(copy.saved);
      setFormOpen(false);
      setEditingKey(null);
      await loadProviders();
    } catch (error) {
      onError?.(`${copy.title}: ${error}`);
    } finally {
      setSaving(false);
    }
  };

  const probeModels = async () => {
    if (!form.key.trim()) return;
    setProbing(true);
    try {
      const response = await api.probeCustomProvider(form.key.trim(), {
        key: form.key.trim(),
        base_url: form.baseUrl.trim(),
        key_env: form.keyEnv.trim(),
        api_key: form.apiKey,
      });
      if (response.models.length > 0) {
        setField("modelsText", response.models.join("\n"));
      }
      onSuccess?.(copy.probeLoaded);
    } catch (error) {
      onError?.(`${copy.probe}: ${error}`);
    } finally {
      setProbing(false);
    }
  };

  const deleteProvider = async () => {
    if (!deleteKey) return;
    setSaving(true);
    try {
      await api.deleteCustomProvider(deleteKey);
      onSuccess?.(copy.removed);
      setDeleteKey(null);
      await loadProviders();
    } catch (error) {
      onError?.(`${copy.title}: ${error}`);
    } finally {
      setSaving(false);
    }
  };

  const content = (
    <>
      <ConfirmDialog
        open={!!deleteKey}
        title={copy.deleteTitle}
        description={copy.deleteMessage}
        destructive
        loading={saving}
        cancelLabel={t.common.cancel}
        confirmLabel={t.common.delete}
        onCancel={() => setDeleteKey(null)}
        onConfirm={deleteProvider}
      />

      <div className={embedded ? "border-b border-border p-4" : "grid gap-4"}>
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <Zap className="h-4 w-4 text-muted-foreground" />
              <h3 className="text-sm font-semibold">{copy.title}</h3>
              <Badge tone="secondary" className="text-[10px]">
                {summary}
              </Badge>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              {context === "config" ? copy.configDescription : copy.description}
            </p>
          </div>
          <Button size="sm" outlined prefix={<Plus />} onClick={startCreate}>
            {copy.add}
          </Button>
        </div>

        {loading ? (
          <div className="flex items-center gap-2 py-4 text-sm text-muted-foreground">
            <Spinner />
            {t.common.loading}
          </div>
        ) : providers.length === 0 && !formOpen ? (
          <p className="py-4 text-sm text-muted-foreground">{copy.noProviders}</p>
        ) : (
          <div className="grid gap-3 pt-4">
            {providers.map((provider) => {
              const active = isActiveProvider(activeProvider, provider.key);
              return (
                <div
                  key={provider.key}
                  className="grid gap-3 border border-border bg-background/30 p-3"
                >
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="font-mono-ui text-sm font-semibold">
                          {provider.name || provider.key}
                        </span>
                        <Badge tone="secondary" className="text-[10px]">
                          {provider.key}
                        </Badge>
                        {active && (
                          <Badge tone="success" className="text-[10px]">
                            {copy.active}
                          </Badge>
                        )}
                        <Badge
                          tone={provider.api_key_set ? "success" : "secondary"}
                          className="text-[10px]"
                        >
                          {provider.api_key_set ? copy.keySet : copy.keyMissing}
                        </Badge>
                      </div>
                      <p className="mt-1 truncate font-mono-ui text-xs text-muted-foreground">
                        {provider.base_url}
                      </p>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      <Button size="sm" outlined onClick={() => startEdit(provider)}>
                        {copy.edit}
                      </Button>
                      <Button
                        size="sm"
                        outlined
                        destructive
                        prefix={<Trash2 />}
                        disabled={active}
                        onClick={() => setDeleteKey(provider.key)}
                      >
                        {t.common.delete}
                      </Button>
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-2 text-[11px] text-muted-foreground">
                    <span className="border border-border px-2 py-1">{provider.api_mode}</span>
                    {provider.key_env && (
                      <span className="border border-border px-2 py-1">
                        {provider.key_env}
                      </span>
                    )}
                    {provider.default_model && (
                      <span className="border border-border px-2 py-1">
                        {provider.default_model}
                      </span>
                    )}
                    <span className="border border-border px-2 py-1">
                      {provider.model_count} {copy.models}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {formOpen && (
          <div className="mt-4 grid gap-4 border border-border bg-background/40 p-4">
            <div className="grid gap-4 md:grid-cols-2">
              <div className="grid gap-2">
                <Label htmlFor="custom-provider-key">{copy.endpointKey}</Label>
                <Input
                  id="custom-provider-key"
                  value={form.key}
                  onChange={(event) => setField("key", event.target.value)}
                  placeholder="openai-router"
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="custom-provider-name">{copy.displayName}</Label>
                <Input
                  id="custom-provider-name"
                  value={form.name}
                  onChange={(event) => setField("name", event.target.value)}
                  placeholder="OpenAI Router"
                />
              </div>
              <div className="grid gap-2 md:col-span-2">
                <Label htmlFor="custom-provider-base-url">{copy.baseUrl}</Label>
                <Input
                  id="custom-provider-base-url"
                  value={form.baseUrl}
                  onChange={(event) => setField("baseUrl", event.target.value)}
                  placeholder="https://example.com/v1"
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="custom-provider-key-env">{copy.keyEnv}</Label>
                <Input
                  id="custom-provider-key-env"
                  value={form.keyEnv}
                  onChange={(event) => setField("keyEnv", event.target.value)}
                  placeholder="CUSTOM_PROVIDER_API_KEY"
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="custom-provider-api-mode">{copy.apiMode}</Label>
                <Select
                  id="custom-provider-api-mode"
                  value={form.apiMode}
                  onValueChange={(value) => setField("apiMode", value)}
                >
                  {API_MODES.map((mode) => (
                    <SelectOption key={mode} value={mode}>
                      {mode}
                    </SelectOption>
                  ))}
                </Select>
              </div>
              <div className="grid gap-2">
                <Label htmlFor="custom-provider-default-model">{copy.defaultModel}</Label>
                <Input
                  id="custom-provider-default-model"
                  value={form.defaultModel}
                  onChange={(event) => setField("defaultModel", event.target.value)}
                  placeholder="gpt-4.1"
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="custom-provider-context-length">{copy.contextLength}</Label>
                <Input
                  id="custom-provider-context-length"
                  type="number"
                  value={form.contextLength}
                  onChange={(event) => setField("contextLength", event.target.value)}
                  placeholder="0"
                />
              </div>
              <div className="grid gap-2 md:col-span-2">
                <Label htmlFor="custom-provider-models">{copy.models}</Label>
                <textarea
                  id="custom-provider-models"
                  className="flex min-h-[90px] w-full border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                  value={form.modelsText}
                  onChange={(event) => setField("modelsText", event.target.value)}
                  placeholder={"gpt-4.1\ngemini-2.5-pro"}
                />
                <p className="text-xs text-muted-foreground">{copy.modelsHelp}</p>
              </div>
              <div className="grid gap-2 md:col-span-2">
                <Label htmlFor="custom-provider-extra-body">{copy.extraBody}</Label>
                <textarea
                  id="custom-provider-extra-body"
                  className="flex min-h-[80px] w-full border border-input bg-transparent px-3 py-2 font-mono text-xs shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                  value={form.extraBodyText}
                  onChange={(event) => setField("extraBodyText", event.target.value)}
                  placeholder={'{"reasoning_effort":"medium"}'}
                />
              </div>
              <div className="grid gap-2 md:col-span-2">
                <Label htmlFor="custom-provider-api-key">{copy.apiKey}</Label>
                <Input
                  id="custom-provider-api-key"
                  type="password"
                  value={form.apiKey}
                  onChange={(event) => setField("apiKey", event.target.value)}
                  placeholder="sk-..."
                />
                <p className="text-xs text-muted-foreground">{copy.apiKeyHelp}</p>
              </div>
            </div>

            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex flex-col gap-2">
                <label className="flex items-center gap-2 text-sm">
                  <Checkbox
                    checked={form.discoverModels}
                    onCheckedChange={(checked) => setField("discoverModels", checked === true)}
                  />
                  {copy.discoverModels}
                </label>
                <label className="flex items-center gap-2 text-sm">
                  <Checkbox
                    checked={form.clearApiKey}
                    onCheckedChange={(checked) => setField("clearApiKey", checked === true)}
                  />
                  {copy.clearStoredKey}
                </label>
              </div>
              <div className="flex flex-wrap justify-end gap-2">
                <Button
                  size="sm"
                  outlined
                  prefix={probing ? <Spinner /> : <RefreshCw />}
                  onClick={probeModels}
                  disabled={probing || !form.key.trim()}
                >
                  {copy.probe}
                </Button>
                <Button
                  size="sm"
                  outlined
                  prefix={<X />}
                  onClick={() => {
                    setFormOpen(false);
                    setEditingKey(null);
                  }}
                >
                  {t.common.cancel}
                </Button>
                <Button
                  size="sm"
                  prefix={saving ? <Spinner /> : <Save />}
                  onClick={saveProvider}
                  disabled={saving}
                >
                  {editingKey ? copy.update : copy.create}
                </Button>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );

  if (embedded) return content;

  return (
    <Card>
      <CardHeader className="border-b border-border bg-card">
        <CardTitle className="text-base">{copy.title}</CardTitle>
        <CardDescription>
          {context === "config" ? copy.configDescription : copy.description}
        </CardDescription>
      </CardHeader>
      <CardContent className="p-0">{content}</CardContent>
    </Card>
  );
}
