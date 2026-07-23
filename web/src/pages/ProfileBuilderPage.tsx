import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { H2 } from "@nous-research/ui/ui/components/typography/h2";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Input } from "@nous-research/ui/ui/components/input";
import { Label } from "@nous-research/ui/ui/components/label";
import { Checkbox } from "@nous-research/ui/ui/components/checkbox";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { api } from "@/lib/api";
import type {
  McpHttpAuth,
  McpServerCreate,
  SkillInfo,
  SkillHubResult,
} from "@/lib/api";
import {
  buildMcpServerCreate,
  emptyMcpServerDraft,
  McpServerCreateError,
  type McpServerDraft,
  type McpTransport,
} from "@/lib/mcp-server-create";
import { cn } from "@/lib/utils";
import { useI18n } from "@/i18n";

// Profile name rule mirrors the backend (`^[a-z0-9][a-z0-9_-]{0,63}$`).
const PROFILE_NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/;

type StepId = "identity" | "model" | "skills" | "mcp" | "review";

const STEPS: StepId[] = ["identity", "model", "skills", "mcp", "review"];

interface ModelChoice {
  provider: string;
  model: string;
  label: string;
}

/**
 * Dashboard-native, full-featured profile builder.
 *
 * Composes the same elements the standalone Models / Skills / MCP pages
 * manage — Name, Description, Model+Provider, Skills (built-in/optional +
 * hub), MCP servers — into one stepped create flow. Nothing is written to
 * disk until "Create profile" on the final step; the single POST /api/profiles
 * call commits model + MCPs + skill selection synchronously and spawns any
 * hub-skill installs (which the success toast reports as in-progress).
 *
 * Skills use REPLACE semantics: the default bundle is seeded server-side, then
 * every seeded skill the user did NOT keep is disabled. The "Start from full
 * bundle" toggle keeps everything (sends no keep list).
 */
export default function ProfileBuilderPage() {
  const { format, t } = useI18n();
  const navigate = useNavigate();
  const { toast, showToast } = useToast();

  const [step, setStep] = useState<StepId>("identity");

  // ── Step 1: identity ──────────────────────────────────────────────
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");

  // ── Step 2: model ─────────────────────────────────────────────────
  const [modelChoices, setModelChoices] = useState<ModelChoice[] | null>(null);
  const [modelChoice, setModelChoice] = useState(""); // `${provider}\u0000${model}`
  const [modelFilter, setModelFilter] = useState("");
  const modelLoading = useRef(false);

  // ── Step 3: skills ────────────────────────────────────────────────
  const [skills, setSkills] = useState<SkillInfo[] | null>(null);
  // keepAll = true: don't send a keep list (full bundle stays active).
  const [keepAll, setKeepAll] = useState(true);
  const [keptSkills, setKeptSkills] = useState<Set<string>>(new Set());
  const [skillFilter, setSkillFilter] = useState("");
  const skillsLoading = useRef(false);
  // Hub search
  const [hubQuery, setHubQuery] = useState("");
  const [hubResults, setHubResults] = useState<SkillHubResult[]>([]);
  const [hubSearching, setHubSearching] = useState(false);
  const [hubSkills, setHubSkills] = useState<SkillHubResult[]>([]);

  // ── Step 4: MCPs ──────────────────────────────────────────────────
  const [mcpServers, setMcpServers] = useState<McpServerCreate[]>([]);
  const [mcpDraft, setMcpDraft] = useState<McpServerDraft>(emptyMcpServerDraft);

  // ── Submit ────────────────────────────────────────────────────────
  const [creating, setCreating] = useState(false);

  const nameValid = PROFILE_NAME_RE.test(name.trim());

  // Lazy-load model choices when the model step is first shown.
  const loadModels = useCallback(() => {
    if (modelChoices !== null || modelLoading.current) return;
    modelLoading.current = true;
    api
      .getModelOptions()
      .then((res) => {
        const flat: ModelChoice[] = [];
        for (const prov of res.providers ?? []) {
          for (const m of prov.models ?? []) {
            flat.push({
              provider: prov.slug,
              model: m,
              label: `${prov.name} · ${m}`,
            });
          }
        }
        setModelChoices(flat);
      })
      .catch(() => setModelChoices([]))
      .finally(() => {
        modelLoading.current = false;
      });
  }, [modelChoices]);

  const loadSkills = useCallback(() => {
    if (skills !== null || skillsLoading.current) return;
    skillsLoading.current = true;
    api
      .getSkills()
      .then((res) => {
        setSkills(res);
        // Default keep = all currently-enabled skills (matches the seeded set).
        setKeptSkills(new Set(res.filter((s) => s.enabled).map((s) => s.name)));
      })
      .catch(() => setSkills([]))
      .finally(() => {
        skillsLoading.current = false;
      });
  }, [skills]);

  useEffect(() => {
    if (step === "model") loadModels();
    if (step === "skills") loadSkills();
  }, [step, loadModels, loadSkills]);

  const runHubSearch = useCallback(() => {
    const q = hubQuery.trim();
    if (!q) return;
    setHubSearching(true);
    api
      .searchSkillsHub(q, "all", 20)
      .then((res) => setHubResults(res.results ?? []))
      .catch(() => setHubResults([]))
      .finally(() => setHubSearching(false));
  }, [hubQuery]);

  const toggleKeep = (skillName: string) => {
    setKeptSkills((prev) => {
      const next = new Set(prev);
      if (next.has(skillName)) next.delete(skillName);
      else next.add(skillName);
      return next;
    });
  };

  const addHubSkill = (r: SkillHubResult) => {
    setHubSkills((prev) =>
      prev.some((x) => x.identifier === r.identifier) ? prev : [...prev, r],
    );
  };
  const removeHubSkill = (identifier: string) =>
    setHubSkills((prev) => prev.filter((x) => x.identifier !== identifier));

  const addMcpDraft = () => {
    let entry: McpServerCreate;
    try {
      entry = buildMcpServerCreate(mcpDraft);
    } catch (error) {
      const validationCopy = {
        "name-required": t.mcp.nameRequired,
        "url-required": t.mcp.urlRequired,
        "bearer-token-required": t.mcp.bearerTokenRequired,
        "command-required": t.mcp.commandRequired,
      } as const;
      showToast(
        error instanceof McpServerCreateError
          ? validationCopy[error.code]
          : t.mcp.invalidServer,
        "error",
      );
      return;
    }
    setMcpServers((prev) => [
      ...prev.filter((server) => server.name !== entry.name),
      entry,
    ]);
    setMcpDraft(emptyMcpServerDraft());
  };
  const removeMcp = (n: string) =>
    setMcpServers((prev) => prev.filter((s) => s.name !== n));

  const setMcpTransport = (transport: McpTransport) => {
    setMcpDraft((draft) =>
      transport === "http"
        ? { ...draft, transport, command: "", args: "", env: "" }
        : {
            ...draft,
            transport,
            url: "",
            httpAuth: "none",
            bearerToken: "",
          },
    );
  };

  const setMcpHttpAuth = (httpAuth: McpHttpAuth) => {
    setMcpDraft((draft) => ({
      ...draft,
      httpAuth,
      bearerToken: httpAuth === "header" ? draft.bearerToken : "",
    }));
  };

  const filteredModels = useMemo(() => {
    if (!modelChoices) return [];
    const f = modelFilter.trim().toLowerCase();
    if (!f) return modelChoices;
    return modelChoices.filter((c) => c.label.toLowerCase().includes(f));
  }, [modelChoices, modelFilter]);

  const filteredSkills = useMemo(() => {
    if (!skills) return [];
    const f = skillFilter.trim().toLowerCase();
    if (!f) return skills;
    return skills.filter(
      (s) =>
        s.name.toLowerCase().includes(f) ||
        (s.description || "").toLowerCase().includes(f) ||
        (s.category || "").toLowerCase().includes(f),
    );
  }, [skills, skillFilter]);

  const pickedModel = useMemo(
    () =>
      modelChoice
        ? modelChoices?.find(
            (c) => `${c.provider}\u0000${c.model}` === modelChoice,
          )
        : undefined,
    [modelChoice, modelChoices],
  );

  const handleCreate = async () => {
    const n = name.trim();
    if (!PROFILE_NAME_RE.test(n)) {
      showToast(t.profileBuilder.invalidName, "error");
      setStep("identity");
      return;
    }
    setCreating(true);
    try {
      const res = await api.createProfile({
        name: n,
        clone_from: null,
        description: description.trim() || undefined,
        provider: pickedModel?.provider,
        model: pickedModel?.model,
        mcp_servers: mcpServers.length ? mcpServers : undefined,
        keep_skills: keepAll ? undefined : Array.from(keptSkills),
        hub_skills: hubSkills.length
          ? hubSkills.map((s) => s.identifier)
          : undefined,
      });
      const pending = (res.hub_installs ?? []).filter((h) => h.pid).length;
      showToast(
        format(
          pending
            ? t.profileBuilder.createdInstalling
            : t.profileBuilder.created,
          { count: pending, name: n },
        ),
        "success",
      );
      navigate("/profiles");
    } catch (e) {
      showToast(
        format(t.profileBuilder.createFailed, { error: String(e) }),
        "error",
      );
    } finally {
      setCreating(false);
    }
  };

  const stepIndex = STEPS.indexOf(step);
  const canAdvance = step !== "identity" || nameValid;

  return (
    <div className="mx-auto w-full max-w-3xl space-y-6 p-4">
      <div className="flex items-center justify-between">
        <H2>{t.profileBuilder.title}</H2>
        <Button ghost onClick={() => navigate("/profiles")}>
          {t.profileBuilder.cancel}
        </Button>
      </div>

      {/* Stepper */}
      <div className="flex items-center gap-2 text-sm">
        {STEPS.map((stepId, i) => (
          <button
            key={stepId}
            // Identity must be valid before jumping ahead.
            disabled={i > 0 && !nameValid}
            onClick={() => setStep(stepId)}
            className={cn(
              "rounded-full px-3 py-1 transition-colors",
              stepId === step
                ? "bg-primary text-primary-foreground"
                : i <= stepIndex
                  ? "bg-muted text-foreground"
                  : "text-muted-foreground",
              i > 0 && !nameValid && "cursor-not-allowed opacity-50",
            )}
          >
            {i + 1}. {t.profileBuilder.steps[stepId]}
          </button>
        ))}
      </div>

      <Card>
        <CardContent className="space-y-4 p-5">
          {step === "identity" && (
            <div className="space-y-4">
              <div className="space-y-1.5">
                <Label htmlFor="pb-name">{t.profileBuilder.profileName}</Label>
                <Input
                  id="pb-name"
                  placeholder="coder"
                  value={name}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                    setName(e.target.value)
                  }
                />
                {name && !nameValid && (
                  <p className="text-xs text-destructive">
                    {t.profileBuilder.nameRule}
                  </p>
                )}
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="pb-desc">
                  {t.profileBuilder.descriptionOptional}
                </Label>
                <Input
                  id="pb-desc"
                  placeholder={t.profileBuilder.descriptionPlaceholder}
                  value={description}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                    setDescription(e.target.value)
                  }
                />
              </div>
            </div>
          )}

          {step === "model" && (
            <div className="space-y-3">
              <p className="text-sm text-muted-foreground">
                {t.profileBuilder.modelHint}
              </p>
              <Input
                placeholder={t.profileBuilder.filterModels}
                value={modelFilter}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                  setModelFilter(e.target.value)
                }
              />
              {modelChoices === null ? (
                <p className="text-sm text-muted-foreground">
                  {t.profileBuilder.loadingModels}
                </p>
              ) : (
                <div className="max-h-72 space-y-1 overflow-y-auto">
                  <button
                    onClick={() => setModelChoice("")}
                    className={cn(
                      "block w-full rounded px-3 py-2 text-left text-sm",
                      modelChoice === "" ? "bg-primary/10" : "hover:bg-muted",
                    )}
                  >
                    {t.profileBuilder.useDefault}
                  </button>
                  {filteredModels.map((c) => {
                    const key = `${c.provider}\u0000${c.model}`;
                    return (
                      <button
                        key={key}
                        onClick={() => setModelChoice(key)}
                        className={cn(
                          "block w-full rounded px-3 py-2 text-left text-sm",
                          modelChoice === key
                            ? "bg-primary/10"
                            : "hover:bg-muted",
                        )}
                      >
                        {c.label}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {step === "skills" && (
            <div className="space-y-4">
              <label className="flex items-center gap-2 text-sm">
                <Checkbox
                  checked={keepAll}
                  onCheckedChange={(v) => setKeepAll(Boolean(v))}
                />
                {t.profileBuilder.keepFullSkills}
              </label>
              {!keepAll && (
                <div className="space-y-2">
                  <p className="text-xs text-muted-foreground">
                    {t.profileBuilder.chooseSkillsHint}
                  </p>
                  <Input
                    placeholder={t.profileBuilder.filterSkills}
                    value={skillFilter}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                      setSkillFilter(e.target.value)
                    }
                  />
                  {skills === null ? (
                    <p className="text-sm text-muted-foreground">
                      {t.profileBuilder.loadingSkills}
                    </p>
                  ) : (
                    <div className="max-h-56 space-y-1 overflow-y-auto">
                      {filteredSkills.map((s) => (
                        <label
                          key={s.name}
                          className="flex items-start gap-2 rounded px-2 py-1.5 text-sm hover:bg-muted"
                        >
                          <Checkbox
                            checked={keptSkills.has(s.name)}
                            onCheckedChange={() => toggleKeep(s.name)}
                          />
                          <span className="flex-1">
                            <span className="font-medium">{s.name}</span>
                            {s.category && (
                              <Badge tone="secondary" className="ml-2">
                                {s.category}
                              </Badge>
                            )}
                            {s.description && (
                              <span className="block text-xs text-muted-foreground">
                                {s.description}
                              </span>
                            )}
                          </span>
                        </label>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Skills hub */}
              <div className="space-y-2 border-t pt-4">
                <Label>{t.profileBuilder.addFromHub}</Label>
                <div className="flex gap-2">
                  <Input
                    placeholder={t.profileBuilder.searchHub}
                    value={hubQuery}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                      setHubQuery(e.target.value)
                    }
                    onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => {
                      if (e.key === "Enter") runHubSearch();
                    }}
                  />
                  <Button
                    outlined
                    onClick={runHubSearch}
                    disabled={hubSearching}
                  >
                    {hubSearching
                      ? t.profileBuilder.searching
                      : t.profileBuilder.search}
                  </Button>
                </div>
                {hubResults.length > 0 && (
                  <div className="max-h-48 space-y-1 overflow-y-auto">
                    {hubResults.map((r) => (
                      <div
                        key={r.identifier}
                        className="flex items-center justify-between rounded px-2 py-1.5 text-sm hover:bg-muted"
                      >
                        <span className="flex-1">
                          <span className="font-medium">{r.name}</span>
                          <Badge tone="secondary" className="ml-2">
                            {r.source}
                          </Badge>
                          {r.description && (
                            <span className="block text-xs text-muted-foreground">
                              {r.description}
                            </span>
                          )}
                        </span>
                        <Button size="sm" ghost onClick={() => addHubSkill(r)}>
                          {t.profileBuilder.add}
                        </Button>
                      </div>
                    ))}
                  </div>
                )}
                {hubSkills.length > 0 && (
                  <div className="flex flex-wrap gap-2 pt-1">
                    {hubSkills.map((r) => (
                      <Badge key={r.identifier} className="gap-1">
                        {r.name}
                        <button
                          className="ml-1 text-xs"
                          onClick={() => removeHubSkill(r.identifier)}
                          aria-label={format(t.profileBuilder.removeNamed, {
                            name: r.name,
                          })}
                        >
                          ×
                        </button>
                      </Badge>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {step === "mcp" && (
            <div className="space-y-5">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="space-y-1">
                  <h3 className="font-expanded text-base font-bold tracking-[0.04em]">
                    {t.profileBuilder.mcpServers}
                  </h3>
                  <p className="text-sm text-muted-foreground">
                    {t.profileBuilder.mcpHint}
                  </p>
                </div>
                <span
                  className="text-xs text-muted-foreground"
                  aria-live="polite"
                >
                  {format(t.profileBuilder.mcpConfigured, {
                    count: mcpServers.length,
                  })}
                </span>
              </div>

              <div className="space-y-4 border border-border bg-background/20 p-4 md:p-5">
                <h4 className="font-medium">{t.profileBuilder.addServer}</h4>

                <div className="grid gap-4 md:grid-cols-2">
                  <div className="grid gap-1.5">
                    <Label htmlFor="pb-mcp-name">
                      {t.profileBuilder.serverName}
                    </Label>
                    <Input
                      id="pb-mcp-name"
                      placeholder={t.profileBuilder.serverNamePlaceholder}
                      value={mcpDraft.name}
                      onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                        setMcpDraft({ ...mcpDraft, name: e.target.value })
                      }
                    />
                  </div>
                  <div className="grid gap-1.5">
                    <Label>{t.mcp.transport}</Label>
                    <div
                      className="grid grid-cols-2 border border-border bg-background/30 p-0.5"
                      role="group"
                      aria-label={t.mcp.transport}
                    >
                      {(
                        [
                          ["http", t.mcp.httpSse],
                          ["stdio", t.mcp.stdio],
                        ] as const
                      ).map(([value, label]) => (
                        <button
                          key={value}
                          type="button"
                          aria-pressed={mcpDraft.transport === value}
                          className={cn(
                            "px-3 py-2 text-sm font-medium transition-colors",
                            mcpDraft.transport === value
                              ? "bg-primary text-primary-foreground"
                              : "text-muted-foreground hover:bg-muted hover:text-foreground",
                          )}
                          onClick={() => setMcpTransport(value)}
                        >
                          {label}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>

                {mcpDraft.transport === "http" ? (
                  <>
                    <div className="grid gap-1.5">
                      <Label htmlFor="pb-mcp-url">{t.mcp.url}</Label>
                      <Input
                        id="pb-mcp-url"
                        placeholder="https://example.com/mcp"
                        value={mcpDraft.url}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                          setMcpDraft({ ...mcpDraft, url: e.target.value })
                        }
                      />
                    </div>
                    <div className="grid gap-1.5">
                      <Label>{t.mcp.authentication}</Label>
                      <div
                        className="grid grid-cols-3 border border-border bg-background/30 p-0.5 md:max-w-md"
                        role="group"
                        aria-label={t.mcp.authentication}
                      >
                        {(
                          [
                            ["none", t.mcp.authNone],
                            ["header", t.mcp.bearerToken],
                            ["oauth", "OAuth"],
                          ] as const
                        ).map(([value, label]) => (
                          <button
                            key={value}
                            type="button"
                            aria-pressed={mcpDraft.httpAuth === value}
                            className={cn(
                              "px-2 py-2 text-sm font-medium transition-colors",
                              mcpDraft.httpAuth === value
                                ? "bg-primary text-primary-foreground"
                                : "text-muted-foreground hover:bg-muted hover:text-foreground",
                            )}
                            onClick={() => setMcpHttpAuth(value)}
                          >
                            {label}
                          </button>
                        ))}
                      </div>
                    </div>
                    {mcpDraft.httpAuth === "header" && (
                      <div className="grid gap-1.5">
                        <Label htmlFor="pb-mcp-bearer-token">
                          {t.mcp.bearerToken}
                        </Label>
                        <Input
                          id="pb-mcp-bearer-token"
                          type="password"
                          autoComplete="new-password"
                          placeholder={t.mcp.bearerTokenPlaceholder}
                          value={mcpDraft.bearerToken}
                          onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                            setMcpDraft({
                              ...mcpDraft,
                              bearerToken: e.target.value,
                            })
                          }
                        />
                        <p className="text-xs text-muted-foreground">
                          {t.profileBuilder.bearerTokenStorageHint}
                        </p>
                      </div>
                    )}
                    {mcpDraft.httpAuth === "oauth" && (
                      <p className="text-xs text-muted-foreground">
                        {t.profileBuilder.oauthAfterCreateHint}
                      </p>
                    )}
                  </>
                ) : (
                  <>
                    <div className="grid gap-4 md:grid-cols-2">
                      <div className="grid gap-1.5">
                        <Label htmlFor="pb-mcp-command">{t.mcp.command}</Label>
                        <Input
                          id="pb-mcp-command"
                          placeholder="npx"
                          value={mcpDraft.command}
                          onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                            setMcpDraft({
                              ...mcpDraft,
                              command: e.target.value,
                            })
                          }
                        />
                      </div>
                      <div className="grid gap-1.5">
                        <Label htmlFor="pb-mcp-args">{t.mcp.args}</Label>
                        <Input
                          id="pb-mcp-args"
                          placeholder="-y @modelcontextprotocol/server"
                          value={mcpDraft.args}
                          onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                            setMcpDraft({ ...mcpDraft, args: e.target.value })
                          }
                        />
                      </div>
                    </div>
                    <div className="grid gap-1.5">
                      <Label htmlFor="pb-mcp-env">{t.mcp.environment}</Label>
                      <textarea
                        id="pb-mcp-env"
                        className="flex min-h-[80px] w-full border border-border bg-background/40 px-3 py-2 text-sm font-courier shadow-sm placeholder:text-muted-foreground focus-visible:border-foreground/25 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/30"
                        placeholder={"API_KEY=secret\nDEBUG=1"}
                        value={mcpDraft.env}
                        onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) =>
                          setMcpDraft({ ...mcpDraft, env: e.target.value })
                        }
                      />
                    </div>
                  </>
                )}

                <div className="flex justify-end">
                  <Button onClick={addMcpDraft}>
                    {t.profileBuilder.addServer}
                  </Button>
                </div>
              </div>

              {mcpServers.length > 0 && (
                <div className="space-y-2">
                  {mcpServers.map((s) => (
                    <div
                      key={s.name}
                      className="flex items-center justify-between gap-4 border border-border bg-muted/40 p-4 text-sm"
                    >
                      <span className="min-w-0">
                        <span className="flex flex-wrap items-center gap-2">
                          <span className="font-medium">{s.name}</span>
                          <Badge tone="outline">
                            {s.url ? "HTTP" : "stdio"}
                          </Badge>
                          {s.auth && (
                            <Badge tone="outline">
                              {t.mcp.authLabel}{" "}
                              {s.auth === "header" ? t.mcp.bearerAuth : s.auth}
                            </Badge>
                          )}
                        </span>
                        <span className="mt-1 block break-all text-xs text-muted-foreground">
                          {s.url || [s.command, ...(s.args || [])].join(" ")}
                        </span>
                      </span>
                      <Button
                        size="sm"
                        ghost
                        destructive
                        className="shrink-0"
                        onClick={() => removeMcp(s.name)}
                      >
                        {t.profileBuilder.remove}
                      </Button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
          {step === "review" && (
            <div className="space-y-3 text-sm">
              <ReviewRow
                label={t.profileBuilder.name}
                value={name.trim() || "—"}
              />
              <ReviewRow
                label={t.profileBuilder.description}
                value={description.trim() || "—"}
              />
              <ReviewRow
                label={t.profileBuilder.model}
                value={
                  pickedModel
                    ? pickedModel.label
                    : t.profileBuilder.defaultLater
                }
              />
              <ReviewRow
                label={t.profileBuilder.skills}
                value={
                  keepAll
                    ? t.profileBuilder.fullDefaultBundle
                    : format(t.profileBuilder.keptSkills, {
                        count: keptSkills.size,
                      }) +
                      (hubSkills.length
                        ? format(t.profileBuilder.hubAddition, {
                            count: hubSkills.length,
                          })
                        : "")
                }
              />
              {!keepAll && hubSkills.length > 0 && (
                <p className="pl-24 text-xs text-muted-foreground">
                  {t.profileBuilder.hub}{" "}
                  {hubSkills.map((s) => s.name).join(", ")}
                </p>
              )}
              {keepAll && hubSkills.length > 0 && (
                <ReviewRow
                  label={t.profileBuilder.hubSkills}
                  value={hubSkills.map((s) => s.name).join(", ")}
                />
              )}
              <ReviewRow
                label={t.profileBuilder.mcpServers}
                value={
                  mcpServers.length
                    ? mcpServers.map((s) => s.name).join(", ")
                    : t.profileBuilder.none
                }
              />
            </div>
          )}
        </CardContent>
      </Card>

      {/* Nav buttons */}
      <div className="flex items-center justify-between">
        <Button
          ghost
          disabled={stepIndex === 0}
          onClick={() => setStep(STEPS[Math.max(0, stepIndex - 1)])}
        >
          {t.profileBuilder.back}
        </Button>
        {step === "review" ? (
          <Button onClick={handleCreate} disabled={creating || !nameValid}>
            {creating
              ? t.profileBuilder.creating
              : t.profileBuilder.createProfile}
          </Button>
        ) : (
          <Button
            disabled={!canAdvance}
            onClick={() =>
              setStep(STEPS[Math.min(STEPS.length - 1, stepIndex + 1)])
            }
          >
            {t.profileBuilder.next}
          </Button>
        )}
      </div>

      <Toast toast={toast} />
    </div>
  );
}

function ReviewRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex gap-3">
      <span className="w-24 shrink-0 text-muted-foreground">{label}</span>
      <span className="flex-1 break-words">{value}</span>
    </div>
  );
}
