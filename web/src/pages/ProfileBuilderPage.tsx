import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router";
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
  type McpServerDraft,
  type McpTransport,
} from "@/lib/mcp-server-create";
import { cn } from "@/lib/utils";
import { errorMessage } from "@/lib/api-error";
import { useI18n } from "@/i18n";
import { en } from "@/i18n/en";
import type { Translations } from "@/i18n/types";

/** Profile builder copy; en seeds the optional block, other locales fall back. */
type ProfileBuilderCopy = NonNullable<Translations["profileBuilder"]>;
function profileBuilderCopy(t: Translations): ProfileBuilderCopy {
  return t.profileBuilder ?? (en.profileBuilder as ProfileBuilderCopy);
}

// Profile name rule mirrors the backend (`^[a-z0-9][a-z0-9_-]{0,63}$`).
const PROFILE_NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/;

type StepId = "identity" | "model" | "skills" | "mcp" | "review";

const STEPS: { id: StepId; labelKey: keyof ProfileBuilderCopy }[] = [
  { id: "identity", labelKey: "stepIdentity" },
  { id: "model", labelKey: "stepModel" },
  { id: "skills", labelKey: "stepSkills" },
  { id: "mcp", labelKey: "stepMcp" },
  { id: "review", labelKey: "stepReview" },
];

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
  const navigate = useNavigate();
  const { toast, showToast } = useToast();
  const { t } = useI18n();
  const PB = profileBuilderCopy(t);

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
      showToast(
        error instanceof Error ? error.message : PB.invalidMcp,
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
      showToast(PB.invalidName, "error");
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
        pending
          ? PB.createdPending
              .replace("{name}", n)
              .replace("{count}", String(pending))
              .replace("{s}", pending === 1 ? "" : "s")
          : PB.created.replace("{name}", n),
        "success",
      );
      navigate("/profiles");
    } catch (e) {
      showToast(PB.createFailed.replace("{error}", errorMessage(e)), "error");
    } finally {
      setCreating(false);
    }
  };

  const stepIndex = STEPS.findIndex((s) => s.id === step);
  const canAdvance = step !== "identity" || nameValid;

  return (
    <div className="mx-auto w-full max-w-3xl space-y-6 p-4">
      <div className="flex items-center justify-between">
        <H2>{PB.newProfile}</H2>
        <Button ghost onClick={() => navigate("/profiles")}>
          {PB.cancel}
        </Button>
      </div>

      {/* Stepper */}
      <div className="flex items-center gap-2 text-sm">
        {STEPS.map((s, i) => (
          <button
            key={s.id}
            // Identity must be valid before jumping ahead.
            disabled={i > 0 && !nameValid}
            onClick={() => setStep(s.id)}
            className={cn(
              "rounded-full px-3 py-1 transition-colors",
              s.id === step
                ? "bg-primary text-primary-foreground"
                : i <= stepIndex
                  ? "bg-muted text-foreground"
                  : "text-muted-foreground",
              i > 0 && !nameValid && "cursor-not-allowed opacity-50",
            )}
          >
            {i + 1}. {PB[s.labelKey]}
          </button>
        ))}
      </div>

      <Card>
        <CardContent className="space-y-4 p-5">
          {step === "identity" && (
            <div className="space-y-4">
              <div className="space-y-1.5">
                <Label htmlFor="pb-name">{PB.profileName}</Label>
                <Input
                  id="pb-name"
                  placeholder={PB.namePlaceholder}
                  value={name}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                    setName(e.target.value)
                  }
                />
                {name && !nameValid && (
                  <p className="text-xs text-destructive">
                    {PB.nameHint}
                  </p>
                )}
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="pb-desc">{PB.descriptionLabel}</Label>
                <Input
                  id="pb-desc"
                  placeholder={PB.descriptionPlaceholder}
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
                {PB.modelHint}
              </p>
              <Input
                placeholder={PB.filterModels}
                value={modelFilter}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                  setModelFilter(e.target.value)
                }
              />
              {modelChoices === null ? (
                <p className="text-sm text-muted-foreground">{PB.loadingModels}</p>
              ) : (
                <div className="max-h-72 space-y-1 overflow-y-auto">
                  <button
                    onClick={() => setModelChoice("")}
                    className={cn(
                      "block w-full rounded px-3 py-2 text-left text-sm",
                      modelChoice === "" ? "bg-primary/10" : "hover:bg-muted",
                    )}
                  >
                    {PB.useDefault}
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
                {PB.keepAllLabel}
              </label>
              {!keepAll && (
                <div className="space-y-2">
                  <p className="text-xs text-muted-foreground">
                    {PB.keepHint}
                  </p>
                  <Input
                    placeholder={PB.filterSkills}
                    value={skillFilter}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                      setSkillFilter(e.target.value)
                    }
                  />
                  {skills === null ? (
                    <p className="text-sm text-muted-foreground">
                      {PB.loadingSkills}
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
                <Label>{PB.addFromHub}</Label>
                <div className="flex gap-2">
                  <Input
                    placeholder={PB.hubSearchPlaceholder}
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
                    {hubSearching ? PB.searching : PB.search}
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
                          {PB.add}
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
                          aria-label={PB.removeAria.replace("{name}", r.name)}
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
                    {PB.mcpHeading}
                  </h3>
                  <p className="text-sm text-muted-foreground">
                    {PB.mcpIntro}
                  </p>
                </div>
                <span
                  className="text-xs text-muted-foreground"
                  aria-live="polite"
                >
                  {PB.configuredCount.replace("{count}", String(mcpServers.length))}
                </span>
              </div>

              <div className="space-y-4 border border-border bg-background/20 p-4 md:p-5">
                <h4 className="font-medium">{PB.addServerHeading}</h4>

                <div className="grid gap-4 md:grid-cols-2">
                  <div className="grid gap-1.5">
                    <Label htmlFor="pb-mcp-name">{PB.serverName}</Label>
                    <Input
                      id="pb-mcp-name"
                      placeholder={PB.serverNamePlaceholder}
                      value={mcpDraft.name}
                      onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                        setMcpDraft({ ...mcpDraft, name: e.target.value })
                      }
                    />
                  </div>
                  <div className="grid gap-1.5">
                    <Label>{PB.transport}</Label>
                    <div
                      className="grid grid-cols-2 border border-border bg-background/30 p-0.5"
                      role="group"
                      aria-label="MCP transport"
                    >
                      {(
                        [
                          ["http", "HTTP/SSE"],
                          ["stdio", "stdio"],
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
                      <Label htmlFor="pb-mcp-url">URL</Label>
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
                      <Label>{PB.authentication}</Label>
                      <div
                        className="grid grid-cols-3 border border-border bg-background/30 p-0.5 md:max-w-md"
                        role="group"
                        aria-label="HTTP authentication"
                      >
                        {(
                          [
                            ["none", PB.authNone],
                            ["header", PB.authHeader],
                            ["oauth", PB.authOauth],
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
                          {PB.bearerToken}
                        </Label>
                        <Input
                          id="pb-mcp-bearer-token"
                          type="password"
                          autoComplete="new-password"
                          placeholder={PB.bearerPlaceholder}
                          value={mcpDraft.bearerToken}
                          onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                            setMcpDraft({
                              ...mcpDraft,
                              bearerToken: e.target.value,
                            })
                          }
                        />
                        <p className="text-xs text-muted-foreground">
                          {PB.bearerHint}
                        </p>
                      </div>
                    )}
                    {mcpDraft.httpAuth === "oauth" && (
                      <p className="text-xs text-muted-foreground">
                        {PB.oauthHint}
                      </p>
                    )}
                  </>
                ) : (
                  <>
                    <div className="grid gap-4 md:grid-cols-2">
                      <div className="grid gap-1.5">
                        <Label htmlFor="pb-mcp-command">{PB.command}</Label>
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
                        <Label htmlFor="pb-mcp-args">{PB.arguments}</Label>
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
                      <Label htmlFor="pb-mcp-env">
                        {PB.environment}
                      </Label>
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
                  <Button onClick={addMcpDraft}>{PB.addServer}</Button>
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
                              {PB.authPrefix}
                              {s.auth === "header" ? "bearer" : s.auth}
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
                        {PB.remove}
                      </Button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
          {step === "review" && (
            <div className="space-y-3 text-sm">
              <ReviewRow label={PB.reviewName} value={name.trim() || "—"} />
              <ReviewRow
                label={PB.reviewDescription}
                value={description.trim() || "—"}
              />
              <ReviewRow
                label={PB.reviewModel}
                value={pickedModel ? pickedModel.label : PB.defaultSetLater}
              />
              <ReviewRow
                label={PB.reviewSkills}
                value={
                  keepAll
                    ? PB.fullDefaultBundle
                    : PB.keptCount.replace("{count}", String(keptSkills.size)) +
                      (hubSkills.length
                        ? PB.hubSuffix.replace("{count}", String(hubSkills.length))
                        : "")
                }
              />
              {!keepAll && hubSkills.length > 0 && (
                <p className="pl-24 text-xs text-muted-foreground">
                  {PB.hubPrefix.replace("{names}", hubSkills.map((s) => s.name).join(", "))}
                </p>
              )}
              {keepAll && hubSkills.length > 0 && (
                <ReviewRow
                  label={PB.reviewHubSkills}
                  value={hubSkills.map((s) => s.name).join(", ")}
                />
              )}
              <ReviewRow
                label={PB.reviewMcp}
                value={
                  mcpServers.length
                    ? mcpServers.map((s) => s.name).join(", ")
                    : PB.none
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
          onClick={() => setStep(STEPS[Math.max(0, stepIndex - 1)].id)}
        >
          {PB.back}
        </Button>
        {step === "review" ? (
          <Button onClick={handleCreate} disabled={creating || !nameValid}>
            {creating ? PB.creating : PB.createProfile}
          </Button>
        ) : (
          <Button
            disabled={!canAdvance}
            onClick={() =>
              setStep(STEPS[Math.min(STEPS.length - 1, stepIndex + 1)].id)
            }
          >
            {PB.next}
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
