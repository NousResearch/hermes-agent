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
import type { McpServerCreate, SkillInfo, SkillHubResult } from "@/lib/api";
import { useI18n } from "@/i18n";
import { en } from "@/i18n/en";
import { cn } from "@/lib/utils";

// Profile name rule mirrors the backend (`^[a-z0-9][a-z0-9_-]{0,63}$`).
const PROFILE_NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/;

type StepId = "identity" | "model" | "skills" | "mcp" | "review";

const STEP_IDS: StepId[] = ["identity", "model", "skills", "mcp", "review"];

function interpolate(
  template: string,
  values: Record<string, string | number>,
): string {
  return Object.entries(values).reduce(
    (result, [key, value]) => result.replaceAll(`{${key}}`, String(value)),
    template,
  );
}

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
  const copy = t.profileBuilderPage ?? en.profileBuilderPage!;
  const steps = [
    { id: "identity" as const, label: copy.identity },
    { id: "model" as const, label: copy.model },
    { id: "skills" as const, label: copy.skills },
    { id: "mcp" as const, label: copy.mcps },
    { id: "review" as const, label: copy.review },
  ];

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
  const [mcpDraft, setMcpDraft] = useState<{
    name: string;
    url: string;
    command: string;
    args: string;
  }>({ name: "", url: "", command: "", args: "" });

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
    const n = mcpDraft.name.trim();
    if (!n) {
      showToast(copy.mcpNameRequired, "error");
      return;
    }
    if (!mcpDraft.url.trim() && !mcpDraft.command.trim()) {
      showToast(copy.mcpEndpointRequired, "error");
      return;
    }
    const entry: McpServerCreate = { name: n };
    if (mcpDraft.url.trim()) entry.url = mcpDraft.url.trim();
    if (mcpDraft.command.trim()) {
      entry.command = mcpDraft.command.trim();
      const args = mcpDraft.args.trim();
      if (args) entry.args = args.split(/\s+/);
    }
    setMcpServers((prev) => [...prev.filter((s) => s.name !== n), entry]);
    setMcpDraft({ name: "", url: "", command: "", args: "" });
  };
  const removeMcp = (n: string) =>
    setMcpServers((prev) => prev.filter((s) => s.name !== n));

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
      showToast(copy.invalidName, "error");
      setStep("identity");
      return;
    }
    setCreating(true);
    try {
      const res = await api.createProfile({
        name: n,
        clone_from_default: false,
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
          ? interpolate(copy.createdWithInstalls, {
              name: n,
              count: pending,
            })
          : interpolate(copy.created, { name: n }),
        "success",
      );
      navigate("/profiles");
    } catch (e) {
      showToast(interpolate(copy.createFailed, { error: String(e) }), "error");
    } finally {
      setCreating(false);
    }
  };

  const stepIndex = STEP_IDS.indexOf(step);
  const canAdvance = step !== "identity" || nameValid;

  return (
    <div className="mx-auto w-full max-w-3xl space-y-6 p-4">
      <div className="flex items-center justify-between">
        <H2>{copy.newProfile}</H2>
        <Button ghost onClick={() => navigate("/profiles")}>
          {copy.cancel}
        </Button>
      </div>

      {/* Stepper */}
      <div className="flex items-center gap-2 text-sm">
        {steps.map((s, i) => (
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
            {i + 1}. {s.label}
          </button>
        ))}
      </div>

      <Card>
        <CardContent className="space-y-4 p-5">
          {step === "identity" && (
            <div className="space-y-4">
              <div className="space-y-1.5">
                <Label htmlFor="pb-name">{copy.profileName}</Label>
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
                    {copy.profileNameHint}
                  </p>
                )}
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="pb-desc">{copy.descriptionOptional}</Label>
                <Input
                  id="pb-desc"
                  placeholder={copy.descriptionPlaceholder}
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
              <p className="text-sm text-muted-foreground">{copy.modelIntro}</p>
              <Input
                placeholder={copy.filterModels}
                value={modelFilter}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                  setModelFilter(e.target.value)
                }
              />
              {modelChoices === null ? (
                <p className="text-sm text-muted-foreground">
                  {copy.loadingModels}
                </p>
              ) : (
                <div className="max-h-72 space-y-1 overflow-y-auto">
                  <button
                    onClick={() => setModelChoice("")}
                    className={cn(
                      "block w-full rounded px-3 py-2 text-start text-sm",
                      modelChoice === "" ? "bg-primary/10" : "hover:bg-muted",
                    )}
                  >
                    {copy.useDefault}
                  </button>
                  {filteredModels.map((c) => {
                    const key = `${c.provider}\u0000${c.model}`;
                    return (
                      <button
                        key={key}
                        onClick={() => setModelChoice(key)}
                        className={cn(
                          "block w-full rounded px-3 py-2 text-start text-sm",
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
                {copy.fullBundle}
              </label>
              {!keepAll && (
                <div className="space-y-2">
                  <p className="text-xs text-muted-foreground">
                    {copy.chooseSkills}
                  </p>
                  <Input
                    placeholder={copy.filterSkills}
                    value={skillFilter}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                      setSkillFilter(e.target.value)
                    }
                  />
                  {skills === null ? (
                    <p className="text-sm text-muted-foreground">
                      {copy.loadingSkills}
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
                              <Badge tone="secondary" className="ms-2">
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
                <Label>{copy.addFromHub}</Label>
                <div className="flex gap-2">
                  <Input
                    placeholder={copy.searchHubPlaceholder}
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
                    {hubSearching ? copy.searching : copy.search}
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
                          <Badge tone="secondary" className="ms-2">
                            {r.source}
                          </Badge>
                          {r.description && (
                            <span className="block text-xs text-muted-foreground">
                              {r.description}
                            </span>
                          )}
                        </span>
                        <Button size="sm" ghost onClick={() => addHubSkill(r)}>
                          {copy.add}
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
                          className="ms-1 text-xs"
                          onClick={() => removeHubSkill(r.identifier)}
                          aria-label={interpolate(copy.removeNamed, {
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
            <div className="space-y-4">
              <p className="text-sm text-muted-foreground">{copy.mcpIntro}</p>
              <div className="grid grid-cols-2 gap-2">
                <Input
                  placeholder={copy.serverName}
                  value={mcpDraft.name}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                    setMcpDraft({ ...mcpDraft, name: e.target.value })
                  }
                />
                <Input
                  placeholder="URL (https://…/mcp)"
                  value={mcpDraft.url}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                    setMcpDraft({ ...mcpDraft, url: e.target.value })
                  }
                />
                <Input
                  placeholder={copy.commandPlaceholder}
                  value={mcpDraft.command}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                    setMcpDraft({ ...mcpDraft, command: e.target.value })
                  }
                />
                <Input
                  placeholder={copy.argsPlaceholder}
                  value={mcpDraft.args}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                    setMcpDraft({ ...mcpDraft, args: e.target.value })
                  }
                />
              </div>
              <Button outlined onClick={addMcpDraft}>
                {copy.addServer}
              </Button>
              {mcpServers.length > 0 && (
                <div className="space-y-1">
                  {mcpServers.map((s) => (
                    <div
                      key={s.name}
                      className="flex items-center justify-between rounded bg-muted px-3 py-1.5 text-sm"
                    >
                      <span>
                        <span className="font-medium">{s.name}</span>{" "}
                        <span className="text-xs text-muted-foreground">
                          {s.url || `${s.command} ${(s.args || []).join(" ")}`}
                        </span>
                      </span>
                      <button
                        className="text-xs text-destructive"
                        onClick={() => removeMcp(s.name)}
                      >
                        {copy.remove}
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {step === "review" && (
            <div className="space-y-3 text-sm">
              <ReviewRow label={copy.name} value={name.trim() || "—"} />
              <ReviewRow
                label={copy.description}
                value={description.trim() || "—"}
              />
              <ReviewRow
                label={copy.model}
                value={pickedModel ? pickedModel.label : copy.defaultLater}
              />
              <ReviewRow
                label={copy.skills}
                value={
                  keepAll
                    ? copy.fullDefaultBundle
                    : interpolate(copy.skillsKept, {
                        count: keptSkills.size,
                      }) +
                      (hubSkills.length
                        ? interpolate(copy.hubAdded, {
                            count: hubSkills.length,
                          })
                        : "")
                }
              />
              {!keepAll && hubSkills.length > 0 && (
                <p className="ps-24 text-xs text-muted-foreground">
                  {copy.hub} {hubSkills.map((s) => s.name).join(", ")}
                </p>
              )}
              {keepAll && hubSkills.length > 0 && (
                <ReviewRow
                  label={copy.hubSkills}
                  value={hubSkills.map((s) => s.name).join(", ")}
                />
              )}
              <ReviewRow
                label={copy.mcpServers}
                value={
                  mcpServers.length
                    ? mcpServers.map((s) => s.name).join(", ")
                    : copy.none
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
          onClick={() => setStep(STEP_IDS[Math.max(0, stepIndex - 1)])}
        >
          {copy.back}
        </Button>
        {step === "review" ? (
          <Button onClick={handleCreate} disabled={creating || !nameValid}>
            {creating ? copy.creating : copy.createProfile}
          </Button>
        ) : (
          <Button
            disabled={!canAdvance}
            onClick={() =>
              setStep(STEP_IDS[Math.min(STEP_IDS.length - 1, stepIndex + 1)])
            }
          >
            {copy.next}
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
