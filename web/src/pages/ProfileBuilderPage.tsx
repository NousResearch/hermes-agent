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
import { Info, Lightbulb } from "lucide-react";
import { api } from "@/lib/api";
import type { McpServerCreate, SkillInfo, SkillHubResult } from "@/lib/api";
import {
  PROFILE_EXAMPLES,
  PROFILE_FIELD_GUIDANCE,
} from "@/lib/profile-presets";
import { cn } from "@/lib/utils";

// Profile name rule mirrors the backend (`^[a-z0-9][a-z0-9_-]{0,63}$`).
const PROFILE_NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/;

type StepId = "identity" | "model" | "skills" | "mcp" | "review";

const STEPS: { id: StepId; label: string }[] = [
  { id: "identity", label: "Identity" },
  { id: "model", label: "Model" },
  { id: "skills", label: "Skills" },
  { id: "mcp", label: "MCPs" },
  { id: "review", label: "Review" },
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
            flat.push({ provider: prov.slug, model: m, label: `${prov.name} · ${m}` });
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
      showToast("MCP server needs a name", "error");
      return;
    }
    if (!mcpDraft.url.trim() && !mcpDraft.command.trim()) {
      showToast("Give the MCP server a URL or a command", "error");
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
        ? modelChoices?.find((c) => `${c.provider}\u0000${c.model}` === modelChoice)
        : undefined,
    [modelChoice, modelChoices],
  );

  const handleCreate = async () => {
    const n = name.trim();
    if (!PROFILE_NAME_RE.test(n)) {
      showToast("Invalid profile name (lowercase, digits, - and _)", "error");
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
        hub_skills: hubSkills.length ? hubSkills.map((s) => s.identifier) : undefined,
      });
      const pending = (res.hub_installs ?? []).filter((h) => h.pid).length;
      showToast(
        pending
          ? `Profile "${n}" created — ${pending} hub skill${pending === 1 ? "" : "s"} installing`
          : `Profile "${n}" created`,
        "success",
      );
      navigate("/profiles");
    } catch (e) {
      showToast(`Create failed: ${e}`, "error");
    } finally {
      setCreating(false);
    }
  };

  const stepIndex = STEPS.findIndex((s) => s.id === step);
  const canAdvance = step !== "identity" || nameValid;

  return (
    <div className="mx-auto w-full max-w-5xl space-y-6 p-4">
      <div className="flex items-center justify-between">
        <H2>New profile</H2>
        <Button ghost onClick={() => navigate("/profiles")}>
          Cancel
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
            {i + 1}. {s.label}
          </button>
        ))}
      </div>

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_20rem]">
      <Card>
        <CardContent className="space-y-4 p-5">
          {step === "identity" && (
            <div className="space-y-4">
              <div className="space-y-1.5">
                <FieldLabel
                  htmlFor="pb-name"
                  label="Profile name"
                  tooltip={PROFILE_FIELD_GUIDANCE.name.tooltip}
                />
                <Input
                  id="pb-name"
                  placeholder="coder"
                  value={name}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setName(e.target.value)}
                />
                <p className="text-xs text-muted-foreground">
                  {PROFILE_FIELD_GUIDANCE.name.hint} Try{" "}
                  <button
                    className="font-mono text-primary hover:underline"
                    type="button"
                    onClick={() => setName("coder")}
                  >
                    coder
                  </button>
                  ,{" "}
                  <button
                    className="font-mono text-primary hover:underline"
                    type="button"
                    onClick={() => setName("repo-researcher")}
                  >
                    repo-researcher
                  </button>
                  , or{" "}
                  <button
                    className="font-mono text-primary hover:underline"
                    type="button"
                    onClick={() => setName("support-triage")}
                  >
                    support-triage
                  </button>
                  .
                </p>
                {name && !nameValid && (
                  <p className="text-xs text-destructive">
                    Lowercase letters, digits, hyphens and underscores; must start with a letter or digit.
                  </p>
                )}
              </div>
              <div className="space-y-1.5">
                <FieldLabel
                  htmlFor="pb-desc"
                  label="Description (optional)"
                  tooltip={PROFILE_FIELD_GUIDANCE.description.tooltip}
                />
                <Input
                  id="pb-desc"
                  placeholder="Great at reading unfamiliar codebases, finding safe fixes, and explaining tradeoffs."
                  value={description}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                    setDescription(e.target.value)
                  }
                />
                <p className="text-xs text-muted-foreground">
                  {PROFILE_FIELD_GUIDANCE.description.hint}
                </p>
              </div>
            </div>
          )}

          {step === "model" && (
            <div className="space-y-3">
              <p className="text-sm text-muted-foreground">
                {PROFILE_FIELD_GUIDANCE.model.hint} Skip to use the default.
              </p>
              <Input
                placeholder="Filter models…"
                value={modelFilter}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                  setModelFilter(e.target.value)
                }
              />
              {modelChoices === null ? (
                <p className="text-sm text-muted-foreground">Loading models…</p>
              ) : (
                <div className="max-h-72 space-y-1 overflow-y-auto">
                  <button
                    onClick={() => setModelChoice("")}
                    className={cn(
                      "block w-full rounded px-3 py-2 text-left text-sm",
                      modelChoice === "" ? "bg-primary/10" : "hover:bg-muted",
                    )}
                  >
                    Use default (set later)
                  </button>
                  {filteredModels.map((c) => {
                    const key = `${c.provider}\u0000${c.model}`;
                    return (
                      <button
                        key={key}
                        onClick={() => setModelChoice(key)}
                        className={cn(
                          "block w-full rounded px-3 py-2 text-left text-sm",
                          modelChoice === key ? "bg-primary/10" : "hover:bg-muted",
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
              <p className="text-sm text-muted-foreground">
                {PROFILE_FIELD_GUIDANCE.skills.hint}
              </p>
              <label className="flex items-center gap-2 text-sm">
                <Checkbox
                  checked={keepAll}
                  onCheckedChange={(v) => setKeepAll(Boolean(v))}
                />
                Start from the full default skill bundle (recommended)
              </label>
              {!keepAll && (
                <div className="space-y-2">
                  <p className="text-xs text-muted-foreground">
                    Choose which built-in / optional skills to keep active. Unchecked skills are disabled in the new profile.
                  </p>
                  <Input
                    placeholder="Filter skills…"
                    value={skillFilter}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                      setSkillFilter(e.target.value)
                    }
                  />
                  {skills === null ? (
                    <p className="text-sm text-muted-foreground">Loading skills…</p>
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
                <Label>Add from the skills hub</Label>
                <div className="flex gap-2">
                  <Input
                    placeholder="Search the hub (e.g. linear, hyperliquid)…"
                    value={hubQuery}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                      setHubQuery(e.target.value)
                    }
                    onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => {
                      if (e.key === "Enter") runHubSearch();
                    }}
                  />
                  <Button outlined onClick={runHubSearch} disabled={hubSearching}>
                    {hubSearching ? "Searching…" : "Search"}
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
                          Add
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
                          aria-label={`Remove ${r.name}`}
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
              <p className="text-sm text-muted-foreground">
                {PROFILE_FIELD_GUIDANCE.mcp.hint} HTTP servers take a URL; stdio servers take a command + args.
              </p>
              <div className="grid grid-cols-2 gap-2">
                <Input
                  placeholder="Server name"
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
                  placeholder="Command (e.g. npx)"
                  value={mcpDraft.command}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                    setMcpDraft({ ...mcpDraft, command: e.target.value })
                  }
                />
                <Input
                  placeholder="Args (space-separated)"
                  value={mcpDraft.args}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                    setMcpDraft({ ...mcpDraft, args: e.target.value })
                  }
                />
              </div>
              <Button outlined onClick={addMcpDraft}>
                Add server
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
                        Remove
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {step === "review" && (
            <div className="space-y-3 text-sm">
              <ReviewRow label="Name" value={name.trim() || "—"} />
              <ReviewRow label="Description" value={description.trim() || "—"} />
              <ReviewRow
                label="Model"
                value={pickedModel ? pickedModel.label : "Default (set later)"}
              />
              <ReviewRow
                label="Skills"
                value={
                  keepAll
                    ? "Full default bundle"
                    : `${keptSkills.size} built-in/optional kept` +
                      (hubSkills.length ? ` + ${hubSkills.length} hub` : "")
                }
              />
              {!keepAll && hubSkills.length > 0 && (
                <p className="pl-24 text-xs text-muted-foreground">
                  Hub: {hubSkills.map((s) => s.name).join(", ")}
                </p>
              )}
              {keepAll && hubSkills.length > 0 && (
                <ReviewRow
                  label="Hub skills"
                  value={hubSkills.map((s) => s.name).join(", ")}
                />
              )}
              <ReviewRow
                label="MCP servers"
                value={mcpServers.length ? mcpServers.map((s) => s.name).join(", ") : "None"}
              />
            </div>
          )}
        </CardContent>
      </Card>

      <ProfileReferencePanel
        onUseExample={(example) => {
          setName(example.name);
          setDescription(example.description);
          setStep("identity");
        }}
      />
      </div>

      {/* Nav buttons */}
      <div className="flex items-center justify-between">
        <Button
          ghost
          disabled={stepIndex === 0}
          onClick={() => setStep(STEPS[Math.max(0, stepIndex - 1)].id)}
        >
          Back
        </Button>
        {step === "review" ? (
          <Button onClick={handleCreate} disabled={creating || !nameValid}>
            {creating ? "Creating…" : "Create profile"}
          </Button>
        ) : (
          <Button
            disabled={!canAdvance}
            onClick={() => setStep(STEPS[Math.min(STEPS.length - 1, stepIndex + 1)].id)}
          >
            Next
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

function FieldLabel({
  htmlFor,
  label,
  tooltip,
}: {
  htmlFor: string;
  label: string;
  tooltip: string;
}) {
  return (
    <div className="flex items-center gap-1.5">
      <Label htmlFor={htmlFor}>{label}</Label>
      <span
        aria-label={tooltip}
        className="inline-flex text-muted-foreground hover:text-foreground"
        role="img"
        title={tooltip}
      >
        <Info className="h-3.5 w-3.5" />
      </span>
    </div>
  );
}

function ProfileReferencePanel({
  onUseExample,
}: {
  onUseExample: (example: (typeof PROFILE_EXAMPLES)[number]) => void;
}) {
  return (
    <aside className="border border-border bg-card p-4 text-sm lg:sticky lg:top-4 lg:self-start">
      <div className="mb-3 flex items-center gap-2">
        <Lightbulb className="h-4 w-4 text-primary" />
        <h2 className="font-mondwest text-display text-sm tracking-wider">
          Great Profiles
        </h2>
      </div>
      <p className="mb-4 text-xs text-muted-foreground">
        Strong profiles are narrow, active, and easy to route work to. Describe the job, the preferred context, and the output bar.
      </p>
      <div className="space-y-3">
        {PROFILE_EXAMPLES.map((example) => (
          <div key={example.name} className="border-t border-border pt-3 first:border-t-0 first:pt-0">
            <div className="flex items-start justify-between gap-2">
              <div>
                <h3 className="text-sm font-medium">{example.title}</h3>
                <p className="text-xs text-muted-foreground">{example.bestFor}</p>
              </div>
              <Button
                ghost
                size="sm"
                onClick={() => onUseExample(example)}
              >
                Use
              </Button>
            </div>
            <p className="mt-2 text-xs text-muted-foreground">
              {example.description}
            </p>
          </div>
        ))}
      </div>
    </aside>
  );
}
