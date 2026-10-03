import { useMemo, useState } from "react";
import { useNavigate } from "react-router";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { Checkbox } from "@nous-research/ui/ui/components/checkbox";
import { Input } from "@nous-research/ui/ui/components/input";
import { Label } from "@nous-research/ui/ui/components/label";
import { H2 } from "@nous-research/ui/ui/components/typography/h2";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { ArrowLeft, ArrowRight, Check, Clipboard, Sparkles } from "lucide-react";
import {
  buildNeonGeniePrompt,
  NEON_GENIE_MISSIONS,
  type NeonGenieBrief,
  type NeonGenieMissionId,
} from "@/lib/neon-genie-wizard";
import { cn } from "@/lib/utils";

type StepId = "mission" | "context" | "guardrails" | "review";
const STEPS: { id: StepId; label: string }[] = [
  { id: "mission", label: "Mission" },
  { id: "context", label: "Context" },
  { id: "guardrails", label: "Guardrails" },
  { id: "review", label: "Review" },
];

const textareaClass =
  "min-h-24 w-full resize-y rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm outline-none placeholder:text-muted-foreground focus-visible:ring-2 focus-visible:ring-ring";

export default function NeonGenieWizardPage() {
  const navigate = useNavigate();
  const { toast, showToast } = useToast();
  const [step, setStep] = useState<StepId>("mission");
  const [brief, setBrief] = useState<NeonGenieBrief>({
    mission: "product-audit",
    requestedOutcome: "",
    targetUser: "",
    currentState: "",
    desiredState: "",
    evidence: "",
    constraints: "",
    researchEnabled: true,
    maxFetches: "",
    allowDrafting: true,
    requireHumanReview: true,
  });

  const stepIndex = STEPS.findIndex((candidate) => candidate.id === step);
  const mission =
    NEON_GENIE_MISSIONS.find((candidate) => candidate.id === brief.mission) ??
    NEON_GENIE_MISSIONS[0];
  const prompt = useMemo(() => buildNeonGeniePrompt(brief), [brief]);
  const contextValid = brief.requestedOutcome.trim() && brief.currentState.trim();

  const patchBrief = <K extends keyof NeonGenieBrief>(key: K, value: NeonGenieBrief[K]) =>
    setBrief((current) => ({ ...current, [key]: value }));

  const copyPrompt = async (): Promise<boolean> => {
    try {
      await navigator.clipboard.writeText(prompt);
      showToast("Neon Genie prompt copied", "success");
      return true;
    } catch {
      showToast("Could not copy the prompt. Select it manually below.", "error");
      return false;
    }
  };

  const openChat = async () => {
    if (await copyPrompt()) navigate("/chat");
  };

  const canContinue = step !== "context" || Boolean(contextValid);

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-4 py-6 sm:px-6">
      <header className="overflow-hidden rounded-2xl border bg-gradient-to-br from-fuchsia-500/10 via-background to-cyan-500/10 p-6 sm:p-8">
        <div className="flex items-start gap-4">
          <div className="rounded-xl bg-fuchsia-500/15 p-3 text-fuchsia-500">
            <Sparkles className="size-7" />
          </div>
          <div className="space-y-2">
            <Badge tone="secondary">Evidence-bound advisory</Badge>
            <H2>Neon Genie</H2>
            <p className="max-w-3xl text-sm text-muted-foreground sm:text-base">
              Turn a product question or weak signal into a governed Hermes brief. This wizard
              creates a complete prompt with claim labels, DataRequests, fail-closed gates, and
              advisory-only authority.
            </p>
          </div>
        </div>
      </header>

      <nav aria-label="Wizard progress" className="grid grid-cols-4 gap-2">
        {STEPS.map((candidate, index) => (
          <button
            key={candidate.id}
            type="button"
            onClick={() => index <= stepIndex && setStep(candidate.id)}
            className={cn(
              "rounded-lg border px-2 py-3 text-left text-xs transition-colors sm:px-4 sm:text-sm",
              candidate.id === step
                ? "border-fuchsia-500 bg-fuchsia-500/10 text-foreground"
                : index < stepIndex
                  ? "border-emerald-500/40 bg-emerald-500/5 text-foreground"
                  : "text-muted-foreground",
            )}
          >
            <span className="mb-1 flex items-center gap-2 font-medium">
              {index < stepIndex ? <Check className="size-3.5" /> : `${index + 1}.`}
              {candidate.label}
            </span>
          </button>
        ))}
      </nav>

      <Card>
        <CardContent className="p-5 sm:p-7">
          {step === "mission" && (
            <section className="space-y-5">
              <div>
                <h3 className="text-lg font-semibold">What should Neon Genie do?</h3>
                <p className="text-sm text-muted-foreground">
                  Pick the closest mission. The wizard automatically selects the smallest useful
                  profile set.
                </p>
              </div>
              <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                {NEON_GENIE_MISSIONS.map((candidate) => (
                  <button
                    key={candidate.id}
                    type="button"
                    onClick={() => patchBrief("mission", candidate.id as NeonGenieMissionId)}
                    className={cn(
                      "rounded-xl border p-4 text-left transition-all hover:border-fuchsia-500/60 hover:bg-muted/40",
                      brief.mission === candidate.id &&
                        "border-fuchsia-500 bg-fuchsia-500/10 ring-1 ring-fuchsia-500/30",
                    )}
                  >
                    <div className="font-medium">{candidate.label}</div>
                    <p className="mt-1 text-sm text-muted-foreground">{candidate.description}</p>
                    <div className="mt-3 flex flex-wrap gap-1">
                      {candidate.profiles.map((profile) => (
                        <Badge key={profile} tone="outline" className="text-[10px]">
                          {profile}
                        </Badge>
                      ))}
                    </div>
                  </button>
                ))}
              </div>
            </section>
          )}

          {step === "context" && (
            <section className="space-y-5">
              <div>
                <h3 className="text-lg font-semibold">Describe the transition</h3>
                <p className="text-sm text-muted-foreground">
                  Outcome and current state are required. Unknown private facts can remain unknown.
                </p>
              </div>
              <div className="grid gap-5 md:grid-cols-2">
                <Field label="Requested outcome" required>
                  <textarea
                    className={textareaClass}
                    value={brief.requestedOutcome}
                    onChange={(event) => patchBrief("requestedOutcome", event.target.value)}
                    placeholder="What decision or packet should this run produce?"
                  />
                </Field>
                <Field label="Target user or beneficiary">
                  <textarea
                    className={textareaClass}
                    value={brief.targetUser}
                    onChange={(event) => patchBrief("targetUser", event.target.value)}
                    placeholder="Who is stuck, affected, buying, or benefiting?"
                  />
                </Field>
                <Field label="Current state" required>
                  <textarea
                    className={textareaClass}
                    value={brief.currentState}
                    onChange={(event) => patchBrief("currentState", event.target.value)}
                    placeholder="What exists today? What is blocked or fragmented?"
                  />
                </Field>
                <Field label="Desired state">
                  <textarea
                    className={textareaClass}
                    value={brief.desiredState}
                    onChange={(event) => patchBrief("desiredState", event.target.value)}
                    placeholder="What should be observably different when done?"
                  />
                </Field>
                <Field label="Known evidence and sources">
                  <textarea
                    className={textareaClass}
                    value={brief.evidence}
                    onChange={(event) => patchBrief("evidence", event.target.value)}
                    placeholder="Links, documents, metrics, interviews, or facts already known."
                  />
                </Field>
                <Field label="Constraints and exclusions">
                  <textarea
                    className={textareaClass}
                    value={brief.constraints}
                    onChange={(event) => patchBrief("constraints", event.target.value)}
                    placeholder="Time, capital, access, privacy, scope, and things Hermes must not do."
                  />
                </Field>
              </div>
            </section>
          )}

          {step === "guardrails" && (
            <section className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold">Set evidence and authority boundaries</h3>
                <p className="text-sm text-muted-foreground">
                  Execution, spending, publishing, outreach, and repository mutation always remain
                  disabled.
                </p>
              </div>
              <div className="space-y-4 rounded-xl border p-5">
                <Toggle
                  checked={brief.researchEnabled}
                  onCheckedChange={(checked) => patchBrief("researchEnabled", checked)}
                  label="Allow public research"
                  description="Hermes may search and fetch public sources for decision-critical facts."
                />
                {brief.researchEnabled && (
                  <div className="max-w-xs space-y-2 pl-7">
                    <Label htmlFor="max-fetches">Optional research fetch limit</Label>
                    <Input
                      id="max-fetches"
                      type="number"
                      min="1"
                      value={brief.maxFetches}
                      onChange={(event) => patchBrief("maxFetches", event.target.value)}
                      placeholder="Host default"
                    />
                  </div>
                )}
                <Toggle
                  checked={brief.allowDrafting}
                  onCheckedChange={(checked) => patchBrief("allowDrafting", checked)}
                  label="Allow drafting"
                  description="Hermes may produce advisory packets, models, and handoff stubs."
                />
                <Toggle
                  checked={brief.requireHumanReview}
                  onCheckedChange={(checked) => patchBrief("requireHumanReview", checked)}
                  label="Require human review"
                  description="Outputs remain drafts until a person explicitly approves them."
                />
              </div>
              <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-4 text-sm">
                <strong>Fail-closed defaults:</strong> public gaps are researched when allowed,
                private gaps become DataRequests, and unsupported conclusions become NOT_COMPUTABLE.
              </div>
            </section>
          )}

          {step === "review" && (
            <section className="space-y-5">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <h3 className="text-lg font-semibold">Review the Hermes prompt</h3>
                  <p className="text-sm text-muted-foreground">
                    Copy it, open Chat, paste, and submit. Neon Genie remains advisory only.
                  </p>
                </div>
                <Badge tone="secondary">{mission.label}</Badge>
              </div>
              <textarea
                aria-label="Generated Neon Genie prompt"
                className={cn(textareaClass, "min-h-[28rem] font-mono text-xs leading-relaxed")}
                readOnly
                value={prompt}
              />
              <div className="flex flex-wrap gap-3">
                <Button
                  type="button"
                  className="border bg-background text-foreground hover:bg-muted"
                  onClick={() => void copyPrompt()}
                >
                  <Clipboard className="mr-2 size-4" /> Copy prompt
                </Button>
                <Button type="button" onClick={openChat}>
                  <Sparkles className="mr-2 size-4" /> Copy and open Chat
                </Button>
              </div>
            </section>
          )}
        </CardContent>
      </Card>

      <footer className="flex items-center justify-between gap-3">
        <Button
          type="button"
          className="border bg-background text-foreground hover:bg-muted"
          disabled={stepIndex === 0}
          onClick={() => setStep(STEPS[stepIndex - 1].id)}
        >
          <ArrowLeft className="mr-2 size-4" /> Back
        </Button>
        {stepIndex < STEPS.length - 1 && (
          <Button
            type="button"
            disabled={!canContinue}
            onClick={() => setStep(STEPS[stepIndex + 1].id)}
          >
            Continue <ArrowRight className="ml-2 size-4" />
          </Button>
        )}
      </footer>
      <Toast toast={toast} />
    </div>
  );
}

function Field({
  label,
  required = false,
  children,
}: {
  label: string;
  required?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-2">
      <Label>
        {label} {required && <span className="text-destructive">*</span>}
      </Label>
      {children}
    </div>
  );
}

function Toggle({
  checked,
  onCheckedChange,
  label,
  description,
}: {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  label: string;
  description: string;
}) {
  return (
    <label className="flex cursor-pointer items-start gap-3">
      <Checkbox checked={checked} onCheckedChange={(value) => onCheckedChange(value === true)} />
      <span>
        <span className="block text-sm font-medium">{label}</span>
        <span className="block text-sm text-muted-foreground">{description}</span>
      </span>
    </label>
  );
}
