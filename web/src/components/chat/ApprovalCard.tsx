import { useEffect, useState } from "react";

export type ApprovalRequest = {
  request_id: string;
  command?: string;
  description?: string;
  choices?: string[];
  allow_session?: boolean;
  allow_permanent?: boolean;
  smart_denied?: boolean;
  expires_at?: number;
};

type ApprovalCardProps = {
  request: ApprovalRequest;
  onRespond: (choice: string) => Promise<void>;
};

const DEFAULT_CHOICES = ["once", "deny"];
const KNOWN_CHOICES = new Set(["once", "session", "always", "deny"]);

export function ApprovalCard({ request, onRespond }: ApprovalCardProps) {
  const [submitting, setSubmitting] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [nowMs, setNowMs] = useState(0);
  const hasExplicitChoices = Array.isArray(request.choices);
  const rawChoices = hasExplicitChoices ? request.choices : DEFAULT_CHOICES;
  const choices = (rawChoices ?? []).filter((choice) => KNOWN_CHOICES.has(choice) && (
    choice !== "session" || (request.allow_session === true && request.smart_denied !== true)
  ) && (
    choice !== "always" || (request.allow_permanent === true && request.smart_denied !== true)
  ));
  const safeChoices = choices.length > 0 ? choices : ["deny"];
  useEffect(() => {
    const update = () => setNowMs(Date.now());
    update();
    if (typeof request.expires_at !== "number" || !Number.isFinite(request.expires_at)) return undefined;
    const delay = Math.max(0, request.expires_at * 1000 - Date.now());
    const timeout = setTimeout(update, delay + 1);
    return () => clearTimeout(timeout);
  }, [request.expires_at]);
  const expired = typeof request.expires_at === "number"
    && Number.isFinite(request.expires_at)
    && nowMs > 0
    && request.expires_at * 1000 <= nowMs;

  const respond = async (choice: string) => {
    if (submitting || expired) return;
    setSubmitting(choice);
    setError(null);
    try {
      await onRespond(choice);
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSubmitting(null);
    }
  };

  return (
    <div role="dialog" aria-label={expired ? "Approval expired" : "Approval required"} aria-busy={submitting !== null} className="rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-sm">
      <strong>{expired ? "Approval expired" : "Approval required"}</strong>
      {(request.description || request.command) && (
        <div className="mt-1 whitespace-pre-wrap break-words">{request.description || request.command}</div>
      )}
      {error && <div role="alert" className="mt-2 text-destructive">{error}</div>}
      <div className="mt-2 flex flex-wrap gap-2" aria-label="Approval choices">
        {safeChoices.map((choice) => (
          <button key={choice} data-choice={choice} type="button" className="rounded border px-2 py-1" disabled={expired || submitting !== null} aria-label={`Approve ${choice}`} onClick={() => void respond(choice)}>
            {submitting === choice ? "Submitting…" : choice}
          </button>
        ))}
      </div>
      {expired && <div role="status" className="mt-2 text-muted-foreground">This approval is no longer actionable. Refresh to reconcile server state.</div>}
    </div>
  );
}
