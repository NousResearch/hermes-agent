import { useState } from "react";

export type ApprovalRequest = {
  request_id: string;
  command?: string;
  description?: string;
  choices?: string[];
  allow_permanent?: boolean;
};

type ApprovalCardProps = {
  request: ApprovalRequest;
  onRespond: (choice: string) => Promise<void>;
};

const DEFAULT_CHOICES = ["once", "session", "always", "deny"];

export function ApprovalCard({ request, onRespond }: ApprovalCardProps) {
  const [submitting, setSubmitting] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const choices = request.choices?.length ? request.choices : DEFAULT_CHOICES;

  const respond = async (choice: string) => {
    if (submitting) return;
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
    <div role="dialog" aria-label="Approval required" aria-busy={submitting !== null} className="rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-sm">
      <strong>Approval required</strong>
      {(request.description || request.command) && (
        <div className="mt-1 whitespace-pre-wrap break-words">{request.description || request.command}</div>
      )}
      {error && <div role="alert" className="mt-2 text-destructive">{error}</div>}
      <div className="mt-2 flex flex-wrap gap-2" aria-label="Approval choices">
        {choices.map((choice) => (
          <button key={choice} data-choice={choice} type="button" className="rounded border px-2 py-1" disabled={submitting !== null} aria-label={`Approve ${choice}`} onClick={() => void respond(choice)}>
            {submitting === choice ? "Submitting…" : choice}
          </button>
        ))}
      </div>
    </div>
  );
}
