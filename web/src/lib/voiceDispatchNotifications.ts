export const TERMINAL_VOICE_DELEGATE_STATUSES = new Set(["completed", "failed", "cancelled", "stopped"]);

export type VoiceDelegateStatusLike = {
  delegate_id?: string;
  run_id: string;
  status: string;
  output?: string;
  error?: string;
  last_event?: string;
  events?: Array<Record<string, unknown>>;
};

export type VoiceDelegateStatusNotification = {
  key: string;
  kind: "run" | "approval";
  title: string;
  transcriptBody: string;
  voicePrompt: string;
};

const MAX_OUTPUT_CHARS = 900;

function truncateForVoice(value: string): string {
  const trimmed = value.trim();
  if (trimmed.length <= MAX_OUTPUT_CHARS) return trimmed;
  return `${trimmed.slice(0, MAX_OUTPUT_CHARS).trimEnd()}…`;
}

export function summarizeVoiceDelegateStatus(status: VoiceDelegateStatusLike): string {
  if (status.error) return `${status.status}: ${truncateForVoice(status.error)}`;
  if (status.output) return `${status.status}: ${truncateForVoice(status.output)}`;
  if (status.last_event) return `${status.status}: ${status.last_event}`;
  return status.status;
}

export function buildVoiceDelegateStatusNotification(
  status: VoiceDelegateStatusLike,
): VoiceDelegateStatusNotification | null {
  const delegateId = status.delegate_id ?? status.run_id;
  const summary = summarizeVoiceDelegateStatus(status);

  if (status.status === "waiting_for_approval") {
    return {
      key: `${delegateId}:waiting_for_approval`,
      kind: "approval",
      title: "Approval needed",
      transcriptBody: summary,
      voicePrompt:
        `Hermes delegate status update: approval is required. Summary: ${summary}. `
        + "Tell the user proactively in one concise spoken update. Explain that approval requires an explicit Hermes UI click. Available choices are approve once, approve for this session, or deny. Do not invent missing details.",
    };
  }

  if (!TERMINAL_VOICE_DELEGATE_STATUSES.has(status.status)) {
    return null;
  }

  return {
    key: `${delegateId}:${status.status}`,
    kind: "run",
    title: "Delegate finished",
    transcriptBody: summary,
    voicePrompt:
      `Hermes delegate finished with status ${status.status}. Summary: ${summary}. `
      + "Tell the user proactively now in one concise spoken update. Do not call another tool unless the user asks for follow-up. Do not mention internal delegate IDs unless necessary.",
  };
}
