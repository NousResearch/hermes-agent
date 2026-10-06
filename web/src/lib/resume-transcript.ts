import type { SessionMessage } from "@/lib/api";

/**
 * Snapshot of the REST-hydrated transcript for the session being resumed.
 * Keyed by (sessionId, profile) so a session or management-profile switch
 * never shows a stale transcript from a previous target.
 */
export interface ResumeTranscriptState {
  sessionId: string;
  profile: string;
  messages: SessionMessage[] | null;
  error: string | null;
}

/** Minimal messages loader — `api.getSessionMessages` satisfies this. */
export type ResumeMessagesLoader = (
  sessionId: string,
  profile: string,
) => Promise<{ messages: SessionMessage[] }>;

/**
 * Hydrate the transcript for a resumed session from
 * `GET /api/sessions/{id}/messages` (profile-scoped). Never throws: load
 * failures become an error state the UI renders instead of a blank area
 * (#60868).
 */
export async function loadResumeTranscript(
  loadMessages: ResumeMessagesLoader,
  sessionId: string,
  profile: string,
): Promise<ResumeTranscriptState> {
  try {
    const resp = await loadMessages(sessionId, profile);
    return { sessionId, profile, messages: resp.messages, error: null };
  } catch (err) {
    const message =
      err instanceof Error && err.message
        ? err.message
        : "failed to load messages";
    return { sessionId, profile, messages: null, error: message };
  }
}

/**
 * Return the transcript state only when it belongs to the requested
 * (sessionId, profile) target; otherwise null, which callers treat as
 * "still loading" — a stale transcript must never render for a new target.
 */
export function activeResumeTranscript(
  state: ResumeTranscriptState | null,
  sessionId: string | null,
  profile: string,
): ResumeTranscriptState | null {
  if (!state || !sessionId) return null;
  return state.sessionId === sessionId && state.profile === profile
    ? state
    : null;
}

export type ResumeTranscriptPhase =
  | "idle" // no resume target
  | "loading" // resume target set, transcript not (yet) arrived
  | "error"
  | "empty"
  | "ready";

/** Derive what the resume UI should render for the current target. */
export function resumeTranscriptPhase(
  state: ResumeTranscriptState | null,
  sessionId: string | null,
  profile: string,
): ResumeTranscriptPhase {
  const active = activeResumeTranscript(state, sessionId, profile);
  if (!sessionId) return "idle";
  if (!active) return "loading";
  if (active.error) return "error";
  if (active.messages && active.messages.length > 0) return "ready";
  return "empty";
}
