import { useEffect, useState } from "react";

import { MessageList } from "@/components/SessionTranscript";
import { useI18n } from "@/i18n";
import { api } from "@/lib/api";
import {
  activeResumeTranscript,
  loadResumeTranscript,
  resumeTranscriptPhase,
  type ResumeMessagesLoader,
  type ResumeTranscriptState,
} from "@/lib/resume-transcript";
import { Spinner } from "@nous-research/ui/ui/components/spinner";

/**
 * Hydrate the transcript for the resumed session from
 * `GET /api/sessions/{id}/messages` (profile-scoped) while the PTY is
 * attaching. Re-runs when the resume target or management profile
 * changes; a cancelled load never lands (#60868).
 */
export function useResumeTranscript(
  sessionId: string | null,
  profile: string,
  loadMessages: ResumeMessagesLoader = (id, p) =>
    api.getSessionMessages(id, p),
): ResumeTranscriptState | null {
  const [transcript, setTranscript] = useState<ResumeTranscriptState | null>(
    null,
  );

  useEffect(() => {
    if (!sessionId) return;

    let cancelled = false;

    loadResumeTranscript(loadMessages, sessionId, profile).then((next) => {
      if (cancelled) return;
      setTranscript(next);
    });

    return () => {
      cancelled = true;
    };
  }, [sessionId, profile, loadMessages]);

  return transcript;
}

/**
 * REST-hydrated transcript for a resumed session, rendered above the live
 * PTY so previous messages are visible immediately instead of leaving the
 * conversation area blank until new PTY output arrives (#60868).
 *
 * Keyed by (sessionId, profile): a stale transcript from a previous
 * session or management profile never renders for the current target.
 */
export function ResumeTranscriptPanel({
  transcript,
  sessionId,
  profile,
}: {
  transcript: ResumeTranscriptState | null;
  sessionId: string | null;
  profile: string;
}) {
  const { t } = useI18n();
  const phase = resumeTranscriptPhase(transcript, sessionId, profile);
  const active = activeResumeTranscript(transcript, sessionId, profile);

  if (!sessionId || phase === "idle") return null;

  return (
    <div className="mb-2 flex max-h-[min(36vh,24rem)] shrink-0 flex-col overflow-hidden rounded border border-white/10 bg-black/35 text-white/85">
      <div className="flex min-h-8 items-center justify-between gap-2 border-b border-white/10 px-3 py-1.5">
        <span className="text-display text-[0.6875rem] tracking-wider text-white/60">
          {t.sessions.history}
        </span>
        {phase === "loading" ? (
          <Spinner className="text-sm text-white/60" />
        ) : active?.messages ? (
          <span className="font-mono-ui text-[0.6875rem] text-white/45">
            {active.messages.length} {t.common.msgs}
          </span>
        ) : null}
      </div>
      <div className="min-h-0 overflow-y-auto p-2">
        {phase === "loading" && (
          <div className="flex items-center justify-center py-6 text-xs text-white/60">
            <Spinner className="text-sm" />
          </div>
        )}
        {phase === "error" && active?.error && (
          <p className="py-4 text-center text-xs text-destructive">
            {active.error}
          </p>
        )}
        {phase === "empty" && (
          <p className="py-4 text-center text-xs text-white/50">
            {t.sessions.noMessages}
          </p>
        )}
        {phase === "ready" && active?.messages && (
          <MessageList
            messages={active.messages}
            className="flex flex-col gap-2 pr-1"
          />
        )}
      </div>
    </div>
  );
}
