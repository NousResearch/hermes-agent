import { api, fetchJSON, type SessionMessage } from "@/lib/api";
import { copyTextToClipboard } from "@/lib/clipboard";

/**
 * "Copy last response" for the dashboard chat, without ever touching the composer.
 *
 * The chat tab drives the Ink TUI through a PTY, so the previous implementation typed `/copy` plus
 * Return into the terminal. That is only safe on an EMPTY composer: with a draft in it the line is
 * not a bare slash command, so the TUI submitted `<draft>/copy` to the model and the user's unsent
 * text was delivered and cleared (#149751). Nothing in the browser can prove the composer is
 * empty, so the copy stops using the PTY altogether: read the transcript over HTTP, write it with
 * the Clipboard API, and report "copied" only once that write resolved.
 */

const ACTIVE_SESSION_PATH = "/api/chat/active-session";
/** How long the button shows its outcome before returning to "copy last response". */
export const COPY_STATE_RESET_MS = 1500;

export type CopyLastState = "idle" | "copied" | "failed";

/** The session a chat channel is on, or null when that chat has not run a turn yet. */
export async function fetchActiveChatSessionId(
  channel: string,
): Promise<string | null> {
  if (!channel) return null;
  try {
    const body = await fetchJSON<{ session_id?: string | null }>(
      `${ACTIVE_SESSION_PATH}?channel=${encodeURIComponent(channel)}`,
    );
    return typeof body.session_id === "string" ? body.session_id || null : null;
  } catch {
    // No marker yet, an older backend, or a transport failure: the caller
    // reports "failed" rather than guessing at a session.
    return null;
  }
}

/**
 * Last assistant text of a transcript page (the API returns the latest page in
 * chronological order). Assistant rows that carry only tool calls have no text,
 * so they are skipped — the same message `/copy` picked server-side.
 */
export function lastAssistantText(messages: SessionMessage[]): string | null {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = messages[i];
    const content = typeof message?.content === "string" ? message.content : "";
    if (message?.role === "assistant" && content.trim()) return content;
  }
  return null;
}

/**
 * Copy the chat's last assistant response. Returns the copied text, or null
 * when there was nothing to copy or the clipboard write failed.
 */
export async function copyLastResponse(opts: {
  channel: string;
  profile?: string;
  sessionId: string | null;
}): Promise<string | null> {
  const sessionId =
    opts.sessionId || (await fetchActiveChatSessionId(opts.channel));
  if (!sessionId) return null;
  let text: string | null = null;
  try {
    const transcript = await api.getSessionMessages(sessionId, opts.profile);
    text = lastAssistantText(transcript.messages ?? []);
  } catch {
    return null;
  }
  if (!text) return null;
  return (await copyTextToClipboard(text)) ? text : null;
}
