/** The row fields the activity line reads. ``last_activity_description`` is what
 *  the running turn was last doing (``sessions.last_activity_description``); the
 *  backend clears it when a turn ends. Absent on older backends. */
export interface SessionActivityFields {
  is_active: boolean;
  last_activity_description?: string | null;
}

/** Render an FTS5 snippet with highlighted matches.
 *  The backend wraps matches in >>> and <<< delimiters. */
export function SnippetHighlight({ snippet }: { snippet: string }) {
  const parts: React.ReactNode[] = [];
  const regex = />>>(.*?)<<</g;
  let last = 0;
  let match: RegExpExecArray | null;
  let i = 0;
  while ((match = regex.exec(snippet)) !== null) {
    if (match.index > last) {
      parts.push(snippet.slice(last, match.index));
    }
    parts.push(
      <mark key={i++} className="bg-warning/30 text-warning px-0.5">
        {match[1]}
      </mark>,
    );
    last = regex.lastIndex;
  }
  if (last < snippet.length) {
    parts.push(snippet.slice(last));
  }
  return (
    <p className="font-mondwest normal-case mt-0.5 min-w-0 max-w-full truncate text-xs text-text-secondary">
      {parts}
    </p>
  );
}

/** What a live session's turn is doing right now ("executing tool: terminal"),
 *  from the durable activity stamp. Only a live row gets one: the backend clears
 *  the label when a turn ends, and an idle row keeps its legacy layout. */
function sessionActivityText(session: SessionActivityFields): string | null {
  if (!session.is_active) return null;
  return session.last_activity_description?.replace(/\s+/g, " ").trim() || null;
}

export function SessionActivityLine({ session }: { session: SessionActivityFields }) {
  const activity = sessionActivityText(session);
  if (!activity) return null;
  return (
    <p
      className="min-w-0 truncate text-xs italic text-success/80"
      data-session-activity
      title={activity}
    >
      {activity}
    </p>
  );
}
