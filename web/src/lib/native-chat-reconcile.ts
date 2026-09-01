export type ReconcileMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  streaming?: boolean;
};

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isPrefix(left: string, right: string): boolean {
  return left === right || left.startsWith(right) || right.startsWith(left);
}

function sameMessageIdentity(left: ReconcileMessage, right: ReconcileMessage): boolean {
  return left.role === right.role && left.id === right.id;
}

function equivalentAssistant(left: ReconcileMessage, right: ReconcileMessage): boolean {
  return left.role === "assistant" && right.role === "assistant" && isPrefix(left.text, right.text);
}

function mergeMessage(durable: ReconcileMessage, live: ReconcileMessage): ReconcileMessage {
  if (durable.role !== "assistant" || live.role !== "assistant") return durable;

  // A live stream may have advanced beyond the last persisted snapshot. Keep
  // the live id so subsequent deltas still target the same DOM/state entry.
  if (live.streaming && live.text.length >= durable.text.length && isPrefix(live.text, durable.text)) {
    return live;
  }

  // The snapshot may have completed while the browser was disconnected. Keep
  // the live id for future event routing, but adopt the durable completed text.
  if (durable.text.length > live.text.length && isPrefix(durable.text, live.text)) {
    return { ...live, text: durable.text, streaming: durable.streaming };
  }

  // Equal completed messages are already fully represented by the durable
  // snapshot; retaining the snapshot id also avoids duplicate history rows.
  if (durable.text === live.text) return durable;

  return live.streaming ? live : durable;
}

/**
 * Merge a durable session snapshot with events already rendered by the live
 * socket. Durable messages form the base; unmatched live messages are appended
 * and equivalent assistant messages are reconciled instead of duplicated.
 */
export function mergeSnapshotTranscript(
  snapshot: readonly ReconcileMessage[],
  current: readonly ReconcileMessage[],
): ReconcileMessage[] {
  const result = snapshot.map((message) => ({ ...message }));

  for (const live of current) {
    const identityIndex = result.findIndex((durable) => sameMessageIdentity(durable, live));
    const equivalentIndex = identityIndex >= 0
      ? identityIndex
      : result.findIndex((durable) => equivalentAssistant(durable, live));

    if (equivalentIndex >= 0) {
      result[equivalentIndex] = mergeMessage(result[equivalentIndex], live);
    } else {
      result.push({ ...live });
    }
  }

  return result;
}

/** Return true only when the backend explicitly sent the field. */
export function snapshotHasField(snapshot: unknown, field: string): boolean {
  return isObject(snapshot) && Object.prototype.hasOwnProperty.call(snapshot, field);
}

/** Snapshots without a session id are legacy-compatible and may be applied. */
export function snapshotMatchesSession(snapshot: unknown, activeSessionId: string | null): boolean {
  if (!activeSessionId || !isObject(snapshot)) return true;
  const snapshotSessionId = snapshot.session_id;
  return typeof snapshotSessionId !== "string" || snapshotSessionId === activeSessionId;
}
