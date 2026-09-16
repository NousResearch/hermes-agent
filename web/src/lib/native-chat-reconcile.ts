export type ReconcileMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  rowId?: number;
  messageId?: string;
  turnId?: string;
  streaming?: boolean;
  error?: string;
  interim?: boolean;
};

export type SnapshotCoverage = "prefix" | "tail";

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isPrefix(left: string, right: string): boolean {
  return left === right || left.startsWith(right) || right.startsWith(left);
}

function sameMessageIdentity(left: ReconcileMessage, right: ReconcileMessage): boolean {
  if (left.role !== right.role) return false;
  if (left.messageId !== undefined || right.messageId !== undefined) {
    if (left.messageId !== undefined && right.messageId !== undefined) return left.messageId === right.messageId;
    return left.id === right.id;
  }
  if (left.id === right.id) return true;
  return left.role === "assistant"
    && left.turnId !== undefined
    && right.turnId !== undefined
    && left.turnId === right.turnId;
}

function equivalentAssistant(left: ReconcileMessage, right: ReconcileMessage): boolean {
  return left.role === "assistant"
    && right.role === "assistant"
    && !(left.messageId !== undefined && right.messageId !== undefined && left.messageId !== right.messageId)
    && left.text.length > 0
    && right.text.length > 0
    && !left.interim
    && !right.interim
    && isPrefix(left.text, right.text);
}

function equivalentUser(left: ReconcileMessage, right: ReconcileMessage): boolean {
  return left.role === "user" && right.role === "user" && left.text === right.text;
}

function preserveLiveMetadata(merged: ReconcileMessage, live: ReconcileMessage, preferMergedError = false): ReconcileMessage {
  const next: ReconcileMessage = {
    ...merged,
    ...(merged.messageId === undefined && live.messageId !== undefined ? { messageId: live.messageId } : {}),
    ...(merged.turnId === undefined && live.turnId !== undefined ? { turnId: live.turnId } : {}),
    ...(live.interim !== undefined ? { interim: live.interim } : {}),
  };
  if (live.error !== undefined && !preferMergedError) next.error = live.error;
  else if (preferMergedError && merged.error !== undefined) next.error = merged.error;
  else if (live.streaming === true || next.error === undefined || preferMergedError) delete next.error;
  return next;
}

function mergeMessage(durable: ReconcileMessage, live: ReconcileMessage, preferMergedError = false): ReconcileMessage {
  if (durable.role !== "assistant" || live.role !== "assistant") return preserveLiveMetadata(durable, live, preferMergedError);

  const preferDurableError = (base: ReconcileMessage): ReconcileMessage => {
    const next = { ...base };
    if (preferMergedError) {
      if (durable.error === undefined) delete next.error;
      else next.error = durable.error;
    }
    return preserveLiveMetadata(next, live, preferMergedError);
  };

  // A reconnect snapshot can lag behind a locally completed live answer. Do
  // not truncate the newer answer merely because the live row is no longer
  // marked streaming.
  if (live.text.length > durable.text.length && isPrefix(live.text, durable.text)) {
    return preferDurableError(live);
  }

  // A live stream may have advanced beyond the last persisted snapshot. Keep
  // the live id so subsequent deltas still target the same DOM/state entry.
  if (live.streaming && live.text.length >= durable.text.length && isPrefix(live.text, durable.text)) {
    return preferDurableError(live);
  }

  // The snapshot may have completed while the browser was disconnected. Keep
  // the live id for future event routing, but adopt the durable completed text.
  if (durable.text.length > live.text.length && isPrefix(durable.text, live.text)) {
    return preferDurableError({ ...live, text: durable.text, streaming: durable.streaming });
  }

  // Equal completed messages are already fully represented by the durable
  // snapshot; retaining the snapshot id also avoids duplicate history rows.
  if (durable.text === live.text) return preferDurableError(durable);

  return preferDurableError(live.streaming ? live : durable);
}

function lastUserIndex(messages: readonly ReconcileMessage[], before = messages.length): number {
  for (let index = Math.min(before, messages.length) - 1; index >= 0; index -= 1) {
    if (messages[index].role === "user") return index;
  }
  return -1;
}

function findCurrentTurnAssistant(
  messages: readonly ReconcileMessage[],
  live: ReconcileMessage,
  limit = messages.length,
  activeUserText?: string | null,
): number {
  if (live.role !== "assistant") return -1;
  const userIndex = activeUserText == null
    ? lastUserIndex(messages)
    : (() => {
      for (let index = messages.length - 1; index >= 0; index -= 1) {
        if (messages[index].role === "user" && messages[index].text === activeUserText) return index;
      }
      return -1;
    })();
  if (userIndex < 0) return -1;
  for (let index = userIndex + 1; index < Math.min(limit, messages.length); index += 1) {
    if (messages[index].role === "user") return -1;
    if (equivalentAssistant(messages[index], live)) return index;
  }
  return -1;
}

export function chooseCompletedText(existing: string, incoming: string): string {
  return existing.length > incoming.length && isPrefix(existing, incoming)
    ? existing
    : incoming;
}

function snapshotMessagesEquivalent(
  durable: ReconcileMessage,
  current: ReconcileMessage,
): boolean {
  if (sameMessageIdentity(durable, current)) return true;
  if (durable.role !== current.role) return false;
  if (durable.role === "user") {
    return current.rowId === undefined && equivalentUser(durable, current);
  }
  return equivalentAssistant(durable, current);
}

function findTailAlignment(
  snapshot: readonly ReconcileMessage[],
  current: readonly ReconcileMessage[],
): number[] | null {
  if (snapshot.length === 0) return [];
  for (let start = current.length - snapshot.length; start >= 0; start -= 1) {
    const indices = snapshot.map((_, offset) => start + offset);
    if (indices.every((index, offset) => {
      if (!snapshotMessagesEquivalent(snapshot[offset], current[index])) return false;
      if (sameMessageIdentity(snapshot[offset], current[index]) || snapshot[offset].role !== "assistant") return true;
      const previousSnapshotUser = lastUserIndex(snapshot, offset);
      const previousCurrentUser = lastUserIndex(current, index);
      return previousSnapshotUser >= 0
        && previousCurrentUser >= 0
        && snapshotMessagesEquivalent(snapshot[previousSnapshotUser], current[previousCurrentUser]);
    })) {
      return indices;
    }
  }
  return null;
}

function mergeTailSnapshotTranscript(
  snapshot: readonly ReconcileMessage[],
  current: readonly ReconcileMessage[],
): ReconcileMessage[] {
  const alignment = findTailAlignment(snapshot, current);
  if (alignment !== null) {
    const result = current.map((message) => ({ ...message }));
    alignment.forEach((currentIndex, snapshotIndex) => {
      result[currentIndex] = mergeMessage(snapshot[snapshotIndex], result[currentIndex], true);
    });
    return result;
  }

  // If the snapshot is not a contiguous window (for example, a live row is
  // still missing locally), match from the end because this is a tail view.
  const matchedCurrentIndices = new Set<number>();
  const matchedSnapshotIndices = new Set<number>();
  const snapshotToCurrent = new Map<number, number>();
  let currentCursor = current.length - 1;

  for (let snapshotIndex = snapshot.length - 1; snapshotIndex >= 0; snapshotIndex -= 1) {
    if (matchedSnapshotIndices.has(snapshotIndex)) continue;
    for (let currentIndex = currentCursor; currentIndex >= 0; currentIndex -= 1) {
      if (matchedCurrentIndices.has(currentIndex)) continue;
      if (!snapshotMessagesEquivalent(snapshot[snapshotIndex], current[currentIndex])) continue;
      if (snapshot[snapshotIndex].role === "assistant"
        && !sameMessageIdentity(snapshot[snapshotIndex], current[currentIndex])
        && lastUserIndex(snapshot, snapshotIndex) < 0) continue;

      const previousSnapshotIndex = snapshotIndex - 1;
      const previousCurrentIndex = currentIndex - 1;
      const hasPreviousUser = snapshot[previousSnapshotIndex]?.role === "user";
      if (hasPreviousUser && current[previousCurrentIndex]?.role === "user") {
        if (previousCurrentIndex < 0 || matchedCurrentIndices.has(previousCurrentIndex)) continue;
        if (!snapshotMessagesEquivalent(snapshot[previousSnapshotIndex], current[previousCurrentIndex])) continue;
        matchedSnapshotIndices.add(previousSnapshotIndex);
        matchedCurrentIndices.add(previousCurrentIndex);
        snapshotToCurrent.set(previousSnapshotIndex, previousCurrentIndex);
        currentCursor = previousCurrentIndex - 1;
      }

      matchedSnapshotIndices.add(snapshotIndex);
      matchedCurrentIndices.add(currentIndex);
      snapshotToCurrent.set(snapshotIndex, currentIndex);
      currentCursor = Math.min(currentCursor, currentIndex - 1);
      break;
    }
  }

  if (snapshotToCurrent.size === 0) {
    return [
      ...current.map((message) => ({ ...message })),
      ...snapshot.map((message) => ({ ...message })),
    ];
  }

  const orderedMatches = [...snapshotToCurrent.entries()].sort(([, left], [, right]) => left - right);
  const merged: ReconcileMessage[] = [];
  let currentIndex = 0;
  let snapshotIndex = 0;
  let previousMatch: [number, number] | null = null;

  const appendSnapshotRange = (until: number) => {
    while (snapshotIndex < until) {
      if (!matchedSnapshotIndices.has(snapshotIndex)) merged.push({ ...snapshot[snapshotIndex] });
      snapshotIndex += 1;
    }
  };

  for (const [matchedSnapshotIndex, matchedCurrentIndex] of orderedMatches) {
    if (previousMatch === null) {
      while (currentIndex < matchedCurrentIndex) {
        merged.push({ ...current[currentIndex] });
        currentIndex += 1;
      }
      appendSnapshotRange(matchedSnapshotIndex);
    } else {
      appendSnapshotRange(matchedSnapshotIndex);
      while (currentIndex < matchedCurrentIndex) {
        merged.push({ ...current[currentIndex] });
        currentIndex += 1;
      }
    }
    merged.push(mergeMessage(snapshot[matchedSnapshotIndex], current[matchedCurrentIndex], true));
    currentIndex = matchedCurrentIndex + 1;
    snapshotIndex = matchedSnapshotIndex + 1;
    previousMatch = [matchedSnapshotIndex, matchedCurrentIndex];
  }

  appendSnapshotRange(snapshot.length);
  while (currentIndex < current.length) {
    merged.push({ ...current[currentIndex] });
    currentIndex += 1;
  }
  return merged;
}

/**
 * Merge a durable session snapshot with events already rendered by the live
 * socket. Durable messages form the base; unmatched live messages are appended
 * and equivalent assistant messages are reconciled instead of duplicated.
 * Synthetic user messages are matched by ordered text occurrence because their
 * local IDs are not the durable row IDs returned by a later snapshot.
 */
export function mergeSnapshotTranscript(
  snapshot: readonly ReconcileMessage[],
  current: readonly ReconcileMessage[],
  coverage: SnapshotCoverage = "prefix",
): ReconcileMessage[] {
  if (coverage === "tail") {
    return mergeTailSnapshotTranscript(snapshot, current);
  }
  const matches: Array<[currentIndex: number, snapshotIndex: number]> = [];
  let snapshotCursor = 0;
  for (let currentIndex = 0; currentIndex < current.length; currentIndex += 1) {
    const live = current[currentIndex];
    for (let snapshotIndex = snapshotCursor; snapshotIndex < snapshot.length; snapshotIndex += 1) {
      const durable = snapshot[snapshotIndex];
      const identityMatch = sameMessageIdentity(durable, live);
      const equivalentMatch = live.role === "assistant"
        ? equivalentAssistant(durable, live)
        : live.role === "user" && live.rowId === undefined && equivalentUser(durable, live);
      if (!identityMatch && !equivalentMatch) continue;
      if (live.role === "assistant" && !identityMatch) {
        const previousSnapshotUser = lastUserIndex(snapshot, snapshotIndex);
        const previousCurrentUser = lastUserIndex(current, currentIndex);
        if (previousSnapshotUser < 0 || previousCurrentUser < 0
          || !snapshotMessagesEquivalent(snapshot[previousSnapshotUser], current[previousCurrentUser])) continue;
      }
      matches.push([currentIndex, snapshotIndex]);
      snapshotCursor = snapshotIndex + 1;
      break;
    }
  }

  const merged: ReconcileMessage[] = [];
  let currentCursor = 0;
  let snapshotCursorForMerge = 0;
  for (const [currentIndex, snapshotIndex] of matches) {
    while (snapshotCursorForMerge < snapshotIndex) {
      merged.push({ ...snapshot[snapshotCursorForMerge] });
      snapshotCursorForMerge += 1;
    }
    while (currentCursor < currentIndex) {
      merged.push({ ...current[currentCursor] });
      currentCursor += 1;
    }
    merged.push(mergeMessage(snapshot[snapshotIndex], current[currentIndex], true));
    snapshotCursorForMerge = snapshotIndex + 1;
    currentCursor = currentIndex + 1;
  }
  while (snapshotCursorForMerge < snapshot.length) {
    merged.push({ ...snapshot[snapshotCursorForMerge] });
    snapshotCursorForMerge += 1;
  }
  while (currentCursor < current.length) {
    merged.push({ ...current[currentCursor] });
    currentCursor += 1;
  }
  return merged;
}

export function mergeInflightTranscript(
  current: readonly ReconcileMessage[],
  inflight: readonly ReconcileMessage[],
  activeUserText?: string | null,
): ReconcileMessage[] {
  const result = current.map((message) => ({ ...message }));
  const originalUserIndex = inflight.findIndex((message) => message.role === "user");
  const originalAssistant = originalUserIndex >= 0
    ? inflight.slice(originalUserIndex + 1).find((message) => message.role === "assistant")
    : undefined;
  const originalAssistantMessageId = originalAssistant?.messageId;
  const assistantAfterUser = (messages: readonly ReconcileMessage[], userIndex: number): ReconcileMessage | undefined => {
    for (let index = userIndex + 1; index < messages.length; index += 1) {
      if (messages[index].role === "user") break;
      if (messages[index].role === "assistant") return messages[index];
    }
    return undefined;
  };
  let lastUserIndex = lastUserIndexInResult(result);
  for (const incoming of inflight) {
    let matchIndex = result.findIndex((message) => sameMessageIdentity(message, incoming));
    if (matchIndex < 0 && incoming.role === "user" && incoming === inflight[originalUserIndex] && originalAssistantMessageId) {
      const durableAssistantIndex = result.findIndex((message) => message.role === "assistant" && message.messageId === originalAssistantMessageId);
      const durableUserIndex = durableAssistantIndex >= 0 ? lastUserIndexInResult(result, durableAssistantIndex) : -1;
      if (durableUserIndex >= 0 && result[durableUserIndex]?.text === incoming.text) matchIndex = durableUserIndex;
    }
    if (matchIndex < 0 && incoming.role === "user" && incoming === inflight[originalUserIndex]
      && (activeUserText !== null && activeUserText !== undefined ? incoming.text === activeUserText : true)
      && lastUserIndex >= 0
      && result[lastUserIndex]?.text === incoming.text
      && (() => {
        const currentAssistant = assistantAfterUser(result, lastUserIndex);
        if (currentAssistant) {
          const conflictingTurnId = currentAssistant.turnId !== undefined && originalAssistant?.turnId !== undefined && currentAssistant.turnId !== originalAssistant.turnId;
          return !conflictingTurnId && Boolean(originalAssistant && (sameMessageIdentity(currentAssistant, originalAssistant) || equivalentAssistant(currentAssistant, originalAssistant)));
        }
        return activeUserText !== null && activeUserText !== undefined && Boolean(originalAssistant);
      })()) {
      matchIndex = lastUserIndex;
    }
    if (matchIndex < 0 && incoming.role === "assistant" && incoming.text && lastUserIndex >= 0) {
      for (let index = lastUserIndex + 1; index < result.length; index += 1) {
        const message = result[index];
        if (message.role === "user") break;
        if (equivalentAssistant(message, incoming)) {
          matchIndex = index;
          break;
        }
      }
    }
    if (matchIndex >= 0) {
      result[matchIndex] = mergeMessage(result[matchIndex], incoming);
    } else {
      result.push({ ...incoming });
      matchIndex = result.length - 1;
    }
    if (incoming.role === "user") lastUserIndex = matchIndex;
  }
  return result;
}

function findDurableReplayAssistant(
  messages: readonly ReconcileMessage[],
  live: ReconcileMessage,
  limit: number,
): number {
  if (live.role !== "assistant" || live.streaming === true || live.text.length === 0) return -1;
  const userIndex = lastUserIndex(messages, limit);
  if (userIndex < 0) return -1;
  let match = -1;
  for (let index = userIndex + 1; index < limit; index += 1) {
    const candidate = messages[index];
    if (candidate.role === "user") break;
    if (candidate.role === "assistant"
      && (candidate.rowId !== undefined || candidate.messageId !== undefined)
      && equivalentAssistant(candidate, live)) {
      if (match >= 0) return -1;
      match = index;
    }
  }
  return match;
}

function lastUserIndexInResult(messages: readonly ReconcileMessage[], before = messages.length): number {
  for (let index = Math.min(before, messages.length) - 1; index >= 0; index -= 1) {
    if (messages[index].role === "user") return index;
  }
  return -1;
}

/**
 * Merge the current live assistant stream into the ordered transcript. Content
 * similarity is allowed only for an assistant row after the latest user turn;
 * an earlier identical answer must never absorb a new turn.
 */
export function mergeLiveTimelineTranscript(
  current: readonly ReconcileMessage[],
  live: readonly ReconcileMessage[],
  activeUserText?: string | null,
): ReconcileMessage[] {
  const result = current.map((message) => ({ ...message }));
  const baseCount = result.length;

  for (const entry of live) {
    const identityIndex = result.findIndex((message) => sameMessageIdentity(message, entry));
    if (identityIndex >= 0) {
      result[identityIndex] = mergeMessage(result[identityIndex], entry);
      continue;
    }

    const equivalentIndex = activeUserText == null
      ? findDurableReplayAssistant(result, entry, baseCount)
      : findCurrentTurnAssistant(result, entry, baseCount, activeUserText);
    if (equivalentIndex >= 0) {
      result[equivalentIndex] = mergeMessage(result[equivalentIndex], entry);
    } else {
      result.push({ ...entry });
    }
  }

  return result;
}

/**
 * Commit one completed live assistant message into the ordered transcript.
 * Stable identity wins; a content match is allowed only after the latest user
 * turn so identical answers from separate turns remain separate rows.
 */
export function mergeCompletedAssistantMessage(
  current: readonly ReconcileMessage[],
  id: string,
  text: string,
  activeUserText?: string | null,
  error?: string,
): ReconcileMessage[] {
  const result = current.map((message) => ({ ...message }));
  const applyCompletion = (message: ReconcileMessage): ReconcileMessage => {
    const next = {
      ...message,
      text: chooseCompletedText(message.text, text),
      streaming: false,
    };
    if (error === undefined) delete next.error;
    else next.error = error;
    return next;
  };
  const identityIndex = result.findIndex(
    (message) => message.role === "assistant" && message.id === id,
  );
  if (identityIndex >= 0) {
    result[identityIndex] = applyCompletion(result[identityIndex]);
    return result;
  }

  const equivalentIndex = activeUserText == null
    ? -1
    : findCurrentTurnAssistant(result, { id, role: "assistant", text }, result.length, activeUserText);
  if (equivalentIndex >= 0) {
    result[equivalentIndex] = applyCompletion(result[equivalentIndex]);
    return result;
  }

  if (activeUserText == null && id.startsWith("assistant-") && text.length > 0) {
    const latestUserIndex = lastUserIndex(result);
    for (let index = latestUserIndex + 1; index < result.length; index += 1) {
      const candidate = result[index];
      if (candidate.role === "user") break;
      if (candidate.role === "assistant" && equivalentAssistant(candidate, { id, role: "assistant", text })) {
        result[index] = applyCompletion(candidate);
        return result;
      }
    }
  }
  if (error !== undefined && activeUserText != null) {
    const userIndex = result.reduce((found, message, index) => message.role === "user" && message.text === activeUserText ? index : found, -1);
    if (userIndex >= 0) {
      for (let index = userIndex + 1; index < result.length; index += 1) {
        const candidate = result[index];
        if (candidate.role === "user") break;
        if (candidate.role === "assistant") {
          result[index] = applyCompletion(candidate);
          return result;
        }
      }
    }
  }
  if (error !== undefined && text.length > 0) {
    const latestUserIndex = lastUserIndex(result);
    for (let index = result.length - 1; index > latestUserIndex; index -= 1) {
      const candidate = result[index];
      if (candidate.role !== "assistant" || candidate.error === undefined) continue;
      if (!equivalentAssistant(candidate, { id, role: "assistant", text })) continue;
      result[index] = applyCompletion(candidate);
      return result;
    }
  }
  if (error !== undefined && text.length === 0) return result;
  result.push({ id, role: "assistant", text, ...(error !== undefined ? { error } : {}) });
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
