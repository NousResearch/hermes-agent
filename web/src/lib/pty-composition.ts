/**
 * Delays an IME/dead-key commit just long enough for xterm to emit onData.
 *
 * xterm is authoritative when it emits the commit. Browsers/layouts where it
 * does not emit onData still forward the compositionend text on the next turn.
 */
// A revised dictation utterance re-fires compositionend within a second or
// two of the previous one as the recognizer refines its guess. A gap longer
// than this means the user moved on (a new, unrelated utterance, possibly
// repeating an earlier word/phrase) rather than revising the same one, so
// the diff baseline must not carry over — otherwise the new utterance gets
// silently truncated or dropped entirely when it shares a prefix with
// whatever was last committed.
const REVISION_WINDOW_MS = 2000;

export function createPtyCompositionForwarder(send: (data: string) => void) {
  let pending: string | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let matchedTerminalPrefix = "";
  let sawUnrelatedTerminalData = false;
  // The full text already reflected in the PTY line via this forwarder (or
  // confirmed to already be there by xterm's own onData). Dictation-style
  // composition re-fires compositionend with the whole revised utterance
  // rather than just the new suffix, so each commit is diffed against this
  // instead of being forwarded whole — otherwise the already-sent prefix
  // gets duplicated. Only valid as a baseline within REVISION_WINDOW_MS of
  // the last commit; see above.
  let lastCommitted = "";
  let lastCommittedAt = 0;

  const clearPending = () => {
    pending = null;
    matchedTerminalPrefix = "";
    sawUnrelatedTerminalData = false;
    if (timer) {
      clearTimeout(timer);
      timer = null;
    }
  };

  const commit = (data: string) => {
    const withinRevisionWindow =
      lastCommittedAt !== 0 && Date.now() - lastCommittedAt <= REVISION_WINDOW_MS;
    const baseline = withinRevisionWindow ? lastCommitted : "";
    const delta = data.startsWith(baseline) ? data.slice(baseline.length) : data;
    lastCommitted = data;
    lastCommittedAt = Date.now();
    if (delta) send(delta);
  };

  return {
    onCompositionEnd(data: string | null) {
      if (!data) return;
      // Preserve rapid consecutive commits instead of discarding the first.
      const previous = pending;
      clearPending();
      if (previous) commit(previous);
      pending = data;
      timer = setTimeout(() => {
        const committed = pending;
        clearPending();
        if (committed) commit(committed);
      }, 16);
    },
    noteTerminalData(data: string) {
      if (!pending || data.startsWith("\x1b") || sawUnrelatedTerminalData) return;

      // xterm may split committed text across callbacks, but only a clean,
      // leading match is authoritative. Once unrelated data arrives, retain
      // the fallback even if later callbacks happen to spell the composition.
      const observed = matchedTerminalPrefix + data;
      if (observed.startsWith(pending)) {
        lastCommitted = pending;
        lastCommittedAt = Date.now();
        clearPending();
      } else if (pending.startsWith(observed)) {
        matchedTerminalPrefix = observed;
      } else {
        sawUnrelatedTerminalData = true;
      }
    },
    dispose: () => {
      clearPending();
      lastCommitted = "";
      lastCommittedAt = 0;
    },
  };
}
