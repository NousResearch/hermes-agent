/**
 * Delays an IME/dead-key commit just long enough for xterm to emit onData.
 *
 * xterm is authoritative when it emits the commit. Browsers/layouts where it
 * does not emit onData still forward the compositionend text on the next turn.
 */
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
  // gets duplicated.
  let lastCommitted = "";

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
    const delta = data.startsWith(lastCommitted)
      ? data.slice(lastCommitted.length)
      : data;
    lastCommitted = data;
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
    },
  };
}
