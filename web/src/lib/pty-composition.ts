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

  const clearPending = () => {
    pending = null;
    matchedTerminalPrefix = "";
    sawUnrelatedTerminalData = false;
    if (timer) {
      clearTimeout(timer);
      timer = null;
    }
  };

  return {
    onCompositionEnd(data: string | null) {
      if (!data) return;
      // Preserve rapid consecutive commits instead of discarding the first.
      const previous = pending;
      clearPending();
      if (previous) send(previous);
      pending = data;
      timer = setTimeout(() => {
        const committed = pending;
        clearPending();
        if (committed) send(committed);
      }, 16);
    },
    noteTerminalData(data: string) {
      if (!pending || data.startsWith("\x1b") || sawUnrelatedTerminalData) return;

      // xterm may split committed text across callbacks, but only a clean,
      // leading match is authoritative. Once unrelated data arrives, retain
      // the fallback even if later callbacks happen to spell the composition.
      const observed = matchedTerminalPrefix + data;
      if (observed.startsWith(pending)) {
        clearPending();
      } else if (pending.startsWith(observed)) {
        matchedTerminalPrefix = observed;
      } else {
        sawUnrelatedTerminalData = true;
      }
    },
    dispose: clearPending,
  };
}
