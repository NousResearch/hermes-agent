/**
 * Delays an IME/dead-key commit just long enough for xterm to emit onData.
 *
 * xterm is authoritative when it emits the commit. Browsers/layouts where it
 * does not emit onData still forward the compositionend text on the next turn.
 */
export function createPtyCompositionForwarder(send: (data: string) => void) {
  let pending: string | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;

  const clearPending = () => {
    pending = null;
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
      // xterm delivers the committed text before any following terminal input.
      if (pending && !data.startsWith("\x1b") && data.startsWith(pending)) clearPending();
    },
    dispose: clearPending,
  };
}
