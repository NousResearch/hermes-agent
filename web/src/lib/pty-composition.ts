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
      clearPending();
      pending = data;
      timer = setTimeout(() => {
        const committed = pending;
        clearPending();
        if (committed) send(committed);
      }, 16);
    },
    noteTerminalData() {
      // Any xterm input in the short grace window is its own composition
      // delivery (possibly chunked or normalization-different), so prefer it
      // over the fallback to avoid duplicate text.
      if (pending) clearPending();
    },
    dispose: clearPending,
  };
}
