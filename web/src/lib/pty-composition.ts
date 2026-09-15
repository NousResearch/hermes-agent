/**
 * Delays an IME/dead-key commit just long enough for xterm to emit onData.
 *
 * xterm is authoritative when it emits the commit. Browsers/layouts where it
 * does not emit onData still forward the native beforeinput/composition commit
 * on the next turn.
 */
export function createPtyCompositionForwarder(send: (data: string) => void) {
  let pending: string | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let matchedTerminalPrefix = "";
  let sawUnrelatedTerminalData = false;
  let recentTerminalData = "";
  let recentTerminalTimer: ReturnType<typeof setTimeout> | null = null;

  const clearPending = () => {
    pending = null;
    matchedTerminalPrefix = "";
    sawUnrelatedTerminalData = false;
    if (timer) {
      clearTimeout(timer);
      timer = null;
    }
  };

  const clearRecentTerminalData = () => {
    recentTerminalData = "";
    if (recentTerminalTimer) {
      clearTimeout(recentTerminalTimer);
      recentTerminalTimer = null;
    }
  };

  const rememberTerminalData = (data: string) => {
    if (data.startsWith("\x1b")) return;
    recentTerminalData += data;
    if (recentTerminalData.length > 512) {
      recentTerminalData = recentTerminalData.slice(-512);
    }
    if (recentTerminalTimer) clearTimeout(recentTerminalTimer);
    recentTerminalTimer = setTimeout(clearRecentTerminalData, 32);
  };

  const scheduleCommit = (data: string | null | undefined) => {
    if (!data) return;
    if (recentTerminalData.endsWith(data)) {
      clearPending();
      clearRecentTerminalData();
      return;
    }
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
  };

  return {
    onBeforeInput(
      inputType: string | undefined,
      data: string | null | undefined,
      isMobileLike: boolean,
    ) {
      if (!shouldForwardPtyBeforeInputCommit(inputType, data, isMobileLike)) {
        return;
      }
      scheduleCommit(data);
    },
    onCompositionEnd(data: string | null) {
      scheduleCommit(data);
    },
    noteTerminalData(data: string) {
      rememberTerminalData(data);
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
    dispose() {
      clearPending();
      clearRecentTerminalData();
    },
  };
}

export function shouldForwardPtyBeforeInputCommit(
  inputType: string | undefined,
  data: string | null | undefined,
  isMobileLike: boolean,
): boolean {
  if (!data) return false;
  if (inputType === "insertFromComposition") return true;
  return (
    isMobileLike && inputType === "insertText" && Array.from(data).length > 1
  );
}
