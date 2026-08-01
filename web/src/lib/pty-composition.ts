/**
 * Sends committed IME/dead-key text when xterm does not emit onData.
 *
 * Some browser/layout combinations leave xterm's CompositionHelper without an
 * onData callback. The DOM compositionend event is the authoritative commit.
 * If xterm does emit the same bytes afterwards, consume that one duplicate.
 */
export function createPtyCompositionForwarder(send: (data: string) => void) {
  let pendingDuplicate: string | null = null;

  return {
    onCompositionEnd(data: string) {
      if (!data) return;
      pendingDuplicate = data;
      send(data);
    },
    shouldForwardTerminalData(data: string) {
      if (data === pendingDuplicate) {
        pendingDuplicate = null;
        return false;
      }
      return true;
    },
  };
}
