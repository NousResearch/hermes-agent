/**
 * Delays an IME/dead-key commit just long enough for xterm to emit onData.
 *
 * xterm is authoritative when it emits onData. Browsers/layouts where it
 * does not emit onData still forward the compositionend text on the next turn.
 *
 * Mobile IME keyboards (Android/Gboard) turn composition events and onData
 * into overlapping sources of the same user intent instead of exclusive
 * ones: compositionend can fire twice with identical data, it can trail
 * onData that already carried the committed text, and onData can echo a
 * word the composition path just forwarded. The forwarder keeps a short
 * record of both delivery channels and drops these re-sends.
 */

// How long committed text stays comparable across the two input channels.
// IME re-fires land within a few milliseconds; a human repeating the same
// commit takes far longer, so the window will not swallow real retypes.
const ECHO_WINDOW_MS = 80;

// Cap on retained plain terminal input; only recent text is compared, and
// this keeps a long typing burst from growing the record without bound.
const MAX_DELIVERED_TRACK_CHARS = 256;

interface DeliveredText {
  text: string;
  at: number;
}

export function createPtyCompositionForwarder(send: (data: string) => void) {
  let pending: string | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let matchedTerminalPrefix = "";
  let sawUnrelatedTerminalData = false;
  // Text recently delivered to the PTY through this forwarder's own send.
  let lastSent: DeliveredText | null = null;
  // Plain text recently delivered to the PTY through xterm's onData.
  let lastTerminalData: DeliveredText | null = null;

  const clearPending = () => {
    pending = null;
    matchedTerminalPrefix = "";
    sawUnrelatedTerminalData = false;
    if (timer) {
      clearTimeout(timer);
      timer = null;
    }
  };

  const sendCommitted = (data: string) => {
    lastSent = { text: data, at: Date.now() };
    send(data);
  };

  const withinEchoWindow = (entry: DeliveredText | null): entry is DeliveredText =>
    entry !== null && Date.now() - entry.at <= ECHO_WINDOW_MS;

  return {
    onCompositionEnd(data: string | null) {
      if (!data) return;
      // A compositionend trailing terminal data that already carried the
      // commit is a late duplicate, not a new commit — re-sending it
      // through the fallback duplicates the word in the PTY.
      if (withinEchoWindow(lastTerminalData) && lastTerminalData.text.endsWith(data)) {
        return;
      }
      // The same IME can fire compositionend twice with identical data;
      // the first copy has already been delivered.
      if (withinEchoWindow(lastSent) && lastSent.text === data) {
        return;
      }
      // Preserve rapid consecutive commits instead of discarding the first.
      const previous = pending;
      clearPending();
      if (previous) sendCommitted(previous);
      pending = data;
      timer = setTimeout(() => {
        const committed = pending;
        clearPending();
        if (committed) sendCommitted(committed);
      }, 16);
    },
    noteTerminalData(data: string) {
      if (!data.startsWith("\x1b")) {
        const now = Date.now();
        const carried = withinEchoWindow(lastTerminalData) ? lastTerminalData.text : "";
        lastTerminalData = {
          text: (carried + data).slice(-MAX_DELIVERED_TRACK_CHARS),
          at: now,
        };
      }

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
    // A mobile IME can re-emit just-committed composition text through
    // xterm's onData after the composition path already forwarded it.
    // Returns the portion still worth forwarding: "" for a full duplicate,
    // the new suffix for a strict extension, the input unchanged otherwise.
    filterTerminalData(data: string): string {
      if (!withinEchoWindow(lastSent)) return data;
      if (data === lastSent.text) return "";
      if (data.length > lastSent.text.length && data.startsWith(lastSent.text)) {
        return data.slice(lastSent.text.length);
      }
      return data;
    },
    dispose: () => {
      clearPending();
      lastSent = null;
      lastTerminalData = null;
    },
  };
}
