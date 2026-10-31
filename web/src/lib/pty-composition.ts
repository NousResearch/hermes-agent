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

export type PtyCompositionForwarder = ReturnType<typeof createPtyCompositionForwarder>;

export interface PtyCompositionBridgeHooks {
  // Opens the mobile line-replacement window for the next terminal input.
  armReplacementWindow: () => void;
  isReplacementInput: (input: InputEvent) => boolean;
  // True on mobile-like surfaces, where the commit belongs to the replacement
  // path (xterm's own slice stays mismatched there on purpose) rather than to
  // onData. The bridge stands down instead of moving the commit's channel.
  isMobileInput?: () => boolean;
}

/**
 * Wires xterm's helper textarea to the forwarder. Returns the cleanup.
 *
 * xterm 6 sends a composition as `textarea.value.substring(start)`, with
 * `start` the mirror length recorded at compositionstart. The DOM inserts the
 * commit at the caret, and the dashboard runs screenReaderMode, so the caret
 * follows the arrow keys: a commit made mid-line slices out a shifted tail
 * ("调奇怪）" for "语调奇怪") that the forwarder cannot match, and the PTY gets
 * both. The mirror is rewritten only when that slice disagrees with the
 * commit, so xterm still delivers the word and the forwarder drops its
 * fallback. Should become unnecessary once xterm reads the compositionend data
 * itself (xtermjs/xterm.js#6049).
 *
 * The rewrite stands down whenever the slice is not the commit plus a tail:
 * the mirror was cleared or shrank below `start` (blur does this), or the
 * composition replaced text that was already in the mirror (a recomposition) —
 * slicing a fragment out of that mirror is worse than leaving xterm alone. It
 * also stands down for mobile-like input, whose commit is owned by the
 * replacement path instead (see `isMobileInput`). Text a commit key leaves
 * after an appended commit keeps working (xterm.js #3191); a mid-line slice is
 * replaced by the commit alone, so a character such a key left there does not
 * survive this rewrite — without the bridge it is mis-delivered inside the
 * stale fragment instead (base sends [' ）','녕'] for that case).
 */
export function bindPtyCompositionBridge(
  textarea: HTMLTextAreaElement,
  forwarder: PtyCompositionForwarder,
  hooks: PtyCompositionBridgeHooks,
): () => void {
  let start = 0;
  let composing = false;
  // Same value xterm's own compositionstart listener records.
  const noteStart = () => {
    start = textarea.value.length;
    composing = true;
  };
  const alignMirrorToCommit = (data: string) => {
    const slice = textarea.value.substring(start);
    // Fewer characters than the commit means the mirror shrank under us or the
    // composition replaced text that was already there. xterm's own slice is
    // short or empty in both, and the forwarder's fallback delivers the commit
    // exactly once — rewriting here would turn that single send into a
    // fragment plus the word.
    if (slice.length < data.length) return;
    // Appending at the end of the line, and every case where a commit key's own
    // character follows the commit (xterm.js #3191), already leave the commit
    // heading the slice: the value is correct as it stands.
    if (slice.startsWith(data)) return;
    textarea.value = textarea.value.slice(0, start) + data;
  };
  const onBeforeInput = (ev: Event) => {
    if (hooks.isReplacementInput(ev as InputEvent)) hooks.armReplacementWindow();
  };
  const onCompositionEnd = (ev: CompositionEvent) => {
    const wasComposing = composing;
    composing = false;
    hooks.armReplacementWindow();
    // An empty compositionend data gives nothing to align to, and assigning
    // textarea.value also moves the caret to the end, so leave the DOM and
    // xterm's own slice untouched. A compositionend with no matching
    // compositionstart (Android may fire it twice or trails later input)
    // leaves `start` stale, and mobile input commits through the replacement
    // path: both belong to xterm and the forwarder, not to this rewrite. An
    // aborted composition can leave `composing` set until the next
    // compositionstart resets it, which is why the offset is only ever used
    // for the composition that recorded it.
    if (ev.data && wasComposing && !hooks.isMobileInput?.()) {
      alignMirrorToCommit(ev.data);
    }
    forwarder.onCompositionEnd(ev.data);
  };

  textarea.addEventListener("beforeinput", onBeforeInput, true);
  textarea.addEventListener("compositionstart", noteStart, true);
  textarea.addEventListener("compositionend", onCompositionEnd, true);
  return () => {
    textarea.removeEventListener("beforeinput", onBeforeInput, true);
    textarea.removeEventListener("compositionstart", noteStart, true);
    textarea.removeEventListener("compositionend", onCompositionEnd, true);
  };
}
