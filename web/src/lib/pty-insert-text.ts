/**
 * Recovers text xterm drops while `screenReaderMode` is on.
 *
 * xterm only honours `insertText` input events when screen-reader mode is off
 * (CoreBrowserTerminal._inputEvent, xterm.js 6). Android soft keyboards and
 * CDP `Input.insertText` commit text as `insertText` with no keydown, so with
 * the mode on the characters pile up in the hidden textarea and never reach
 * onData — the composer looks dead (#145296).
 *
 * `screenReaderMode` is what gives assistive technology its composer
 * (#36784), so it stays on for everyone. This forwarder reads the committed
 * text off the textarea's `input` event instead, and drops anything xterm
 * already delivered through onData so ordinary keydown typing is not doubled.
 */

// How long an onData delivery may still answer for the next input event. Both
// channels fire inside one keystroke, so the window only covers an event loop
// hiccup reordering them.
const ECHO_WINDOW_MS = 50;

// Cap on retained plain terminal input; only the latest burst is compared.
const MAX_DELIVERED_TRACK_CHARS = 256;

interface DeliveredText {
  text: string;
  at: number;
}

export function createPtyInsertTextForwarder() {
  // Plain text delivered to the PTY through xterm's onData, still answerable
  // for the input event that follows it.
  let lastTerminalData: DeliveredText | null = null;

  const withinEchoWindow = (entry: DeliveredText | null): entry is DeliveredText =>
    entry !== null && Date.now() - entry.at <= ECHO_WINDOW_MS;

  return {
    noteTerminalData(data: string) {
      if (data.startsWith("\x1b")) return;
      const carried = withinEchoWindow(lastTerminalData) ? lastTerminalData.text : "";
      lastTerminalData = {
        text: (carried + data).slice(-MAX_DELIVERED_TRACK_CHARS),
        at: Date.now(),
      };
    },
    // Returns the text the PTY still needs, "" when xterm already carried it,
    // and null for input that belongs to another path (composition, no data).
    // The record is consumed per event: a character typed twice must not match
    // its predecessor's delivery and vanish.
    takeUnforwardedText(
      input: Pick<InputEvent, "data" | "inputType" | "isComposing">,
    ): string | null {
      const typed = input.data;
      if (!typed) return null;
      // Composition text is committed on compositionend through
      // createPtyCompositionForwarder; forwarding it here would double it up.
      if (input.isComposing || input.inputType === "insertCompositionText") {
        return null;
      }
      const carried =
        withinEchoWindow(lastTerminalData) && lastTerminalData.text.endsWith(typed);
      lastTerminalData = null;
      return carried ? "" : typed;
    },
    dispose() {
      lastTerminalData = null;
    },
  };
}
