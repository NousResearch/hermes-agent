// Mobile textarea bridge for xterm.js. Data shape: keypress window keydown(229)
// -> beforeinput -> input -> selectionchange (#136179, #125108: Android reports
// every key as 229, so ownership is decided once per keypress at beforeinput; the
// bridge owns all edits except xterm's single edits at the end caret).

import {
  applyPtyLineEdits,
  CURSOR_LEFT,
  CURSOR_RIGHT,
  caretEditBytes,
  caretGraphemeIndex,
  graphemes,
  isPlainTextData,
} from "@/lib/pty-mobile-input";

export interface MobilePtyTextareaBridge {
  textarea: HTMLTextAreaElement;
  ownsInputEvent(): boolean;
  onTerminalData(data: string): boolean;
  caret(): number;
  // Consumed by ChatPage's #125379 composition guard; stays for non-bridge keyboards
  // that never deliver an input event.
  followsReplayedDelete(): boolean;
  dispose(): void;
}

export function bridgeMobileTextarea(
  textarea: HTMLTextAreaElement,
  send: (data: string) => void,
): MobilePtyTextareaBridge {
  let snapshot = textarea.value;
  let lastEventIsInput = false;
  let lastEventIsDelete = false;
  let ownsEvent = false;
  // Every edit is replayed from this position, so a mid-line caret lands at the
  // caret, not at the end of the line.
  let mirrorCaret = textarea.selectionStart ?? textarea.value.length;

  // A real Gboard correction is one window: several edits inside one keypress
  // (delete then re-insert within a word); ownership is decided at the window's
  // first beforeinput.
  let windowSnapshot = snapshot;
  let windowCaret = mirrorCaret;
  let windowEdits = 0;
  let windowHasBeforeInput = false;

  const caret = () => textarea.selectionStart ?? textarea.value.length;

  const caretAfterEdit = (data: string, from: number): number => {
    let pos = from;
    let i = 0;
    while (i < data.length) {
      const ch = data[i];
      if (ch === "\r" || ch === "\n" || ch === "\x03" || ch === "\x15") {
        pos = 0;
        i += 1;
      } else if (ch === "\x7f" || ch === "\x08") {
        if (pos > 0) pos -= 1;
        i += 1;
      } else if (ch === "\x1b" && data.startsWith(CURSOR_LEFT, i)) {
        pos = Math.max(0, pos - 1);
        i += 3;
      } else if (ch === "\x1b" && data.startsWith(CURSOR_RIGHT, i)) {
        pos += 1;
        i += 3;
      } else if (isPlainTextData(ch)) {
        pos += 1;
        i += 1;
      } else {
        i += 1;
      }
    }
    return pos;
  };

  const graphemeCaret = () => caretGraphemeIndex(graphemes(textarea.value), caret());

  const openWindow = () => {
    snapshot = textarea.value;
    lastEventIsInput = false;
    lastEventIsDelete = false;
    ownsEvent = false;
    windowSnapshot = snapshot;
    windowCaret = graphemeCaret();
    windowEdits = 0;
    windowHasBeforeInput = false;
  };

  const applyEdit = (data: string) => {
    if (isPlainTextData(data)) {
      // Plain text reached the textarea through the browser, so the mirror
      // absorbs it as it stands; rewriting the value here would restart the IME.
      snapshot = textarea.value;
      mirrorCaret = caret();
      openWindow();
      return;
    }
    const next = applyPtyLineEdits(snapshot, data, false, mirrorCaret);
    if (next !== snapshot) textarea.value = next;
    snapshot = next;
    mirrorCaret = caretAfterEdit(data, mirrorCaret);
    // The PTY cursor moved, so the visual caret must move with it; otherwise
    // the next keystroke diffs at the end of the line instead of the caret the
    // user sees.
    const offset = graphemes(next).slice(0, mirrorCaret).join("").length;
    textarea.setSelectionRange(offset, offset);
    openWindow();
  };

  const mirrorCaretTo = (target: number) => {
    if (target > mirrorCaret) send(CURSOR_RIGHT.repeat(target - mirrorCaret));
    else if (target < mirrorCaret) send(CURSOR_LEFT.repeat(mirrorCaret - target));
    mirrorCaret = target;
  };

  // Decided from what the browser is about to apply, not from the observed diff
  // at input.
  const classifyAtBeforeInput = (ev: InputEvent): boolean => {
    const inputType = ev.inputType ?? "";
    if (inputType.startsWith("delete")) return true;
    const ranges =
      typeof ev.getTargetRanges === "function"
        ? Array.from(ev.getTargetRanges())
        : ((ev as InputEvent & { targetRanges?: Array<{ collapsed: boolean }> }).targetRanges ?? []);
    if (ranges.some((r) => r && !r.collapsed)) {
      return true;
    }
    const wLen = graphemes(windowSnapshot).length, gc = graphemeCaret();
    return !(windowEdits === 1 && inputType === "insertText" && windowCaret === wLen && gc === windowCaret);
  };

  const onBeforeInput = (ev: Event) => {
    const input = ev as InputEvent;
    if (!input.inputType) return;
    windowEdits += 1;
    windowHasBeforeInput = true;
    ownsEvent = classifyAtBeforeInput(input);
  };

  const onInput = (ev: Event) => {
    const input = ev as InputEvent;
    // A synthetic input event carries no inputType; it is not a real keyboard
    // edit and must not be replayed (#136179).
    if (!input.inputType) {
      openWindow();
      return;
    }
    // Deletions inside an active composition belong to the composition path;
    // the bridge must not replay them (#124498) and must not claim ownership.
    if (input.isComposing) {
      lastEventIsInput = true;
      openWindow();
      return;
    }
    lastEventIsInput = true;
    // Fallback for keyboards that never fire beforeinput (#136179): the keydown
    // path owns the edit. Never re-own an event already owned at beforeinput.
    if (!windowHasBeforeInput) ownsEvent = true;

    // The byte stream starts from the caret the server already has: the walk the
    // caret handle fired on selectionchange is part of the stream, so the replay
    // continues from the mirror caret, not from the caret at keydown.
    const bytes = caretEditBytes(windowSnapshot, mirrorCaret, textarea.value, graphemeCaret());
    if (bytes !== "") {
      lastEventIsDelete = input.inputType.startsWith("delete");
      if (ownsEvent) {
        send(bytes);
        mirrorCaret = graphemeCaret();
      }
    }
    windowSnapshot = textarea.value;
    windowCaret = graphemeCaret();
    mirrorCaret = windowCaret;
  };

  const onSelectionChange = () => {
    if (textarea.value === windowSnapshot) mirrorCaretTo(graphemeCaret());
  };

  const onKeydown = (ev: KeyboardEvent) => {
    // A synthetic keydown is not the input event's keypress (#136179).
    if (ev.keyCode === 229) openWindow();
  };

  textarea.addEventListener("keydown", onKeydown);
  textarea.addEventListener("beforeinput", onBeforeInput);
  textarea.addEventListener("input", onInput);
  textarea.addEventListener("compositionend", openWindow);
  // Chromium dispatches selectionchange on document for the focused field, not
  // on the element, so an element-level listener never sees IME caret jumps.
  const onDocumentSelectionChange = () => {
    if (document.activeElement === textarea) onSelectionChange();
  };
  document.addEventListener("selectionchange", onDocumentSelectionChange);

  const ownsInputEvent = () => lastEventIsInput && ownsEvent;
  const followsReplayedDelete = () => lastEventIsInput && lastEventIsDelete;

  const onTerminalData = (data: string): boolean => {
    // Plain text for an event the bridge replayed is the device's copy of the
    // same edit (#138081), so ChatPage must not count it twice.
    if (isPlainTextData(data) && ownsInputEvent()) return false;
    applyEdit(data);
    return true;
  };

  return {
    textarea,
    ownsInputEvent,
    onTerminalData,
    caret: graphemeCaret,
    followsReplayedDelete,
    dispose: () => {
      textarea.removeEventListener("keydown", onKeydown);
      textarea.removeEventListener("beforeinput", onBeforeInput);
      textarea.removeEventListener("input", onInput);
      textarea.removeEventListener("compositionend", openWindow);
      document.removeEventListener("selectionchange", onDocumentSelectionChange);
    },
  };
}

