// Text input from keyboards that report every keystroke as keyCode 229
// without composition events (Android Gboard, IMEs typing punctuation).
//
// xterm 6 recovers that input in CompositionHelper._handleAnyTextareaChanges:
// it snapshots the textarea at each 229 keydown and on the next tick sends
// `newValue.replace(oldValue, "")`. That only holds for a pure append. Gboard
// applies a spelling correction or its double-space period as two 229
// keystrokes in one task (delete, then insert), so the first snapshot is no
// longer a substring and xterm sends the whole textarea ("helo" + "hello " +
// "lo "), the second sends the insert again, and the delete is never sent.
//
// This takes that input over: each beforeinput/input pair is translated into
// the exact edit at the end of the line — a DEL per removed character, then
// the inserted text — and xterm's 229 handler is skipped. Keys with a real
// keyCode and IME compositions stay on xterm's own paths.

const DELETE = "\x7f";

export function textareaEditToPtyData(before: string, after: string): string {
  const removed = Array.from(before);
  const added = Array.from(after);
  let common = 0;
  while (common < removed.length && common < added.length && removed[common] === added[common]) {
    common++;
  }
  return DELETE.repeat(removed.length - common) + added.slice(common).join("");
}

// Pastes and drops are routed through term.paste; compositions belong to
// xterm's CompositionHelper.
function isOwnInputType(inputType: string): boolean {
  return !/Composition|FromPaste|FromDrop/.test(inputType);
}

export function createImeTextInput(
  textarea: HTMLTextAreaElement,
  emit: (data: string) => void,
) {
  let lastKeydownWasIme = false;
  let composing = false;
  let before: string | null = null;

  const owns = (ev: InputEvent) =>
    lastKeydownWasIme && !composing && !ev.isComposing && isOwnInputType(ev.inputType);

  const onKeydown = (ev: KeyboardEvent) => {
    lastKeydownWasIme = ev.keyCode === 229;
  };
  const onCompositionStart = () => {
    composing = true;
  };
  const onCompositionEnd = () => {
    composing = false;
  };
  const onBeforeInput = (ev: Event) => {
    before = owns(ev as InputEvent) ? textarea.value : null;
  };
  const onInput = (ev: Event) => {
    if (before === null || !owns(ev as InputEvent)) return;
    const data = textareaEditToPtyData(before, textarea.value);
    before = null;
    if (data) emit(data);
  };

  const listeners: [string, EventListener][] = [
    ["keydown", onKeydown as EventListener],
    ["compositionstart", onCompositionStart],
    ["compositionend", onCompositionEnd],
    ["beforeinput", onBeforeInput],
    ["input", onInput],
  ];
  for (const [type, fn] of listeners) textarea.addEventListener(type, fn, true);

  return {
    // For term.attachCustomKeyEventHandler: false hands the keystroke to this
    // module so xterm's replace()-based recovery does not run alongside it.
    xtermShouldHandleKeydown: (ev: KeyboardEvent) =>
      !(ev.type === "keydown" && ev.keyCode === 229 && !composing),
    dispose: () => {
      for (const [type, fn] of listeners) textarea.removeEventListener(type, fn, true);
    },
  };
}
