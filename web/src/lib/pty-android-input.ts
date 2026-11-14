import type { Terminal } from "@xterm/xterm";

/**
 * Android screen-reader textarea edits have one synchronous delivery owner.
 * xterm's 229 diff is disabled by the caller; native composition events still
 * reach xterm so its IME view/positioning and the accessible textarea survive.
 * During that transaction only xterm's untagged composition text is discarded.
 * Hardware onKey and explicitly scoped paste remain native delivery owners.
 */
export function createAndroidPtyInput(term: Terminal, send: (data: string) => void) {
  const textarea = term.textarea!;
  const host = term.element!;
  let value = textarea.value;
  let before: string | null = null;
  let composing = false;
  let compositionBase: string | null = null;
  let compositionEnded = false;
  let compositionEpoch = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let hardwareEdit: string | null = null;
  let hardwareData: string | null = null;
  let nativeDepth = 0;

  // Rewrite only the changed suffix, counting Unicode scalars rather than UTF-16
  // code units. Never clear/rewrite the textarea: AT and the browser IME own it.
  const difference = (oldValue: string, newValue: string) => {
    const oldChars = Array.from(oldValue), newChars = Array.from(newValue);
    let prefix = 0;
    while (prefix < oldChars.length && oldChars[prefix] === newChars[prefix]) prefix++;
    return "\x7f".repeat(oldChars.length - prefix) + newChars.slice(prefix).join("");
  };
  const commitComposition = () => {
    if (compositionBase === null) return;
    const delta = difference(compositionBase, textarea.value);
    compositionBase = null;
    compositionEnded = false;
    value = textarea.value;
    before = null;
    if (delta) send(delta);
  };
  const finishEpoch = () => {
    if (timer !== undefined) clearTimeout(timer);
    // Called by a host bubble listener, after xterm has queued its zero-delay
    // finalizer. This fence outlives that finalizer, not an arbitrary echo window.
    timer = setTimeout(() => {
      timer = undefined;
      if (!composing) {
        commitComposition();
        compositionEpoch = false;
      }
    }, 0);
  };
  const start = () => {
    if (compositionEnded) commitComposition();
    composing = true;
    compositionEpoch = true;
    compositionBase = textarea.value;
    hardwareEdit = null;
    before = null;
  };
  const end = () => {
    composing = false;
    // Duplicate/late end events may no longer have an owned DOM transaction.
    compositionEnded = compositionBase !== null;
    // A delayed end can follow a hardware/paste flush whose fence has drained.
    // xterm nevertheless queues another native finalizer for this event.
    compositionEpoch = true;
  };
  const beforeInput = (event: Event) => {
    const ev = event as InputEvent;
    // Chromium may apply the final composition DOM mutation after end. Do not
    // commit the preceding preedit when this event belongs to that same commit.
    const finalComposition = ev.isComposing || /Composition/.test(ev.inputType);
    if (compositionEnded && !finalComposition) commitComposition();
    before = textarea.value;
  };
  const input = (event: Event) => {
    const ev = event as InputEvent;
    const oldValue = before ?? value;
    before = null;
    value = textarea.value;
    if (composing || ev.isComposing) return;
    if (compositionEnded) {
      commitComposition();
      return;
    }
    // Paste is delivered by xterm's native paste path, not as a DOM text edit.
    if (ev.inputType === "insertFromPaste" || ev.inputType === "insertFromDrop") {
      hardwareEdit = null;
      return;
    }
    const delta = difference(oldValue, value);
    const alreadyDelivered = hardwareEdit;
    hardwareEdit = null;
    if (delta && delta !== alreadyDelivered) send(delta);
  };
  const keyUp = () => { hardwareEdit = null; hardwareData = null; };
  const key = term.onKey(({ key }) => {
    hardwareEdit = key;
    hardwareData = key;
    value = textarea.value;
  });
  host.addEventListener("compositionstart", start, true);
  host.addEventListener("compositionend", end, true);
  host.addEventListener("compositionend", finishEpoch);
  host.addEventListener("beforeinput", beforeInput, true);
  host.addEventListener("input", input, true);
  host.addEventListener("keyup", keyUp);

  return {
    beforeKey(event: KeyboardEvent) {
      hardwareEdit = null;
      hardwareData = null;
      // Hardware Enter can synchronously flush native composition before onKey.
      // Commit its DOM transaction first, retaining the suppression fence.
      if (event.keyCode !== 229 && event.key !== "Process" &&
          !["Shift", "Control", "Alt", "Meta", "CapsLock"].includes(event.key)) {
        if (compositionBase !== null) {
          commitComposition();
          composing = false;
          finishEpoch();
        }
      }
      return event.keyCode !== 229 && event.key !== "Process";
    },
    suppressTerminalData(data: string) {
      if (hardwareData === data) { hardwareData = null; return false; }
      // Terminal replies/control traffic are not composition text. Paste calls
      // are explicitly scoped because onData itself carries no provenance.
      return compositionEpoch && !nativeDepth && !data.includes("\x1b");
    },
    native<T>(action: () => T): T {
      if (compositionBase !== null) {
        commitComposition();
        composing = false;
        finishEpoch();
      }
      nativeDepth++;
      try { return action(); } finally {
        nativeDepth--;
        value = textarea.value;
        before = null;
      }
    },
    dispose() {
      if (timer !== undefined) clearTimeout(timer);
      key.dispose();
      host.removeEventListener("compositionstart", start, true);
      host.removeEventListener("compositionend", end, true);
      host.removeEventListener("compositionend", finishEpoch);
      host.removeEventListener("beforeinput", beforeInput, true);
      host.removeEventListener("input", input, true);
      host.removeEventListener("keyup", keyUp);
    },
  };
}
