/**
 * Recovers plain `insertText` letters that xterm 6.0 silently swallows on
 * Android keyboards (#136179).
 *
 * Gboard's English layout delivers every letter as a `keydown` with
 * `key = "Unidentified"` / `keyCode = 229` and **no matching keyup**, followed
 * by a `beforeinput`/`input` pair with `inputType = "insertText"`. xterm's
 * `_inputEvent` drops that insertText because `_keyDownSeen` is still armed
 * (`!e.composed || !this._keyDownSeen`) and the 229 never resolves into real
 * composition events — so mobile chat becomes paste-only. The tracker below
 * recognises exactly that dangling-229 shape and routes the letter through
 * the composition commit forwarder, which already defers to xterm's onData
 * and de-duplicates echoes. Every other shape is left to xterm: a 229 keyup
 * before the input, plain desktop keydowns, and real composition commits
 * (whose trailing insertText lands right after compositionend).
 */

import { shouldTreatInputAsMobileReplacement } from "./pty-mobile-input";

// A real composition commit can emit a trailing `insertText` input event a
// few milliseconds after compositionend; that path already owns the text, so
// the fallback must not arm for it. Matches the forwarder's echo window.
const RECENT_COMPOSITION_END_MS = 80;

export function isImePlaceholderKey(
  ev: Pick<KeyboardEvent, "keyCode" | "key">,
): boolean {
  return ev.keyCode === 229 || ev.key === "Unidentified";
}

export function createDanglingImeInsertTextTracker() {
  let danglingImeKeydown = false;
  let lastCompositionEndAt = 0;

  return {
    onKeydown(ev: Pick<KeyboardEvent, "keyCode" | "key">) {
      danglingImeKeydown = isImePlaceholderKey(ev);
    },
    onKeyup(ev: Pick<KeyboardEvent, "keyCode" | "key">) {
      // A keyup for the placeholder key means the sequence is not dangling
      // and xterm will deliver the input itself.
      if (isImePlaceholderKey(ev)) {
        danglingImeKeydown = false;
      }
    },
    noteCompositionEnd(at: number = Date.now()) {
      lastCompositionEndAt = at;
    },
    shouldFallback(
      ev: Pick<InputEvent, "inputType" | "data">,
      now: number = Date.now(),
    ): boolean {
      if (ev.inputType !== "insertText" || !ev.data) return false;
      if (!danglingImeKeydown) return false;
      return now - lastCompositionEndAt > RECENT_COMPOSITION_END_MS;
    },
  };
}

export interface PtyTextareaInputGuardHooks {
  isMobileLike: boolean;
  markReplacementWindow(): void;
  /** compositionend commits — the composition forwarder's main channel. */
  onCompositionCommit(data: string | null): void;
  /** insertText letters xterm swallowed behind a dangling IME keydown. */
  onDanglingInsertText(data: string): void;
}

/**
 * Installs the terminal textarea's mobile-input listeners (replacement
 * window marking, composition commits, and the dangling-229 insertText
 * fallback) and returns their cleanup.
 */
export function attachPtyTextareaInputGuards(
  textarea: HTMLTextAreaElement,
  hooks: PtyTextareaInputGuardHooks,
): () => void {
  const tracker = createDanglingImeInsertTextTracker();

  const onBeforeInput = (ev: Event) => {
    const input = ev as InputEvent;
    if (
      shouldTreatInputAsMobileReplacement(
        input.inputType,
        input.data,
        hooks.isMobileLike,
      )
    ) {
      hooks.markReplacementWindow();
    }
    if (tracker.shouldFallback(input)) {
      hooks.onDanglingInsertText(input.data ?? "");
    }
  };
  const onCompositionEnd = (ev: CompositionEvent) => {
    hooks.markReplacementWindow();
    tracker.noteCompositionEnd();
    hooks.onCompositionCommit(ev.data);
  };
  const onKeydown = (ev: KeyboardEvent) => tracker.onKeydown(ev);
  const onKeyup = (ev: KeyboardEvent) => tracker.onKeyup(ev);

  textarea.addEventListener("beforeinput", onBeforeInput, true);
  textarea.addEventListener("compositionend", onCompositionEnd, true);
  textarea.addEventListener("keydown", onKeydown, true);
  textarea.addEventListener("keyup", onKeyup, true);
  return () => {
    textarea.removeEventListener("beforeinput", onBeforeInput, true);
    textarea.removeEventListener("compositionend", onCompositionEnd, true);
    textarea.removeEventListener("keydown", onKeydown, true);
    textarea.removeEventListener("keyup", onKeyup, true);
  };
}
