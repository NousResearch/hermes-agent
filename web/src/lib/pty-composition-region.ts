/**
 * Decides what a compositionend hands to the composition fallback forwarder.
 *
 * SwiftKey commits Latin letters one by one (keyCode 229 + insertText, no
 * composition) and, when the space bar is pressed, re-opens a composition
 * over the word it just committed: compositionstart with the word already
 * in the textarea, compositionupdate carrying that same word, compositionend
 * with it, then the space as plain insertText. The PTY received the word
 * seconds earlier, so the fallback forwarder (which only remembers the last
 * 80 ms of terminal data) re-sent it: "The The".
 *
 * Only text the composition added beyond what the textarea held at
 * compositionstart can be something xterm dropped. A composition re-opened
 * over committed text is left to xterm, which forwards just the tail (the
 * space); a correction made inside that re-opened region cannot be applied
 * without retracting committed bytes and is not attempted here.
 */

export interface CompositionRegionHandlers {
  /** Every compositionend, before the commit decision. */
  onEnd(): void;
  /** Composed text worth handing to the fallback forwarder. */
  onCommit(data: string): void;
}

export function createCompositionRegionTracker(
  textarea: Pick<HTMLTextAreaElement, "value">,
  handlers: CompositionRegionHandlers,
) {
  let startLength = 0;
  let reopened = false;
  let sawUpdate = false;

  return {
    onStart() {
      startLength = textarea.value.length;
      reopened = false;
      sawUpdate = false;
    },
    onUpdate(data: string) {
      if (sawUpdate) return;
      sawUpdate = true;
      // The first update of a fresh composition grows the textarea; a
      // composition re-opened over committed text only re-states its tail.
      const value = textarea.value;
      reopened =
        data.length > 0 && value.length === startLength && value.endsWith(data);
    },
    onEnd(data: string | null) {
      handlers.onEnd();
      if (!data || reopened || textarea.value.length <= startLength) return;
      handlers.onCommit(data);
    },
    isReopened: () => reopened,
  };
}

/** Installs capture-phase listeners (they run before xterm's own). */
export function attachCompositionRegionTracker(
  textarea: HTMLTextAreaElement,
  handlers: CompositionRegionHandlers,
): { dispose(): void } {
  const tracker = createCompositionRegionTracker(textarea, handlers);
  const onStart = () => tracker.onStart();
  const onUpdate = (ev: CompositionEvent) => tracker.onUpdate(ev.data ?? "");
  const onEnd = (ev: CompositionEvent) => tracker.onEnd(ev.data);
  textarea.addEventListener("compositionstart", onStart, true);
  textarea.addEventListener("compositionupdate", onUpdate, true);
  textarea.addEventListener("compositionend", onEnd, true);
  return {
    dispose() {
      textarea.removeEventListener("compositionstart", onStart, true);
      textarea.removeEventListener("compositionupdate", onUpdate, true);
      textarea.removeEventListener("compositionend", onEnd, true);
    },
  };
}
