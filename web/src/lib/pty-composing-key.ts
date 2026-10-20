/**
 * Keys pressed while an IME composition is open.
 *
 * SwiftKey ends a composed word with a real space or Enter key (keyCode 32
 * or 13), unlike Gboard's keyCode-229 placeholders. xterm 6 reacts to such
 * a keydown by finalizing the composition at once with the text composed so
 * far, and the compositionend that follows finalizes it again with the
 * textarea tail. With autocorrect the first send is the uncorrected word
 * ("helo" then "hello "), and Enter submits the wrong word before the
 * corrected one lands on the next line (#125517 family).
 *
 * The fix keeps such keydowns away from xterm altogether: compositionend
 * then delivers the final text exactly once (the browser still inserts the
 * space into the textarea, so it rides along), and Enter is replayed after
 * that commit.
 */

// How long a deferred Enter waits for compositionend before it is sent
// anyway, so an IME that swallows the composition cannot eat the Enter.
export const COMPOSING_ENTER_FALLBACK_MS = 100;

// Delay between compositionend and the replayed Enter: past xterm's own
// setTimeout(0) finalize and the 16 ms composition fallback forwarder.
export const COMPOSING_ENTER_REPLAY_MS = 20;

type ComposingKeyEvent = Pick<
  KeyboardEvent,
  "type" | "keyCode" | "key" | "isComposing" | "preventDefault"
>;

export interface ComposingKeydown {
  /** The event must not reach xterm's keydown path. */
  consume: boolean;
  /** The key was Enter: replay it after the composition commits. */
  deferEnter: boolean;
}

export function resolveComposingKeydown(
  ev: Pick<KeyboardEvent, "type" | "keyCode" | "key" | "isComposing">,
  composing: boolean,
): ComposingKeydown {
  if (ev.type !== "keydown" || ev.keyCode === 229 || ev.key === "Process") {
    return { consume: false, deferEnter: false };
  }
  if (!(ev.isComposing || composing)) {
    return { consume: false, deferEnter: false };
  }
  return { consume: true, deferEnter: ev.key === "Enter" };
}

export interface PtyComposingKeyGuard {
  isComposing(): boolean;
  /** Returns true when xterm must ignore this keydown. */
  onKeydown(ev: ComposingKeyEvent): boolean;
  dispose(): void;
}

/**
 * Tracks the textarea's composition state from its own capture-phase
 * listeners (which run before xterm's) and replays a deferred Enter once
 * the composition has committed.
 */
export function createComposingKeyGuard(
  textarea: EventTarget,
  sendEnter: () => void,
): PtyComposingKeyGuard {
  let composing = false;
  let fallback: ReturnType<typeof setTimeout> | null = null;
  let replay: ReturnType<typeof setTimeout> | null = null;

  const clearTimers = () => {
    if (fallback) clearTimeout(fallback);
    if (replay) clearTimeout(replay);
    fallback = null;
    replay = null;
  };

  const onCompositionStart = () => {
    composing = true;
  };
  const onCompositionEnd = () => {
    composing = false;
    if (!fallback) return;
    clearTimeout(fallback);
    fallback = null;
    // xterm's own compositionend listener runs after this capture-phase one
    // and delivers the committed text from a setTimeout(0); the dashboard's
    // composition fallback forwarder fires 16 ms later if xterm did not.
    // The Enter follows both.
    replay = setTimeout(() => {
      replay = null;
      sendEnter();
    }, COMPOSING_ENTER_REPLAY_MS);
  };

  textarea.addEventListener("compositionstart", onCompositionStart, true);
  textarea.addEventListener("compositionend", onCompositionEnd, true);

  return {
    isComposing: () => composing,
    onKeydown(ev) {
      const decision = resolveComposingKeydown(ev, composing);
      if (!decision.consume) return false;
      if (decision.deferEnter) {
        // No newline in the textarea: the replayed "\r" is the only Enter.
        ev.preventDefault();
        if (fallback) clearTimeout(fallback);
        fallback = setTimeout(() => {
          fallback = null;
          sendEnter();
        }, COMPOSING_ENTER_FALLBACK_MS);
      }
      return true;
    },
    dispose() {
      clearTimers();
      textarea.removeEventListener("compositionstart", onCompositionStart, true);
      textarea.removeEventListener("compositionend", onCompositionEnd, true);
    },
  };
}
