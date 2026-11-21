/**
 * IME-aware key handling for dashboard text fields whose bare Enter or Escape
 * performs an action (submit, rename, search, close).
 *
 * Same policy as the desktop app's `apps/desktop/src/lib/ime.ts`: CJK/IME users
 * press Enter to commit a composition and Escape to cancel one, and neither
 * keystroke may double as the field's or the dialog's own shortcut. Browsers
 * signal an in-flight composition two ways:
 *
 * - `isComposing` on the (native) keyboard event — the standard signal.
 * - legacy `keyCode === 229` (VK_PROCESSKEY) — Chromium and Safari still stamp
 *   it on keydowns at composition boundaries, including the commit Enter that
 *   can arrive after `compositionend` with `isComposing` already false.
 *
 * Accepts both React synthetic events (composition state on `nativeEvent`)
 * and plain DOM `KeyboardEvent`s, so window-level listeners share the policy.
 */
export interface ImeAwareKeyEvent {
  key: string;
  isComposing?: boolean;
  keyCode?: number;
  nativeEvent?: {
    isComposing?: boolean;
    keyCode?: number;
  };
}

/** Whether this keyboard event belongs to an active IME composition. */
export function isImeComposing(event: ImeAwareKeyEvent): boolean {
  const native = event.nativeEvent ?? event;

  return (
    Boolean(native.isComposing || event.isComposing) ||
    native.keyCode === 229 ||
    event.keyCode === 229
  );
}

/** Enter pressed as a real submit — not an IME composition commit. */
export function isSubmitEnter(event: ImeAwareKeyEvent): boolean {
  return event.key === "Enter" && !isImeComposing(event);
}
