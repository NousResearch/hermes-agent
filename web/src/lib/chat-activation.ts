/**
 * Chat PTY activation latch.
 *
 * The dashboard keeps `ChatPage` mounted persistently (just hidden with CSS)
 * on every route so the embedded chat PTY survives tab switches. The downside
 * is that the PTY-connect effect would otherwise open `/api/pty` — which spawns
 * the whole TUI + agent bootstrap (on a fresh checkout this prints
 * `Installing TUI dependencies…` and runs `npm install`) — the moment the
 * dashboard loads *any* page, even one the user never navigates the chat into.
 *
 * The fix is to only open the PTY once the chat tab has actually been active,
 * while keeping activation **sticky** so the PTY still persists across later
 * tab switches. This helper computes that latch: once `true`, it stays `true`.
 */
export function latchChatActivation(previous: boolean, isActive: boolean): boolean {
  return previous || isActive;
}

/**
 * Module-level external store holding the latch.
 *
 * `ChatPage` reads it through `useSyncExternalStore`: a render-phase
 * `setState` latch is illegal under react-hooks/set-state-in-effect, and a
 * plain state+effect latch would have to call setState synchronously in the
 * effect body. An external store sidesteps both — the effect only calls
 * `activate()`, and React re-renders from the subscription notification.
 *
 * Module scope (not React state) also matches the real-world semantics: the
 * "sticky" property survives even an unmount/remount of the page within one
 * document lifetime, which is exactly when the PTY bootstrap cost would
 * otherwise be paid twice.
 */
const listeners = new Set<() => void>();
let activated = false;

export const chatActivationStore = {
  subscribe(listener: () => void): () => void {
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  },
  getSnapshot(): boolean {
    return activated;
  },
  /** Mark the chat tab as activated; sticky, no-op when already active. */
  activate(): void {
    const next = latchChatActivation(activated, true);
    if (next === activated) return;
    activated = next;
    for (const listener of listeners) listener();
  },
};
