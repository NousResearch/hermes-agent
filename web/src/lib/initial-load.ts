/**
 * Initial-load state for dashboard pages whose first render is gated on a
 * required read.
 *
 * ConfigPage and EnvPage both block their entire render behind a nullable
 * data object (`!config || !schema`, `!vars`) that is only ever populated
 * from a mount-effect fetch whose rejection was swallowed by
 * `.catch(() => {})`. When that fetch fails the object stays null forever,
 * so the page shows a spinner with no error text and no way to retry — the
 * user cannot tell a down backend from a hung request, and a full browser
 * reload is the only escape.
 *
 * Pages that gate on a `loading` boolean cleared in `.finally()` do not have
 * this problem: they fall through to a rendered (if empty) page. Only the
 * two null-gated pages need this state.
 *
 * Kept pure and React-free so it is unit-testable without a DOM — the same
 * shape as `session-refresh.ts`. The page owns the state; this module owns
 * the transitions and the render decision.
 */

export type InitialLoadStatus = "loading" | "ready" | "failed";

export interface InitialLoadState {
  status: InitialLoadStatus;
  /** Message from the rejected required read; null unless `status` is "failed". */
  error: string | null;
}

export const INITIAL_LOAD_START: InitialLoadState = {
  status: "loading",
  error: null,
};

/**
 * Turn an unknown rejection value into a message safe to render.
 *
 * `api.*` helpers reject with an `Error` whose message carries the server's
 * detail string, but a network-layer failure can reject with anything, so
 * this never assumes a shape. Falls back to a generic message rather than
 * rendering "undefined" or "[object Object]" at the user.
 */
export function initialLoadErrorMessage(reason: unknown): string {
  if (reason instanceof Error && reason.message) return reason.message;
  if (typeof reason === "string" && reason) return reason;
  return "Failed to load. The Hermes server may be unreachable.";
}

/**
 * Record a rejected **required** read.
 *
 * Only reads the page's render gate actually depends on may call this.
 * Optional enrichment reads (ConfigPage's `getDefaults`, `getConfigRaw` and
 * `getStatus`) must keep failing silently — the page is fully usable without
 * them, and raising the error state for a merely-slow `/api/status` would
 * regress a page that renders fine today.
 */
export function initialLoadFailed(reason: unknown): InitialLoadState {
  return { status: "failed", error: initialLoadErrorMessage(reason) };
}

/**
 * Record that every required read resolved.
 *
 * A late-arriving success does not clear a failure: if one required read
 * rejected the page is broken regardless of what its sibling returned, and
 * flipping back to "ready" would strand the page with partial data and no
 * error. Retry is the only way out of "failed".
 */
export function initialLoadSucceeded(prev: InitialLoadState): InitialLoadState {
  return prev.status === "failed" ? prev : { status: "ready", error: null };
}

/**
 * Reset for a retry: back to "loading" with the error cleared, so a second
 * failure surfaces its own message instead of the stale one sticking.
 */
export function initialLoadRetrying(): InitialLoadState {
  return INITIAL_LOAD_START;
}

/**
 * What the page should render right now.
 *
 * `hasRequiredData` is the page's existing null gate (`!!config && !!schema`,
 * `!!vars`), passed in rather than duplicated here so the two stay in sync.
 * The error branch wins over the data branch: a required read that failed
 * means what data did arrive is incomplete.
 */
export function initialLoadView(
  state: InitialLoadState,
  hasRequiredData: boolean,
): "error" | "spinner" | "content" {
  if (state.status === "failed") return "error";
  return hasRequiredData ? "content" : "spinner";
}
