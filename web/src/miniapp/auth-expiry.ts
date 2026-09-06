// Decides whether a 401 response means "your session expired" or is just
// the normal first answer for a caller who was never authenticated.
//
// A Mini App reuses ONE initData credential for its whole open lifetime, so
// a 401 AFTER a session was established means the replay window lapsed and
// the only remedy is reopening the app -- worth a dedicated full-screen
// state. But a 401 BEFORE any session existed is simply what an
// unauthenticated browser visitor or an unpaired Telegram user gets on
// their very first /api/miniapp/me call, and those callers must see the
// "not authorized" / "not paired" screens, not "Session expired" (an
// earlier version treated every 401 as expiry and mislabeled both).
//
// A factory rather than module-level state so each mounted app (and each
// test) gets an independent gate.
export interface AuthExpiryGate {
  /** Call when /api/miniapp/me returns a real (non-null) tier. */
  markSessionEstablished(): void;
  /** True when a 401 arriving NOW should be treated as session expiry. */
  isExpiry401(): boolean;
}

export function createAuthExpiryGate(): AuthExpiryGate {
  let established = false;
  return {
    markSessionEstablished() {
      established = true;
    },
    isExpiry401() {
      return established;
    },
  };
}
