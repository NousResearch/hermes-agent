import type { MiniAppStatusExtra } from "./types";

/** The Users tab's restart-needed banner condition: the gateway started before
 * the last `.env` change. Explicit `!== null` checks on both operands, short-circuiting
 * before the `>` comparison ever runs -- a raw `>` with either side `null`
 * would coerce (`null > 5` is `false`, `5 > null` is `true`, `null > null`
 * is `false`), which is a JS trap that reads as "safe" by accident rather
 * than by an explicit fail-closed decision. Any null on either operand
 * must resolve to "no banner", never a false-positive or false-negative.
 */
export function needsRestartBanner(statusExtra: MiniAppStatusExtra): boolean {
  return (
    statusExtra.gateway_start_time !== null &&
    statusExtra.telegram_allowlist_updated_at !== null &&
    statusExtra.telegram_allowlist_updated_at > statusExtra.gateway_start_time
  );
}
