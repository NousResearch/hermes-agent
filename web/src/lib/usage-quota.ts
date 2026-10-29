export interface UsageAccountStatus {
  available: boolean;
  windows: unknown[];
  unavailable_reason?: string;
}

export function getUsageAccountUnavailableMessage(account: UsageAccountStatus): string | null {
  return account.available && account.windows.length > 0 ? null : account.unavailable_reason ?? "unavailable";
}

/**
 * Collapse a backend unavailable_reason into a short tag for one-line display.
 * The full text stays available via a title/tooltip at the call site.
 */
export function shortUnavailableReason(reason: string): string {
  const http = /HTTP\s+(\d{3})/i.exec(reason);
  if (http) return `HTTP ${http[1]}`;
  if (/no usable token/i.test(reason)) return "no token";
  if (/exhausted|rate.?limit/i.test(reason)) return "exhausted";
  const firstClause = reason.split(/[.;:\n]/)[0]?.trim() ?? "";
  return firstClause.length > 28 ? `${firstClause.slice(0, 27)}…` : firstClause || "unavailable";
}
