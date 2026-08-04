export const DASHBOARD_UNREACHABLE_CODE = "DASHBOARD_UNREACHABLE" as const;

export class DashboardUnreachableError extends Error {
  readonly code = DASHBOARD_UNREACHABLE_CODE;
  readonly baseUrl: string;

  constructor(baseUrl: string, options?: { cause?: unknown }) {
    const cause = options?.cause;
    const detail = cause instanceof Error && cause.message ? `: ${cause.message}` : "";
    super(`Dashboard backend is unreachable at ${baseUrl}${detail}`, options);
    this.name = "DashboardUnreachableError";
    this.baseUrl = baseUrl;
  }
}

export function isDashboardUnreachableError(
  error: unknown,
): error is DashboardUnreachableError {
  return (
    error instanceof DashboardUnreachableError ||
    (typeof error === "object" &&
      error !== null &&
      (error as { code?: unknown }).code === DASHBOARD_UNREACHABLE_CODE)
  );
}

export type DashboardReachability = "reachable" | "unreachable";

type ReachabilityListener = () => void;

let reachability: DashboardReachability = "reachable";
const listeners = new Set<ReachabilityListener>();

export function getDashboardReachability(): DashboardReachability {
  return reachability;
}

export function subscribeDashboardReachability(
  listener: ReachabilityListener,
): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function markDashboardReachable(): void {
  setDashboardReachability("reachable");
}

export function markDashboardUnreachable(): void {
  setDashboardReachability("unreachable");
}

function setDashboardReachability(next: DashboardReachability): void {
  if (next === reachability) return;
  reachability = next;
  for (const listener of listeners) {
    try {
      listener();
    } catch {
      // Reachability observers must never alter the request that reported it.
    }
  }
}
