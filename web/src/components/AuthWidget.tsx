/**
 * AuthWidget — sidebar "Logged in as …" affordance for the dashboard
 * OAuth gate (Phase 7 of .hermes/plans/2026-05-21-dashboard-oauth-auth.md).
 *
 * Renders nothing in loopback / --insecure mode. In gated mode, fetches
 * /api/auth/me on mount and surfaces:
 *
 *   - the user_id (truncated to 14 chars + ellipsis) since the Nous Portal
 *     contract V1 doesn't emit email/display_name claims (Contract Anchor
 *     C4 in the plan; the API responds with empty strings for those
 *     fields, so we use user_id as the display value)
 *   - the provider's display_name (looked up from /api/auth/providers,
 *     defaults to the bare provider key)
 *   - a logout button that POSTs /auth/logout and full-page-navigates to
 *     /login (the dashboard becomes inaccessible again)
 *
 * Failure modes:
 *   - The server-injected auth-required flag is false in loopback mode, so
 *     the widget renders nothing without issuing an expected-failure probe.
 *   - Network error in gated mode shows a minimal "auth status unavailable" message
 *     so the user knows the widget tried.
 */

import { useEffect, useState } from "react";
import { api, type AuthMeResponse } from "@/lib/api";
import { shouldProbeDashboardAuth } from "@/lib/dashboard-auth";
import { cn } from "@/lib/utils";
import { LogOut } from "lucide-react";

interface AuthWidgetProps {
  className?: string;
}

/** Truncate ``user_id`` to fit a small UI without revealing the full
 *  opaque identifier. 14 chars is enough to disambiguate users in a
 *  small org and short enough to fit a single sidebar row. */
function truncateUserId(id: string): string {
  if (id.length <= 14) return id;
  return `${id.slice(0, 14)}…`;
}

export function AuthWidget({ className }: AuthWidgetProps) {
  const authRequired = shouldProbeDashboardAuth();
  const [me, setMe] = useState<AuthMeResponse | null>(null);
  const [hidden, setHidden] = useState(!authRequired);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    if (!authRequired) return;
    api
      .getAuthMe()
      .then((data) => {
        if (cancelled) return;
        setMe(data);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        // A 401/403 in gated mode means the session expired between the
        // document load and this identity request. The middleware normally
        // redirects first, but keep the widget quiet during that race.
        const msg = err instanceof Error ? err.message : String(err);
        if (msg.startsWith("401:") || msg.startsWith("403:")) {
          setHidden(true);
          return;
        }
        setError("auth status unavailable");
      });
    return () => {
      cancelled = true;
    };
  }, [authRequired]);

  if (hidden) return null;

  if (error) {
    return (
      <div
        className={cn(
          "px-5 py-2 text-[0.65rem] tracking-[0.05em] text-muted-foreground/70",
          className,
        )}
      >
        {error}
      </div>
    );
  }

  if (!me) {
    // Loading. Reserve the row height so the sidebar doesn't flicker
    // when the data arrives.
    return (
      <div
        className={cn(
          "h-9 px-5 py-2 text-[0.65rem] text-muted-foreground/40",
          className,
        )}
        aria-busy="true"
      >
        …
      </div>
    );
  }

  const handleLogout = () => {
    void api.logout();
  };

  // Prefer display_name → email → truncated user_id. Contract V1 only
  // populates user_id; the fallthroughs are forward-compat for a future
  // Portal that adds a userinfo endpoint (OQ-C1 in the plan).
  const label = me.display_name || me.email || truncateUserId(me.user_id);

  return (
    <div
      className={cn(
        "flex shrink-0 items-center justify-between gap-2",
        "px-5 py-2",
        "border-t border-current/10",
        "text-[0.65rem] tracking-[0.05em]",
        className,
      )}
      role="status"
      aria-label={`Logged in as ${label}`}
    >
      <div className="flex min-w-0 flex-col">
        <span className="truncate font-mono text-foreground/90" title={me.user_id}>
          {label}
        </span>
        <span className="truncate text-muted-foreground/70">
          via {me.provider}
        </span>
      </div>
      <button
        type="button"
        onClick={handleLogout}
        className={cn(
          "shrink-0 rounded p-1.5 text-muted-foreground/70",
          "transition-colors hover:bg-current/10 hover:text-foreground",
          "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-current/40",
        )}
        aria-label="Log out"
        title="Log out"
      >
        <LogOut className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}
