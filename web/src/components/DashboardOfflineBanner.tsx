import { useState } from "react";
import { AlertTriangle, RefreshCw, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { useI18n } from "@/i18n";
import type { SidebarStatus } from "@/hooks/useSidebarStatus";

/**
 * Sticky top-of-app banner shown when `useSidebarStatus` reports
 * `kind === "unreachable"`.
 *
 * Why this is its own component instead of inlining into App.tsx:
 * - Mirrors `ProfileScopeBanner` (one job, top-of-app status banner).
 * - Dismiss state is local and ephemeral. A page reload re-shows the
 *   banner — the user shouldn't be able to permanently hide a critical
 *   "your dashboard is offline" warning.
 *
 * The Retry button forces an immediate poll via the hook's `retry()`
 * rather than waiting up to POLL_MS for the next interval tick. On
 * success the hook flips back to `kind === "live"` and the banner
 * unmounts.
 */
export function DashboardOfflineBanner({
  status,
  retry,
}: DashboardOfflineBannerProps) {
  const { t } = useI18n();
  const [dismissed, setDismissed] = useState(false);

  if (status.kind !== "unreachable" || dismissed) {
    return null;
  }

  // Localized fallbacks: these keys are optional on Translations so
  // non-English locales that haven't been updated yet still render
  // sensibly instead of an empty banner.
  const title =
    t.app.offlineBannerTitle ??
    "Dashboard unreachable — is the terminal session still running?";
  const body =
    t.app.offlineBannerBody ??
    "The backend stopped responding. Data shown here may be stale until the connection comes back.";
  const retryLabel = t.app.offlineRetry ?? t.common.retry;
  const dismissLabel = t.common.close;

  return (
    // mt-14 on mobile clears the fixed lg:hidden header (h-14, z-40) so
    // this banner — the offline signal — is never hidden behind it;
    // lg:mt-0 restores desktop flow. Same convention as ProfileScopeBanner.
    <div
      role="alert"
      aria-live="polite"
      className={cn(
        "mt-14 lg:mt-0",
        "flex items-center justify-between gap-3",
        "border-b border-destructive/40 bg-destructive/10",
        "px-4 py-1.5 text-xs text-destructive",
      )}
    >
      <div className="flex min-w-0 items-center gap-2">
        <AlertTriangle className="h-3.5 w-3.5 shrink-0" aria-hidden />
        <span className="truncate font-mondwest text-display tracking-[0.08em] uppercase">
          {title}
        </span>
        <span className="hidden truncate text-text-secondary sm:inline">
          {body}
        </span>
      </div>

      <div className="flex shrink-0 items-center gap-1">
        <button
          type="button"
          aria-label={retryLabel}
          onClick={retry}
          className={cn(
            "inline-flex items-center gap-1.5",
            "rounded-sm px-2 py-1",
            "font-mondwest text-display tracking-[0.08em] uppercase",
            "text-destructive transition-colors",
            "hover:bg-destructive/15",
            "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-destructive/50",
          )}
        >
          <RefreshCw className="h-3.5 w-3.5" aria-hidden />
          {retryLabel}
        </button>
        <button
          type="button"
          aria-label={dismissLabel}
          onClick={() => setDismissed(true)}
          className={cn(
            "inline-flex h-7 w-7 items-center justify-center",
            "rounded-sm text-destructive transition-colors",
            "hover:bg-destructive/15",
            "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-destructive/50",
          )}
        >
          <X className="h-3.5 w-3.5" aria-hidden />
        </button>
      </div>
    </div>
  );
}

interface DashboardOfflineBannerProps {
  status: SidebarStatus;
  retry: () => void;
}