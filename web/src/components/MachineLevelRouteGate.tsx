import type { ReactNode } from "react";
import { ShieldAlert } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { useProfileScope } from "@/contexts/useProfileScope";
import { useI18n } from "@/i18n";

/**
 * Route-level guard for machine-level pages (/files, /logs, /plugins,
 * /system, and plugins that declare `tab.machineLevel`).
 *
 * The sidebar hides these entries while the ProfileSwitcher manages another
 * profile, but the routes stay reachable by URL (a bookmark must not 404 —
 * see lib/nav-applicability.ts). That leaves one gap: a deep link such as
 * `/system?profile=architect` would render live machine-level controls
 * (restart gateway, browse the filesystem) under a banner claiming every
 * write targets `architect`. Instead of redirecting — which would either
 * drop the scope silently or keep the page one nav away — we render an
 * explicit empty state and offer to switch the dashboard back to its own
 * profile.
 */
export function MachineLevelRouteGate({ children }: { children: ReactNode }) {
  const { profile, currentProfile, setProfile } = useProfileScope();
  const { t } = useI18n();

  // "" = this dashboard's own profile; a scope equal to the dashboard's
  // process profile (e.g. managing `default` from the default dashboard)
  // also means the page shows exactly what the banner says. Same rule as
  // ProfileScopeBanner.
  if (!profile || profile === currentProfile) return <>{children}</>;

  const title =
    t.app.machineLevelPageTitle ??
    "This page is not tied to the managed profile";
  const body = (t.app.machineLevelPageBody ??
    'This page reads and writes the dashboard process itself (the host filesystem, log files, installed plugins, or the running install) — not profile "{name}". Switch the dashboard back to its own profile to use it.')
    .replace("{name}", profile);
  const cta = (t.app.machineLevelPageSwitchCta ??
    "Manage this dashboard ({name})").replace("{name}", currentProfile);

  return (
    <div className="flex min-h-[50vh] flex-col items-center justify-center gap-4 px-6 py-16 text-center">
      <ShieldAlert className="h-10 w-10 text-amber-300" aria-hidden />
      <div className="space-y-2">
        <h2 className="text-lg font-semibold">{title}</h2>
        <p className="mx-auto max-w-md text-sm text-text-secondary">{body}</p>
      </div>
      <Button outlined size="sm" onClick={() => setProfile("")}>
        {cta}
      </Button>
    </div>
  );
}
