import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { useLocation, useSearchParams } from "react-router";
import { api, setManagementProfile } from "@/lib/api";
import { ProfileContext } from "@/contexts/profile-context";
import {
  dashboardInitialProfile,
  initialProfileScope,
  shouldAdoptActiveProfile,
} from "@/lib/profile-bootstrap";

/**
 * Machine-level management-profile scope.
 *
 * One switcher (rendered in the sidebar) decides which profile every
 * management page reads/writes. React STATE is the source of truth; the
 * URL (`?profile=<name>`) is a synchronized projection of it so deep links
 * land scoped and refresh survives. The selection is mirrored into the api
 * module so `fetchJSON` transparently appends it to the profile-scoped
 * endpoint families. "" = the dashboard's own profile.
 *
 * Why state-first instead of URL-first: sidebar nav links are bare paths
 * (`/config`, `/skills`). A URL-derived scope would silently reset to the
 * dashboard's own profile on every nav click — the switcher would LOOK
 * global while normal navigation dropped the write target. With state as
 * truth, the effect below re-asserts `?profile=` onto the new location
 * after each navigation, so the scope survives nav and stays deep-linkable.
 *
 * This exists because "Set as active" on the Profiles page historically only
 * flipped the sticky active_profile file (future CLI/gateway runs). The
 * switcher is the dashboard's write-target selector for Chat and management
 * pages. We now sync the switcher when the sticky active profile differs from
 * the dashboard process on load, and ProfilesPage updates the switcher when
 * you click "Set as active".
 *
 * Persistence: the selection is mirrored to localStorage so a bare page
 * reload / re-navigation without `?profile=` in the URL does NOT silently
 * fall back to the dashboard's own profile. Without this, a destructive
 * action (gateway restart) fired right after a reload could target the
 * wrong profile's gateway while the switcher still visually showed the
 * intended one moments before the reload (#profile-scope-restart-footgun).
 */
const MANAGEMENT_PROFILE_STORAGE_KEY = "hermes.dashboard.managementProfile";

function readStoredProfile(): string {
  try {
    return localStorage.getItem(MANAGEMENT_PROFILE_STORAGE_KEY) ?? "";
  } catch {
    return ""; // localStorage unavailable (private mode, disabled storage): fall back to URL/default
  }
}

function writeStoredProfile(name: string): void {
  try {
    if (name) localStorage.setItem(MANAGEMENT_PROFILE_STORAGE_KEY, name);
    else localStorage.removeItem(MANAGEMENT_PROFILE_STORAGE_KEY);
  } catch {
    // best-effort only — persistence is a convenience, never a hard requirement
  }
}

export function ProfileProvider({ children }: { children: ReactNode }) {
  const [searchParams, setSearchParams] = useSearchParams();
  const { pathname } = useLocation();
  const [profiles, setProfiles] = useState<string[]>([]);
  const [currentProfile, setCurrentProfile] = useState("default");
  const bootstrapProfile = dashboardInitialProfile();

  // Precedence: an explicit URL param (deep link / in-app nav) > the last
  // stored selection (survives a bare reload) > the unified-launch bootstrap
  // preselection injected by the server. Afterwards state leads and the URL
  // follows. The stored selection is a fallback only: it must never mask a
  // scope the URL or the bootstrap payload already named.
  const [profile, setProfileState] = useState(() => {
    const urlScope = searchParams.get("profile");
    if (urlScope !== null) return urlScope;
    return readStoredProfile() || initialProfileScope(searchParams, bootstrapProfile);
  });

  // A profile param that CHANGED (e.g. the Profiles page's "Manage skills &
  // tools" linking to /skills?profile=X) is an explicit scope request and
  // wins over current state. Adopt it during render, before any effect runs,
  // so the URL sync below never sees the old state next to the new URL and
  // writes it back.
  const urlProfile = searchParams.get("profile");
  const [seenUrlProfile, setSeenUrlProfile] = useState(urlProfile);
  if (urlProfile !== seenUrlProfile) {
    setSeenUrlProfile(urlProfile);
    if (urlProfile !== null && urlProfile !== profile) {
      setProfileState(urlProfile);
      // An explicit URL scope is a deliberate selection: persist it so a
      // subsequent reload without ?profile= keeps targeting the same profile.
      writeStoredProfile(urlProfile);
    }
  }

  // Mirror into the api module synchronously on every render where it
  // changed, so fetches fired by child effects in the same commit see it.
  setManagementProfile(profile);

  // Re-assert ?profile= after navigations that dropped it (bare nav links)
  // and after state-only changes. No-ops when already in sync.
  useEffect(() => {
    if ((profile || "") === (urlProfile ?? "")) return;
    setSearchParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        if (profile) next.set("profile", profile);
        else next.delete("profile");
        return next;
      },
      { replace: true },
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pathname, urlProfile, profile]);

  useEffect(() => {
    let cancelled = false;
    const urlProfile = searchParams.get("profile");

    Promise.all([api.getProfiles(), api.getActiveProfile()])
      .then(([profilesRes, info]) => {
        if (cancelled) return;

        setProfiles(profilesRes.profiles.map((p) => p.name));

        const current = info.current || "default";
        const active = info.active || "default";
        setCurrentProfile(current);

        // Explicit URL and unified-launch bootstrap scopes win, and so does a
        // profile already restored from localStorage (the user's last explicit
        // choice for this browser) — otherwise a returning user editing profile
        // B would get silently bounced back to the sticky "active" profile on
        // every reload. Only align to "active" when none of those named a
        // scope, matching the original cold-start behavior.
        if (
          !profile &&
          shouldAdoptActiveProfile(
            urlProfile,
            bootstrapProfile,
            current,
            active,
          )
        ) {
          setManagementProfile(active);
          setProfileState(active);
          writeStoredProfile(active);
        }
      })
      .catch(() => {});

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const setProfile = useCallback(
    (name: string) => {
      setManagementProfile(name);
      setProfileState(name);
      writeStoredProfile(name);
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          if (name) next.set("profile", name);
          else next.delete("profile");
          return next;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  const value = useMemo(
    () => ({ profile, currentProfile, profiles, setProfile }),
    [profile, currentProfile, profiles, setProfile],
  );

  return (
    <ProfileContext.Provider value={value}>{children}</ProfileContext.Provider>
  );
}
