import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { useLocation, useSearchParams } from "react-router";
import { profileVisibleName } from "@hermes/shared";
import { api, setManagementProfile, type ProfileInfo } from "@/lib/api";
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
 */
export function ProfileProvider({ children }: { children: ReactNode }) {
  const [searchParams, setSearchParams] = useSearchParams();
  const { pathname } = useLocation();
  const [profileDetails, setProfileDetails] = useState<ProfileInfo[]>([]);
  const profiles = useMemo(
    () => profileDetails.map((p) => p.name),
    [profileDetails],
  );
  const profileLabel = useCallback(
    (name: string) =>
      profileVisibleName(profileDetails.find((p) => p.name === name) ?? { name }),
    [profileDetails],
  );
  const [currentProfile, setCurrentProfile] = useState("default");
  const bootstrapProfile = dashboardInitialProfile();

  // An explicit URL wins; profile-less deep links inherit the unified-launch
  // preselection injected by the server. Afterwards state leads and the URL
  // follows.
  const [profile, setProfileState] = useState(
    () => initialProfileScope(searchParams, bootstrapProfile),
  );

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
    let latestRequest = 0;
    const urlProfile = searchParams.get("profile");

    const loadProfiles = async () => {
      const request = ++latestRequest;
      const result = await api.getProfiles();
      if (!cancelled && request === latestRequest) {
        setProfileDetails(result.profiles);
      }
    };
    // Names edited in another surface refresh without reapplying active-profile
    // bootstrap: metadata changes must never change the management target.
    const refreshLabels = () => {
      void loadProfiles().catch(() => {});
    };
    window.addEventListener("focus", refreshLabels);

    Promise.all([loadProfiles(), api.getActiveProfile()])
      .then(([, info]) => {
        if (cancelled) return;

        const current = info.current || "default";
        const active = info.active || "default";
        setCurrentProfile(current);

        // Explicit URL and unified-launch bootstrap scopes win. Without
        // either, align the switcher with the sticky active profile so Chat
        // and management pages match what Profiles shows as "active".
        if (
          shouldAdoptActiveProfile(
            urlProfile,
            bootstrapProfile,
            current,
            active,
          )
        ) {
          setManagementProfile(active);
          setProfileState(active);
        }
      })
      .catch(() => {});

    return () => {
      cancelled = true;
      window.removeEventListener("focus", refreshLabels);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const setProfile = useCallback(
    (name: string) => {
      setManagementProfile(name);
      setProfileState(name);
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
    () => ({
      profile,
      currentProfile,
      profiles,
      profileDetails,
      profileLabel,
      setProfile,
    }),
    [profile, currentProfile, profiles, profileDetails, profileLabel, setProfile],
  );

  return (
    <ProfileContext.Provider value={value}>{children}</ProfileContext.Provider>
  );
}
