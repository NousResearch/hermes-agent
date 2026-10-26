import { useCallback, useEffect, useMemo, useState } from "react";
import { useProfileScope } from "@/contexts/useProfileScope";
import { useTheme } from "@/themes";
import { fetchJSON } from "@/lib/api";

export interface ProfileThemeGetResponse {
  profile: string;
  theme?: string;
  inherit_from_default: boolean;
  source: "global" | "default" | "override";
}

export type ProfileThemeSetBody = {
  profile: string;
  theme?: string;
  inherit_from_default: boolean;
};

function profileThemeUrl(profile: string): string {
  return `/api/dashboard/profile-theme?profile=${encodeURIComponent(profile)}`;
}

function setProfileThemeUrl(): string {
  return `/api/dashboard/profile-theme`;
}

export async function getProfileTheme(profile: string) {
  return fetchJSON<ProfileThemeGetResponse>(
    profileThemeUrl(profile),
    { method: "GET" },
  );
}

export async function setProfileTheme(body: ProfileThemeSetBody) {
  return fetchJSON<ProfileThemeGetResponse>(
    setProfileThemeUrl(),
    {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    },
  );
}

export function useProfileTheme() {
  const { profile } = useProfileScope();
  const { setThemeOverride } = useTheme();
  const [raw, setRaw] = useState<ProfileThemeGetResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    getProfileTheme(profile)
      .then((r) => {
        if (!cancelled) setRaw(r);
      })
      .catch(() => {
        if (!cancelled) setRaw(null);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [profile]);

  const effectiveThemeName = useMemo(() => {
    if (!raw) return undefined;
    if (!raw.theme) return undefined;
    if (raw.source === "global" || raw.source === "default") {
      return undefined;
    }
    return raw.theme;
  }, [raw]);

  // Render the profile's override WITHOUT touching the global theme: the
  // provider's `setTheme` persists to localStorage and the server, so using it
  // here leaked one profile's choice into every other profile. The effect's
  // cleanup clears the override whenever the effective theme changes (profile
  // switch, "inherit" toggled on) and on unmount, so `undefined` always means
  // "render the global theme" again.
  useEffect(() => {
    if (!effectiveThemeName) return;
    setThemeOverride(effectiveThemeName);
    return () => setThemeOverride(undefined);
  }, [effectiveThemeName, setThemeOverride]);

  const setOverride = useCallback(
    (themeName: string) => {
      setProfileTheme({ profile, theme: themeName, inherit_from_default: false }).then(
        (r) => setRaw(r),
      );
    },
    [profile],
  );

  const setInherit = useCallback(
    (inherit: boolean) => {
      setProfileTheme({ profile, inherit_from_default: inherit }).then((r) => setRaw(r));
    },
    [profile],
  );

  return {
    profileTheme: raw,
    effectiveThemeName,
    isInherited: raw ? raw.inherit_from_default : false,
    // The theme list marks this one active for a profile-scoped switcher.
    overrideThemeName: raw?.theme,
    source: raw?.source ?? "global",
    isProfileScoped: Boolean(profile) && profile !== "default",
    setOverride,
    setInherit,
    loading,
  };
}
