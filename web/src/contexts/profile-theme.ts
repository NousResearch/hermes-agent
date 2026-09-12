import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import { useProfileScope } from "@/contexts/useProfileScope";
import { useTheme } from "@/themes";
import type { ProfileThemeGetResponse } from "@/lib/api";

/** Per-profile theme bridge.

 * When the dashboard is scoped to a named profile (not "" and not "default"),
 * the user can either inherit the default profile's theme or pin an explicit
 * override. The bridge resolves the effective theme name and keeps the existing
 * ThemeProvider in sync.
 *
 * Storage: server-side config.yaml under `dashboard.profile_themes.<name>`
 * (same layer as the global theme). We cache the response in memory; the
 * ThemeProvider's own localStorage/global theme plumbing stays untouched.
 */
export function useProfileTheme() {
  const { profile } = useProfileScope();
  const { setTheme } = useTheme();
  const [raw, setRaw] = useState<ProfileThemeGetResponse | null>(null);
  const [loading, setLoading] = useState(true);

  // Re-fetch + re-apply whenever the scoped profile changes (or on mount).
  useEffect(() => {
    setLoading(true);
    let cancelled = false;
    api
      .getProfileTheme(profile)
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

  // When the per-profile resolved theme becomes explicit, push it into the
  // global ThemeProvider so the palette changes immediately (without this, the
  // ThemeProvider would keep showing the previous global/default theme until
  // the user manually picks one).
  useEffect(() => {
    if (!raw) return;
    if (raw.source === "override" && raw.theme) {
      setTheme(raw.theme);
    } else {
      // Resolved to inherit from default / global — call setTheme with the
      // currently-active theme name so the ThemeProvider does not visibly switch
      // away from whatever the user is seeing.
      const { themeName } = useTheme();
      if (themeName) setTheme(themeName);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [raw]);

  const effectiveThemeName = useMemo(() => {
    if (!raw || raw.theme === undefined) return undefined;
    if (raw.source === "global" || raw.source === "default") {
      return undefined; // inherit — let ThemeProvider resolve globally
    }
    return raw.theme;
  }, [raw]);

  const setInherit = useCallback(
    (inherit: boolean) => {
      if (!profile) return;
      api
        .setProfileTheme({ profile, inherit_from_default: inherit ? true : false })
        .catch(() => {});
    },
    [profile],
  );

  const setOverride = useCallback(
    (themeName: string) => {
      if (!profile) return;
      api
        .setProfileTheme({ profile, inherit_from_default: false, theme: themeName })
        .catch(() => {});
    },
    [profile],
  );

  return {
    /** Current scoped profile ("" = dashboard's own, "default" = sticky default). */
    profile,
    /** Whether we're showing the per-profile picker at all. */
    isProfileScoped: Boolean(profile) && profile !== "default",
    /** Effective theme name that ThemeProvider should render. */
    effectiveThemeName,
    /** True when the per-profile override is inherited from the default profile. */
    isInherited: raw ? raw.source === "default" || raw.source === "global" : true,
    /** Source of the current effective theme. */
    source: raw?.source ?? "global",
    /** Current override theme (only meaningful when isInherited is false). */
    overrideThemeName: raw?.theme,
    /** Switch the per-profile setting to inherit from default. */
    setInherit,
    /** Switch the per-profile setting to an explicit override. */
    setOverride,
    /** Whether the backend data is being fetched. */
    loading,
  };
}

interface UseProfileThemeReturn {
  profile: string;
  isProfileScoped: boolean;
  effectiveThemeName?: string;
  isInherited: boolean;
  source: "global" | "default" | "override";
  overrideThemeName?: string;
  setInherit: (inherit: boolean) => void;
  setOverride: (themeName: string) => void;
  loading: boolean;
}

export type { UseProfileThemeReturn };
