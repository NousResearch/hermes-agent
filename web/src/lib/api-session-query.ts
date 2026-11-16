/**
 * Session-list query helpers (#C3, t_406dbcb3).
 *
 * Moved out of `api.ts` verbatim when the conditional-GET plumbing landed in
 * that file: the health ratchet allows an over-cap file to shrink only, and
 * the session list's URL construction is cohesive enough to own a sibling.
 * The management profile is passed IN (the caller reads it) so this module
 * needs no import back into `api.ts`.
 */

export function appendProfileParam(url: string, profile?: string): string {
  if (!profile || url.includes("profile=")) return url;
  return `${url}${url.includes("?") ? "&" : "?"}profile=${encodeURIComponent(profile)}`;
}

export function appendQueryParam(url: string, key: string, value?: string): string {
  if (!value) return url;
  return `${url}${url.includes("?") ? "&" : "?"}${key}=${encodeURIComponent(value)}`;
}

export interface SessionQueryOptions {
  profile?: string;
  order?: "created" | "recent";
  source?: string | null;
  sources?: string[];
  excludeSources?: string[];
}

export function normalizeSessionQueryOptions(
  profileOrOptions: string | SessionQueryOptions | undefined,
  order: "created" | "recent",
  defaultProfile: string,
): SessionQueryOptions {
  if (typeof profileOrOptions === "string") {
    return { profile: profileOrOptions, order };
  }
  return {
    profile: defaultProfile,
    order,
    ...(profileOrOptions ?? {}),
  };
}

export function appendSessionFilters(url: string, options: SessionQueryOptions): string {
  let next = url;
  next = appendQueryParam(next, "source", options.source ?? undefined);
  if (options.sources && options.sources.length > 0) {
    next = appendQueryParam(next, "sources", options.sources.join(","));
  }
  if (options.excludeSources && options.excludeSources.length > 0) {
    next = appendQueryParam(next, "exclude_sources", options.excludeSources.join(","));
  }
  return appendProfileParam(next, options.profile);
}
