/** URL/query-string builders for the dashboard API client.

Extracted from ``api.ts`` (topical sibling; that file is over the FILE_LINES
cap and may only shrink). All functions are pure string builders.
 */

/** Encode a plugin registry key for URL paths (preserves `/` segment separators). */
export function pluginPath(name: string): string {
  return name.split("/").map(encodeURIComponent).join("/");
}

/** Build a ``?profile=<name>`` query suffix, or "" when unset.
 *
 * Used by the skills/toolsets endpoints so the dashboard can manage a
 * profile other than the one the server process runs under. */
export function profileQuery(profile?: string): string {
  return profile ? `?profile=${encodeURIComponent(profile)}` : "";
}

export function appendProfileParam(url: string, profile?: string): string {
  if (!profile || url.includes("profile=")) return url;
  return `${url}${url.includes("?") ? "&" : "?"}profile=${encodeURIComponent(profile)}`;
}

export function appendQueryParam(url: string, key: string, value?: string): string {
  if (!value) return url;
  return `${url}${url.includes("?") ? "&" : "?"}${key}=${encodeURIComponent(value)}`;
}
