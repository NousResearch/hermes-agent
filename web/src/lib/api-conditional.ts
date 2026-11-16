/**
 * Conditional-GET plumbing for the dashboard's list reads (#C3, t_406dbcb3).
 *
 * A reconnect refreshes lists even when nothing moved. The server stamps a
 * content ETag (see `hermes_cli/web_routers/sessions.py`); this module keeps
 * the validator + payload beside each URL in a JS-side Map — deliberately not
 * the HTTP cache, so the behavior cannot drift with browser cache policy — and
 * re-serves the cached payload on a 304: zero JSON on the wire, zero re-parse
 * and zero re-render churn for the caller.
 *
 * `fetchJSON` is injected rather than imported: the request pipeline (session
 * token, management-profile scoping, same-origin policy, 401 handling) stays
 * in one place, and this module keeps no knowledge of it.
 */

/** Sentinel a conditional fetch resolves with when the server answered 304. */
export const HTTP_NOT_MODIFIED = Symbol("HTTP 304 Not Modified");

/** The slice of `fetchJSON` this module needs, structurally. */
export type ConditionalFetcher = <T>(
  url: string,
  options: {
    ifNoneMatch?: string;
    onEtag: (etag: string | null) => void;
  },
) => Promise<T>;

export interface ConditionalGet {
  /** GET `url`, revalidating with the stored ETag when there is one. */
  get: <T>(url: string) => Promise<T>;
  /** Drop every stored validator (tests). */
  reset: () => void;
}

/** Bounded because a long-lived tab can cycle query params; oldest entry first. */
const MAX_ENTRIES = 32;

export function createConditionalGet(fetcher: ConditionalFetcher): ConditionalGet {
  const entries = new Map<string, { etag: string; value: unknown }>();

  return {
    async get<T>(url: string): Promise<T> {
      const cached = entries.get(url) as { etag: string; value: T } | undefined;
      let etag: string | null = null;
      const result = await fetcher<T | typeof HTTP_NOT_MODIFIED>(url, {
        ...(cached ? { ifNoneMatch: cached.etag } : {}),
        onEtag: (value) => {
          etag = value;
        },
      });

      if (result === HTTP_NOT_MODIFIED && cached) {
        return cached.value;
      }

      if (etag) {
        entries.delete(url); // re-insert refreshes eviction recency
        entries.set(url, { etag, value: result as T });
        while (entries.size > MAX_ENTRIES) {
          const oldest = entries.keys().next().value;

          if (oldest === undefined) break;
          entries.delete(oldest);
        }
      }

      return result as T;
    },
    reset() {
      entries.clear();
    },
  };
}
