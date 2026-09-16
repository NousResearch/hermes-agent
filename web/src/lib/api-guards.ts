/**
 * Runtime coercion for API response fields.
 *
 * `fetchJSON<T>` casts responses to `T` without validating them, so a
 * mis-shaped payload (an older backend, an error envelope, a proxy page
 * answering with HTML-as-JSON) flows into callers typed as if it were
 * correct. These guards let pages degrade gracefully — an empty list, a
 * zeroed count — instead of crashing on a bad response.
 */

/** Return `value` if it is an array, otherwise `[]`. */
export function asArray<T>(value: unknown): T[] {
  return Array.isArray(value) ? (value as T[]) : [];
}

/** Return `value` if it is a finite number, otherwise `fallback`. */
export function asCount(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

/**
 * Return `value` if it is a record of numeric counts, coercing
 * numeric-string values and dropping non-numeric ones; otherwise `{}`.
 */
export function asCountMap(value: unknown): Record<string, number> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return {};
  }
  const out: Record<string, number> = {};
  for (const [key, count] of Object.entries(value)) {
    if (typeof count === "number") {
      if (Number.isFinite(count)) out[key] = count;
    } else if (typeof count === "string" && count.trim() !== "") {
      // Numeric strings from loosely-typed backends — but note that
      // Number(null)/Number("")/Number(false) are all 0, so anything
      // non-numeric-string is dropped rather than coerced to a fake zero.
      const n = Number(count);
      if (Number.isFinite(n)) out[key] = n;
    }
  }
  return out;
}
