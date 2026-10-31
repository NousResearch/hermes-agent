/**
 * Reverse-search ranking for Ctrl+R: newest first, one row per distinct entry, every
 * whitespace-separated query term must appear (case-insensitive, any order).
 */
export function searchHistory(entries: readonly string[], query: string): string[] {
  const terms = query.toLowerCase().split(/\s+/).filter(Boolean)
  const seen = new Set<string>()
  const out: string[] = []

  for (let i = entries.length - 1; i >= 0; i--) {
    const entry = entries[i]!

    if (seen.has(entry)) {
      continue
    }

    seen.add(entry)
    const haystack = entry.toLowerCase()

    if (terms.every(term => haystack.includes(term))) {
      out.push(entry)
    }
  }

  return out
}

/** One display line for a possibly multi-line entry. */
export const historyPreview = (entry: string) => entry.replace(/\s*\n\s*/g, ' ⏎ ')
