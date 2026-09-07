/** Description-aware fuzzy scoring for the slash-command menu.
 *
 *  Ported from superagent-ai/grok-cli `src/ui/slash-menu.ts`: candidates are
 *  scored in tiers — exact match on id/label/alias (0), prefix (1), substring
 *  (2) — and the DESCRIPTION text is tokenized and matched at a +3 offset
 *  (exact word 3, word prefix 4, word substring 5). Typing `/summary` thus
 *  surfaces a command whose description mentions summaries even though no
 *  command name starts with it. Lower score wins; `Infinity` means no match.
 */

export interface SlashScoreItem {
  aliases?: string[]
  description?: string
  id: string
  label?: string
}

/** Lowercase the value and return it alongside its alphanumeric word tokens. */
export function tokenizeSearchText(value: string): string[] {
  const normalized = value.toLowerCase()

  return [normalized, ...normalized.split(/[^a-z0-9]+/).filter(Boolean)]
}

/** Trim, drop leading slashes, lowercase — `/Model ` and `model` score alike. */
export function normalizeSlashSearchQuery(query: string): string {
  return query.trim().replace(/^\/+/, '').toLowerCase()
}

function scoreFields(fields: string[], query: string, offset: number): number {
  for (const field of fields) {
    if (field === query || `/${field}` === query) {
      return offset
    }
  }

  for (const field of fields) {
    if (field.startsWith(query) || `/${field}`.startsWith(query)) {
      return offset + 1
    }
  }

  for (const field of fields) {
    if (field.includes(query)) {
      return offset + 2
    }
  }

  return Number.POSITIVE_INFINITY
}

/** Score one item against a normalized query. Lower is better; Infinity = no match. */
export function scoreSlashMenuItem(item: SlashScoreItem, query: string): number {
  const commandFields = [item.id, item.label ?? '', ...(item.aliases ?? [])].filter(Boolean).flatMap(tokenizeSearchText)

  const descriptionFields = tokenizeSearchText(item.description ?? '')

  return Math.min(scoreFields(commandFields, query, 0), scoreFields(descriptionFields, query, 3))
}

/** Filter and stable-sort `items` by score (then original order). An empty
 *  query returns the list untouched so browsing keeps the caller's order. */
export function rankSlashItems<T>(items: T[], query: string, toScoreItem: (item: T) => SlashScoreItem): T[] {
  const normalized = normalizeSlashSearchQuery(query)

  if (!normalized) {
    return items
  }

  return items
    .map((item, index) => ({ index, item, score: scoreSlashMenuItem(toScoreItem(item), normalized) }))
    .filter(entry => entry.score !== Number.POSITIVE_INFINITY)
    .sort((a, b) => a.score - b.score || a.index - b.index)
    .map(entry => entry.item)
}
