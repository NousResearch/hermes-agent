import type { CompletionItem } from '../app/interfaces.js'

import { applyCompletion, looksLikeSlashCommand } from './slash.js'

/**
 * Inline ghost text for the composer — the TUI half of the CLI's
 * `SlashCommandAutoSuggest` (hermes_cli/commands_completion.py).
 *
 * The CLI builds its suggestion by re-walking the command registry on every
 * keystroke. The TUI can't: the registry lives in the gateway process. But the
 * dropdown ALREADY asks the gateway for exactly that registry slice
 * (`complete.slash` / `complete.path`, the same `SlashCommandCompleter` the CLI
 * completes from), so the ghost is derived from the rows that request returned
 * instead of costing a second round trip per keystroke. Two consequences worth
 * keeping:
 *
 *   - The ghost can never disagree with the menu underneath it. It is one of
 *     the visible rows, by construction.
 *   - Rows lag the input by one debounce (60ms). That is survivable because the
 *     ghost is recomputed against the CURRENT text: a stale row that still
 *     extends what is typed keeps ghosting through the gap (type `/he` → `/hel`
 *     and "lp" simply shrinks to "p"), and one that no longer extends it is
 *     dropped by the prefix test rather than shown wrong.
 *
 * Selection rule matches the CLI's: the SHORTEST completion wins, so `/he`
 * ghosts `lp` (→ `/help`) rather than `artbeat` (→ `/heartbeat`).
 */

/** Ghost text derived from the completion rows fetched for this input, or ''. */
export function completionGhost(value: string, completions: CompletionItem[], compReplace: number): string {
  // A replace point past the end of the text belongs to an older, longer input
  // (the rows are one debounce behind). Applying it would append the row to the
  // whole value and ghost nonsense: `/he` + `add` → `/headd`.
  if (compReplace < 0 || compReplace > value.length) {
    return ''
  }

  let best = ''

  for (const item of completions) {
    if (!item.text) {
      continue
    }

    // Reuse the menu's own replace semantics so ghost text and Tab-accept can
    // never diverge on the leading-slash rule.
    const next = applyCompletion(value, item.text, compReplace)

    if (!next.startsWith(value)) {
      continue
    }

    const remainder = next.slice(value.length)

    // An exact match completes to itself plus the trailing space the gateway
    // appends to keep the menu open. There is nothing to ghost there.
    if (!remainder.trim()) {
      continue
    }

    if (!best || remainder.length < best.length) {
      best = remainder
    }
  }

  return best
}

/** Ghost text from the most recent matching history entry, or ''. */
export function historyGhost(value: string, history: readonly string[]): string {
  if (!value.trim() || value.includes('\n')) {
    return ''
  }

  for (let i = history.length - 1; i >= 0; i--) {
    // Multi-line recalls ghost their FIRST line only: the composer is one
    // logical line here, and a `\n` in the ghost would reflow the input box.
    const entry = (history[i] ?? '').split('\n')[0] ?? ''

    if (entry.length > value.length && entry.startsWith(value)) {
      return entry.slice(value.length)
    }
  }

  return ''
}

/** True while the user is still typing the command NAME (`/he`, not `/help x`). */
const typingCommandName = (value: string) => looksLikeSlashCommand(value) && !/\s/.test(value)

export interface InlineSuggestArgs {
  compReplace: number
  completions: CompletionItem[]
  history: readonly string[]
  value: string
}

/**
 * The single ghost string to draw after the cursor, or '' for none.
 *
 * Order mirrors the CLI: completions first, history second — except while the
 * command name itself is being typed, where a history hit would fight the
 * command being completed (`/c` must ghost `lear`, never the `/cron add ...`
 * line from yesterday). The CLI returns `None` in exactly that state.
 */
export function inlineSuggestion({ compReplace, completions, history, value }: InlineSuggestArgs): string {
  if (!value || value.includes('\n')) {
    return ''
  }

  const fromCompletions = completionGhost(value, completions, compReplace)

  if (fromCompletions) {
    return fromCompletions
  }

  return typingCommandName(value) ? '' : historyGhost(value, history)
}
