/** Appended by TUI pickers; converted to the backend's `--session` flag before `config.set`. */
export const TUI_SESSION_MODEL_FLAG = '--tui-session'

export const sessionScopedModelArg = (value: string) => {
  const parts = value.trim().split(/\s+/).filter(Boolean)
  const kept = parts.filter(part => part !== TUI_SESSION_MODEL_FLAG && part !== '--global' && part !== '--session')

  return kept.length ? `${kept.join(' ')} --session` : ''
}

export const looksLikeSlashCommand = (text: string) => /^\/[^\s/]*(?:\s|$)/.test(text)

// A `/` means two different things depending on where it sits:
//
//  - At position 0 it's a COMMAND invocation the TUI executes
//    (`looksLikeSlashCommand`, and the backend's dispatch, are both
//    `^`-anchored). Completion stays live past the command name so arg
//    completion works (`/cron ad` → `add`).
//  - After whitespace it's an inline SKILL reference the user is dropping into
//    prose ("clean this up with /clean"). The text submits as an ordinary
//    message, so there are no args to complete — the token ends at the caret.
//
// The inline shape is what makes skills reachable anywhere in a prompt. The
// trailing `$` matters for both: completion runs against the text the user has
// typed so far, so the match has to end where they're typing.
//
// Requiring a single bare segment (no second `/`) keeps real paths out:
// `look at /usr/local/bin` and `check src/foo/bar` never match. A bare `/us` is
// genuinely ambiguous with an absolute path, and resolves as a skill reference
// — typing the next `/` flips it straight back to path completion.
const INLINE_SLASH_RE = /\s\/([a-zA-Z][\w-]*)?$/

/**
 * Locate an inline `/skill` reference at the end of `text`, or null when the
 * text isn't one. `start` is the index of the `/` itself, so a completion
 * replaces the typed token and leaves the prose in front of it untouched.
 */
export const inlineSlashTrigger = (text: string): { query: string; start: number } | null => {
  const match = INLINE_SLASH_RE.exec(text)

  if (!match) {
    return null
  }

  const query = match[1] ?? ''

  return { query, start: text.length - query.length - 1 }
}

// Only the separator between the command name and its argument is whitespace
// the parser owns. Everything after it is the user's text and survives
// verbatim: splitting the whole line on `\s+` and rejoining with a space
// flattened every pasted diff, log, or PR thread into one run-on line before
// the command ever saw it.
const SLASH_PARTS_RE = /^(\S*)\s*([\s\S]*)$/

export const parseSlashCommand = (cmd: string) => {
  const [, name = '', arg = ''] = SLASH_PARTS_RE.exec(cmd.slice(1)) ?? []

  return { arg, cmd, name: name.toLowerCase() }
}

/**
 * Apply a completion row to the current input, mirroring the editor's
 * replace semantics: replace from `compReplace` with the row text, dropping
 * the row's leading slash when the input already has one immediately before
 * the replace point (the gateway's slash completer returns bare command names
 * whose replace span begins after the leading `/`).
 *
 * Keyed off the character before `compReplace` rather than the start of the
 * input so a `/token` anywhere in the message behaves the same as one at
 * position 0 — an inline `run /cle` replaces from after its own slash.
 */
export const applyCompletion = (value: string, rowText: string, compReplace: number): string => {
  const text = value[compReplace - 1] === '/' && rowText.startsWith('/') ? rowText.slice(1) : rowText

  return value.slice(0, compReplace) + text
}

/**
 * Decide what Enter does when a completion is highlighted: returns the value
 * to set (accept the completion) or `null` to fall through to submit.
 *
 * Enter accepts a completion only when it changes the command/argument token.
 * A completion that merely appends trailing whitespace to an already-complete
 * command (e.g. `/exit` → `/exit `, the trailing space the gateway adds so the
 * classic CLI's prompt_toolkit dropdown stays open) must NOT swallow the Enter
 * — otherwise every slash command needs an extra keypress: type → Enter
 * completes the name → Enter adds the space → Enter finally submits. Treating a
 * whitespace-only delta as "already complete" collapses that back to the
 * expected one/two presses.
 */
export const completionToApplyOnSubmit = (
  value: string,
  rowText: string | undefined,
  compReplace: number
): string | null => {
  if (!rowText) {
    return null
  }

  const next = applyCompletion(value, rowText, compReplace)

  return next !== value && next.trimEnd() !== value.trimEnd() ? next : null
}
