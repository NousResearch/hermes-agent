/** Appended by TUI pickers; converted to the backend's `--session` flag before `config.set`. */
export const TUI_SESSION_MODEL_FLAG = '--tui-session'

export const sessionScopedModelArg = (value: string) => {
  const parts = value.trim().split(/\s+/).filter(Boolean)
  const kept = parts.filter(part => part !== TUI_SESSION_MODEL_FLAG && part !== '--global' && part !== '--session')

  return kept.length ? `${kept.join(' ')} --session` : ''
}

export const looksLikeSlashCommand = (text: string) => /^\/[^\s/]*(?:\s|$)/.test(text)

export const parseSlashCommand = (cmd: string) => {
  // Peel only the command name off the front and keep the remainder verbatim so
  // a multi-line argument (e.g. a pasted multi-line `/goal`) keeps its newlines
  // instead of being flattened into spaces by a `split`/`join` (#41323).
  const match = cmd.slice(1).match(/^(\S+)\s*([\s\S]*)$/)

  return { arg: (match?.[2] ?? '').trim(), cmd, name: (match?.[1] ?? '').toLowerCase() }
}

/**
 * Apply a completion row to the current input, mirroring the editor's
 * replace semantics: replace from `compReplace` with the row text, dropping
 * the leading slash when both the input and the row carry one (the gateway's
 * slash completer returns bare command names whose replace span begins after
 * the leading `/`).
 */
export const applyCompletion = (value: string, rowText: string, compReplace: number): string => {
  const text = value.startsWith('/') && rowText.startsWith('/') ? rowText.slice(1) : rowText

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
