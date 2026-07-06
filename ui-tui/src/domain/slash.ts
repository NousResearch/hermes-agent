/** Appended to `/model` args from the TUI picker for session scope; stripped in `session` slash before `config.set`. */
export const TUI_SESSION_MODEL_FLAG = '--tui-session'

export const looksLikeSlashCommand = (text: string) => /^\/[^\s/]*(?:\s|$)/.test(text)

export const parseSlashCommand = (cmd: string) => {
  const [name = '', ...rest] = cmd.slice(1).split(/\s+/)

  return { arg: rest.join(' '), cmd, name: name.toLowerCase() }
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
