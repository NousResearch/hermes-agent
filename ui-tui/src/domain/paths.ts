export const shortCwd = (cwd: string, max = 28) => {
  const h = process.env.HOME
  const p = h && cwd.startsWith(h) ? `~${cwd.slice(h.length)}` : cwd

  return p.length <= max ? p : `…${p.slice(-(max - 1))}`
}

export const fmtCwdBranch = (cwd: string, branch: null | string, max = 40) => {
  if (!branch) {
    return shortCwd(cwd, max)
  }

  const tag = ` (${branch.length > 16 ? `…${branch.slice(-15)}` : branch})`

  return `${shortCwd(cwd, Math.max(8, max - tag.length))}${tag}`
}

export const shortProject = (projectName: string, max = 18) => {
  const name = projectName.trim()

  return name.length <= max ? name : `${name.slice(0, Math.max(1, max - 1))}…`
}

// Status-bar workspace label: the terminal has no hover tooltip, so the project
// name is shown INLINE alongside the cwd/branch (`<project> · ~/cwd (branch)`).
// Falls back to the plain cwd/branch label when the session sits in no named
// project, and when space is tight the project name wins (it's the identity the
// user recognizes) with the cwd/branch dropped.
export const fmtProjectCwdBranch = (cwd: string, branch: null | string, projectName?: null | string, max = 40) => {
  const project = shortProject(projectName || '')

  if (!project) {
    return fmtCwdBranch(cwd, branch, max)
  }

  const separator = ' · '
  const remaining = max - project.length - separator.length

  if (remaining < 8) {
    return shortProject(project, max)
  }

  return `${project}${separator}${fmtCwdBranch(cwd, branch, remaining)}`
}

/** Default cap on the session-name segment of a composed title. */
export const TITLE_MAX_NAME = 28

/** Default cap on the cwd segment of a composed title. */
export const TITLE_MAX_CWD = 24

export const shortSessionName = (sessionName: string, max = TITLE_MAX_NAME): string => {
  const name = sessionName.trim()

  return name.length > max ? `${name.slice(0, max - 1)}…` : name
}

/**
 * Compose the terminal titlebar string:
 *   `<marker> <session name> · <model> · <cwd>`
 *
 * The session name and cwd are each omitted when empty, and a long session
 * name is truncated. The marker is always glued to the first present segment
 * with a plain space (not a ` · ` separator). When no model is known yet the
 * caller should fall back to a plain brand string instead of calling this.
 */
export const composeTabTitle = (
  marker: string,
  sessionName: string,
  model: string,
  cwd: string,
  maxName = TITLE_MAX_NAME
): string => {
  const shortName = shortSessionName(sessionName, maxName)

  const segments = [shortName, model, cwd].filter(Boolean)

  return segments.length ? `${marker} ${segments.join(' · ')}` : marker
}

/** Values a title template can interpolate. `*_full` variants skip the
 *  length caps and prefix-stripping that the short forms apply. */
export interface TitleTokens {
  cwd: string
  cwdFull: string
  marker: string
  model: string
  modelFull: string
  session: string
  sessionFull: string
}

// Recognized placeholders. Unknown `{tokens}` are left VERBATIM rather than
// blanked: a typo stays visible in the title bar instead of silently
// vanishing, which is the cheaper failure to diagnose.
const TITLE_TOKEN_PATTERN = /\{(cwd|cwd_full|marker|model|model_full|session|session_full)\}/g

/**
 * Render a user-supplied title template (`display.tab_title_template` /
 * `display.window_title_template`).
 *
 * Empty/blank template ⇒ `null`, meaning "caller keeps its built-in default".
 *
 * Segment separators live in the template, so a token that resolves to an empty
 * string would leave a dangling ` · `. Cleanup is driven by WHICH tokens came
 * back empty rather than by scanning the finished string: a global
 * whitespace/`·` collapse would also rewrite the user's literal text and the
 * deliberately verbatim `*_full` values (a cwd or session name holding two
 * spaces must survive intact). Each empty substitution is marked with a
 * sentinel that cannot occur in user input, the separator on one side of it is
 * dropped, then the sentinel is removed.
 */
export const renderTitleTemplate = (template: string, tokens: TitleTokens): null | string => {
  if (!template.trim()) {
    return null
  }

  const substituted = template.replace(TITLE_TOKEN_PATTERN, (match, name: string) => {
    const value = titleTokenValue(name, tokens)

    if (value === undefined) {
      return match
    }

    return value === '' ? EMPTY_TOKEN_SENTINEL : value
  })

  if (!substituted.includes(EMPTY_TOKEN_SENTINEL)) {
    return substituted.trim()
  }

  return (
    substituted
      // One alternation, not two passes: the match CONSUMES the hole, so a
      // single hole can never have the separator eaten on both sides. The
      // following separator is preferred; the leading one covers a trailing
      // hole. `[^\S\n]` keeps the class to horizontal space only.
      .replace(SENTINEL_ADJACENT_SEP, EMPTY_TOKEN_SENTINEL)
      // A hole between plain spaces (`{marker} {session}` with no marker) must
      // not leave a double space behind.
      .replace(SENTINEL_BETWEEN_SPACES, ' ')
      .replaceAll(EMPTY_TOKEN_SENTINEL, '')
      .trim()
  )
}

// A private-use code point: it cannot appear in a YAML scalar, a session title
// or a path, and unlike U+0000 it is not a control char (no-control-regex).
const EMPTY_TOKEN_SENTINEL = '\uE000'

const SENTINEL_ADJACENT_SEP = /[^\S\n]*·\s*\uE000|\uE000\s*·[^\S\n]*/g
const SENTINEL_BETWEEN_SPACES = /[^\S\n]\uE000[^\S\n]/g

/** Token value, or `undefined` for a name the pattern does not cover. */
const titleTokenValue = (name: string, tokens: TitleTokens): string | undefined => {
  switch (name) {
    case 'cwd':
      return tokens.cwd

    case 'cwd_full':
      return tokens.cwdFull

    case 'marker':
      return tokens.marker

    case 'model':
      return tokens.model

    case 'model_full':
      return tokens.modelFull

    case 'session':
      return tokens.session

    case 'session_full':
      return tokens.sessionFull

    default:
      return undefined
  }
}

/**
 * Resolve what `useTerminalTitle` should receive: a `{tab, window}` pair, or the
 * plain brand string when there is nothing session-specific to show yet.
 *
 * A CONFIGURED template wins even before the model is known — `{session_full}`
 * or literal text renders fine with no model, and gating on the model would
 * ignore the user's config for the whole startup window. The built-in
 * composition still needs a model, else it degrades to a bare marker.
 */
export const resolveTerminalTitle = (
  templates: { tab: string; window: string },
  tokens: TitleTokens,
  sessionName: string
): string | { tab: string; window: string } => {
  const tab = renderTitleTemplate(templates.tab, tokens)
  const window = renderTitleTemplate(templates.window, tokens)

  if (!tab && !window && !tokens.model) {
    return 'Hermes'
  }

  return {
    tab: tab || composeTabTitle(tokens.marker, sessionName, '', ''),
    window: window || composeTabTitle(tokens.marker, sessionName, tokens.model, tokens.cwd)
  }
}
