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

/**
 * Compose the terminal titlebar string:
 *   `<marker> <session name> · <model> · <cwd> · Hermes <state>`
 *
 * The session name and cwd are each omitted when empty, and a long session
 * name is truncated. The marker is always glued to the first present segment
 * with a plain space (not a ` · ` separator). When no model is known yet the
 * caller should fall back to a plain brand string instead of calling this.
 *
 * The trailing `Hermes <state>` segment is for agent-aware terminals (Orca,
 * and any host that classifies agent panes by OSC title). Those hosts key on
 * the word `hermes` to recognise the pane and on plain English state words
 * (`working` / `waiting` / `ready`) to pick a status glyph — they do not
 * read our `⏳ ⚠ ✓` markers. Human-first segments stay in front so a narrow
 * tab bar still shows marker + session name; the machine-readable tail is
 * what gets clipped first.
 */

/** Human marker → the status word agent-aware terminals understand. */
export const AGENT_STATE_WORDS: Readonly<Record<string, string>> = {
  '⏳': 'working',
  '⚠': 'waiting',
  '✓': 'ready'
}

export const composeTabTitle = (
  marker: string,
  sessionName: string,
  model: string,
  cwd: string,
  maxName = 28
): string => {
  const name = sessionName.trim()
  const shortName = name.length > maxName ? `${name.slice(0, maxName - 1)}…` : name
  const stateWord = AGENT_STATE_WORDS[marker]
  const agentTail = stateWord ? `Hermes ${stateWord}` : ''

  const segments = [shortName, model, cwd, agentTail].filter(Boolean)

  return segments.length ? `${marker} ${segments.join(' · ')}` : marker
}
