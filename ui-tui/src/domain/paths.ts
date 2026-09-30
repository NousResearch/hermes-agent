import type { GitStatus } from '../hooks/useGitBranch.js'

export interface ForgeUrl {
  // Base browse URL for the repository, with no trailing slash or `.git`.
  repo: string
  // Path segment this forge uses before a ref: GitHub `/tree/`, GitLab `/-/tree/`.
  tree: string
}

// Only forges whose browse-path shape we actually know are linked; anything
// else (self-hosted, bare paths, unknown hosts) stays plain text rather than
// guessing a URL that 404s.
const FORGE_TREE_PATHS: Record<string, string> = {
  'github.com': '/tree/',
  'gitlab.com': '/-/tree/'
}

/**
 * Resolve a git remote to its https browse URL.
 *
 * Mirrors `~/.config/zsh/bin/starship-git-url`, the resolver the zsh prompt
 * already uses, so the TUI status bar and the shell prompt agree on the link
 * target: scp-style (`git@host:owner/repo`), `ssh://`, `git://`, `https://`
 * and `http://` all collapse to the same canonical URL.
 */
export const forgeUrlFromRemote = (remote: string): ForgeUrl | null => {
  const url = remote.trim()

  if (!url) {
    return null
  }

  let host: string
  let path: string

  const scp = /^[^/@\s]+@([^:/\s]+):(.+)$/.exec(url)

  if (scp) {
    host = scp[1] ?? ''
    path = scp[2] ?? ''
  } else if (/^(?:ssh|git|https?):\/\//i.test(url)) {
    const rest = url.replace(/^[a-z+]+:\/\//i, '').replace(/^[^@/]*@/, '')
    const slash = rest.indexOf('/')

    if (slash === -1) {
      return null
    }

    host = rest.slice(0, slash).split(':')[0] ?? ''
    path = rest.slice(slash + 1)
  } else {
    return null
  }

  const repoPath = path.replace(/\.git$/i, '').replace(/^\/+|\/+$/g, '')
  const tree = FORGE_TREE_PATHS[host.toLowerCase()]

  if (!repoPath || !tree) {
    return null
  }

  return { repo: `https://${host.toLowerCase()}/${repoPath}`, tree }
}

/**
 * Which remote a branch's browse URL should point at.
 *
 * A branch pushed to a fork does not exist on `origin`, so linking it against
 * origin produces a 404. Git records the real publish target in
 * `branch.<name>.remote`; `.` means "local-only" (no remote), so that and an
 * absent value both fall back to origin.
 */
export const gitRemoteNameForBranch = (branchRemote: null | string | undefined): string => {
  const name = (branchRemote ?? '').trim()

  return !name || name === '.' ? 'origin' : name
}

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
  maxName = 28
): string => {
  const name = sessionName.trim()
  const shortName = name.length > maxName ? `${name.slice(0, maxName - 1)}…` : name

  const segments = [shortName, model, cwd].filter(Boolean)

  return segments.length ? `${marker} ${segments.join(' · ')}` : marker
}

// Fish-style contraction: keep only the final component in full and contract
// every parent to its first character (dot-dirs keep the dot), e.g.
// `/private/tmp/hermes-tui-vim` → `/p/t/hermes-tui-vim`.
export const formatStatusPath = (cwd: string, keep = 1, fishLen = 1): string => {
  const home = process.env.HOME ?? ''
  const path = home && cwd === home ? '~' : home && cwd.startsWith(`${home}/`) ? `~/${cwd.slice(home.length + 1)}` : cwd

  if (path === '~' || path === '/') {
    return path
  }

  const prefix = path.startsWith('~/') ? '~' : ''
  const parts = path.replace(/^~\//, '').replace(/^\//, '').split('/').filter(Boolean)

  if (parts.length <= keep) {
    return prefix ? `~/${parts.join('/')}` : `/${parts.join('/')}`
  }

  const front = parts.slice(0, parts.length - keep).map(part => {
    const width = part.startsWith('.') ? fishLen + 1 : fishLen

    return part.length > width ? part.slice(0, width) : part
  })

  const body = [...front, ...parts.slice(-keep)].join('/')

  return prefix ? `~/${body}` : `/${body}`
}

// Deterministic glyph order, matching the classic CLI renderer:
// `= conflicted`, `+ staged`, `! modified`, `? untracked`, `⇡ ahead`, `⇣ behind`.
export const formatGitStatus = (status: GitStatus | null): string => {
  if (!status?.branch) {
    return ''
  }

  const counters: [number, string][] = [
    [status.conflicted, '='],
    [status.staged, '+'],
    [status.modified, '!'],
    [status.untracked, '?'],
    [status.ahead, '⇡'],
    [status.behind, '⇣']
  ]

  return [` ${status.branch}`, ...counters.filter(([n]) => n > 0).map(([n, glyph]) => `${glyph}${n}`)].join(' ')
}
