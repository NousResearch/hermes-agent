import { stringWidth } from '@hermes/ink'

import { shortCwd, shortProject } from './paths.js'

export type WorkspaceHudField = 'git_branch' | 'git_root' | 'git_state' | 'git_sync' | 'github' | 'pr' | 'project'

export type WorkspaceHudTone = 'accent' | 'error' | 'good' | 'muted' | 'warn'

export interface GithubRepository {
  fullName: string
  owner: string
  repo: string
}

export interface PullRequestSummary {
  number: number
  state: string
  title: string
}

export interface UpstreamCounts {
  ahead: number
  behind: number
}

export interface GitWorkspaceSnapshot {
  branch: null | string
  dirty: boolean | null
  gitRoot: null | string
  github: GithubRepository | null
  pullRequest: PullRequestSummary | null
  upstream: UpstreamCounts | null
}

export interface WorkspaceHudSnapshot extends GitWorkspaceSnapshot {
  projectName: null | string
}

export interface WorkspaceHudPart {
  field: WorkspaceHudField
  text: string
  tone: WorkspaceHudTone
}

export const EMPTY_GIT_WORKSPACE: GitWorkspaceSnapshot = {
  branch: null,
  dirty: null,
  gitRoot: null,
  github: null,
  pullRequest: null,
  upstream: null
}

const URL_OR_REMOTE = /(?:https?|ssh):\/\/[^\s]+|git@github\.com:[^\s]+/gi
const SECRET_ASSIGNMENT = /\b(?:token|password|passwd|secret|api[_-]?key|authorization)\s*[:=]\s*[^\s]+/gi
const GITHUB_NAME = /^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$/
const HUD_PREFIX = '└ '
const HUD_SEPARATOR = ' · '

const isRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value)

const recordOf = (value: unknown): Record<string, unknown> | null => (isRecord(value) ? value : null)

const cleanText = (value: string) => {
  const withoutControls = [...value]
    .map(char => {
      const code = char.codePointAt(0) ?? 0

      return code <= 0x1f || code === 0x7f ? ' ' : char
    })
    .join('')

  return withoutControls.replace(/\s+/g, ' ').trim()
}

export function truncateDisplay(value: string, maxWidth: number): string {
  const width = Math.floor(maxWidth)

  if (width < 1) {
    return ''
  }

  if (stringWidth(value) <= width) {
    return value
  }

  if (width === 1) {
    return '…'
  }

  let prefix = ''

  for (const char of [...value]) {
    if (stringWidth(`${prefix}${char}…`) > width) {
      break
    }

    prefix += char
  }

  return `${prefix}…`
}

const boundedText = (value: string, maxWidth: number) => truncateDisplay(cleanText(value), maxWidth)

const safeDisplayText = (value: string, maxWidth: number) =>
  boundedText(value.replace(URL_OR_REMOTE, '[url]').replace(SECRET_ASSIGNMENT, '[redacted]'), maxWidth)

const githubRepository = (owner: string, repo: string): GithubRepository | null => {
  const normalizedOwner = owner.trim()
  const normalizedRepo = repo.trim().replace(/\.git$/i, '')

  if (!GITHUB_NAME.test(normalizedOwner) || !GITHUB_NAME.test(normalizedRepo)) {
    return null
  }

  return { fullName: `${normalizedOwner}/${normalizedRepo}`, owner: normalizedOwner, repo: normalizedRepo }
}

const repositoryFromPath = (path: string): GithubRepository | null => {
  let raw = path.trim().replace(/^\/+|\/+$/g, '')

  if (raw.toLowerCase().endsWith('.git')) {
    raw = raw.slice(0, -4).replace(/\/+$/, '')
  }

  let decoded = raw

  try {
    decoded = decodeURIComponent(raw)
  } catch {
    return null
  }

  const parts = decoded.split('/')

  return parts.length === 2 ? githubRepository(parts[0] ?? '', parts[1] ?? '') : null
}

/** Return only a safe owner/repo pair from a GitHub remote URL. */
export function normalizeGithubOrigin(raw: null | string | undefined): GithubRepository | null {
  const value = raw?.trim() ?? ''

  if (!value) {
    return null
  }

  const scp = value.match(/^git@github\.com:(.+)$/i)

  if (scp) {
    return repositoryFromPath(scp[1] ?? '')
  }

  const candidate = !value.includes('://') && /^(?:www\.)?github\.com\//i.test(value) ? `https://${value}` : value

  try {
    const url = new URL(candidate)

    if (!['github.com', 'www.github.com'].includes(url.hostname.toLowerCase())) {
      return null
    }

    return repositoryFromPath(url.pathname)
  } catch {
    return null
  }
}

const safePrTitle = (value: string, maxWidth = 64) => safeDisplayText(value, maxWidth)

const positiveInteger = (value: unknown): number | null => {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 1) {
    return null
  }

  return value
}

const nonNegativeInteger = (value: unknown): number | null => {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0) {
    return null
  }

  return value
}

export function parsePullRequest(value: unknown): PullRequestSummary | null {
  const record = recordOf(value)

  if (!record) {
    return null
  }

  const number = positiveInteger(record.number)

  if (number === null) {
    return null
  }

  const title = typeof record.title === 'string' ? safePrTitle(record.title) : ''
  const rawState = typeof record.state === 'string' ? cleanText(record.state).toLowerCase() : ''
  const state = rawState === 'open' || rawState === 'closed' || rawState === 'merged' ? rawState : ''

  return { number, state, title }
}

export function parseWorkspaceInfo(value: unknown): GitWorkspaceSnapshot {
  const record = recordOf(value)

  if (!record) {
    return { ...EMPTY_GIT_WORKSPACE }
  }

  const rawGithub = record.github
  const githubRecord = recordOf(rawGithub)

  const github = githubRecord
    ? githubRepository(
        typeof githubRecord.owner === 'string' ? githubRecord.owner : '',
        typeof githubRecord.repo === 'string' ? githubRecord.repo : ''
      )
    : null

  const upstreamRecord = recordOf(record.upstream)
  const ahead = upstreamRecord ? nonNegativeInteger(upstreamRecord.ahead) : null
  const behind = upstreamRecord ? nonNegativeInteger(upstreamRecord.behind) : null

  return {
    branch: typeof record.branch === 'string' && record.branch.trim() ? boundedText(record.branch, 80) : null,
    dirty: typeof record.dirty === 'boolean' ? record.dirty : null,
    gitRoot: typeof record.git_root === 'string' && record.git_root.trim() ? record.git_root.trim() : null,
    github,
    pullRequest: parsePullRequest(record.pull_request),
    upstream: ahead !== null && behind !== null ? { ahead, behind } : null
  }
}

const fieldAliases: Record<WorkspaceHudField, readonly string[]> = {
  git_branch: ['branch', 'git'],
  git_root: ['root', 'git'],
  git_state: ['state', 'git'],
  git_sync: ['sync', 'git'],
  github: ['remote'],
  pr: ['pull_request'],
  project: []
}

const fieldVisible = (fields: null | ReadonlySet<string>, field: WorkspaceHudField) =>
  fields === null ||
  fields.has('workspace') ||
  fields.has(field) ||
  fieldAliases[field].some(alias => fields.has(alias))

const compactCount = (value: number) => {
  if (value >= 1_000_000) {
    return `${Math.floor(value / 1_000_000)}m+`
  }

  if (value >= 1_000) {
    return `${Math.floor(value / 1_000)}k+`
  }

  return String(value)
}

export function workspaceHudParts(
  snapshot: WorkspaceHudSnapshot,
  statusBarFields: null | ReadonlySet<string> = null
): WorkspaceHudPart[] {
  const parts: WorkspaceHudPart[] = []

  const add = (field: WorkspaceHudField, text: string, tone: WorkspaceHudTone) => {
    if (fieldVisible(statusBarFields, field) && text) {
      parts.push({ field, text, tone })
    }
  }

  const project = shortProject(snapshot.projectName ?? '', 20)

  if (project) {
    add('project', `◆ ${safeDisplayText(project, 22)}`, 'accent')
  }

  if (snapshot.gitRoot) {
    add('git_root', `⌂ ${safeDisplayText(shortCwd(snapshot.gitRoot, 36), 38)}`, 'muted')
  }

  if (snapshot.branch) {
    add('git_branch', `⎇ ${safeDisplayText(snapshot.branch, 22)}`, 'accent')
  }

  if (snapshot.dirty === true) {
    add('git_state', '● dirty', 'error')
  } else if (snapshot.dirty === false) {
    add('git_state', '✓ clean', 'good')
  }

  if (snapshot.github) {
    add('github', `GH ${boundedText(snapshot.github.fullName, 42)}`, 'muted')
  }

  if (snapshot.upstream) {
    add(
      'git_sync',
      `↑${compactCount(snapshot.upstream.ahead)} ↓${compactCount(snapshot.upstream.behind)}`,
      snapshot.upstream.ahead || snapshot.upstream.behind ? 'warn' : 'muted'
    )
  }

  if (snapshot.pullRequest) {
    const state = snapshot.pullRequest.state ? ` ${boundedText(snapshot.pullRequest.state, 12)}` : ''
    const title = snapshot.pullRequest.title ? ` ${snapshot.pullRequest.title}` : ''
    add('pr', `PR #${compactCount(snapshot.pullRequest.number)}${state}${title}`, 'accent')
  }

  return parts
}

const partMinimumWidth: Record<WorkspaceHudField, number> = {
  git_branch: 7,
  git_root: 8,
  git_state: 8,
  git_sync: 7,
  github: 10,
  pr: 12,
  project: 8
}

const lowPriorityFields: readonly WorkspaceHudField[] = ['pr', 'git_sync', 'github', 'git_state', 'git_root']

const lineWidth = (parts: readonly WorkspaceHudPart[], prefix = HUD_PREFIX) =>
  stringWidth(prefix) +
  parts.reduce((total, part, index) => total + stringWidth(part.text) + (index ? stringWidth(HUD_SEPARATOR) : 0), 0)

/** Fit by dropping low-priority details first, then shrinking bounded labels. */
export function fitWorkspaceHudParts(parts: readonly WorkspaceHudPart[], cols: number): WorkspaceHudPart[] {
  const width = Math.max(1, Math.floor(cols))
  let fitted = parts.map(part => ({ ...part }))
  const prefix = width >= stringWidth(HUD_PREFIX) ? HUD_PREFIX : ''

  // A PR number/state is useful even when its title is not. Compact that
  // detail before removing the whole PR segment, then continue dropping the
  // low-priority segments if the summary itself still cannot fit.
  if (lineWidth(fitted, prefix) > width) {
    fitted = fitted.map(part => {
      if (part.field !== 'pr') {
        return part
      }

      const compact = part.text.match(/^PR #[^\s]+(?:\s+[^\s]+)?/i)?.[0]

      return compact && compact.length < part.text.length ? { ...part, text: compact } : part
    })
  }

  for (const field of lowPriorityFields) {
    if (lineWidth(fitted, prefix) <= width) {
      break
    }

    fitted = fitted.filter(part => part.field !== field)
  }

  const separatorWidth = stringWidth(HUD_SEPARATOR)
  const textBudget = Math.max(0, width - stringWidth(prefix) - Math.max(0, fitted.length - 1) * separatorWidth)
  const minimums = fitted.map(part => Math.min(partMinimumWidth[part.field], stringWidth(part.text)))

  while (fitted.length > 1 && minimums.reduce((sum, value) => sum + value, 0) > textBudget) {
    fitted.pop()
    minimums.pop()
  }

  // The prefix consumes part of the row even when one identity segment
  // remains. Cap that segment to the actual text budget so the colored Ink
  // renderer cannot overflow on the smallest terminals.
  if (fitted.length > 0 && textBudget < 1) {
    return []
  }

  if (fitted.length === 1) {
    minimums[0] = Math.min(minimums[0] ?? 1, textBudget)
  }

  let remaining = textBudget

  for (let i = 0; i < fitted.length; i += 1) {
    const part = fitted[i]

    if (!part) {
      continue
    }

    const laterMinimum = minimums.slice(i + 1).reduce((sum, value) => sum + value, 0)
    const target = Math.max(minimums[i] ?? 1, Math.min(stringWidth(part.text), remaining - laterMinimum))
    part.text = truncateDisplay(part.text, target)
    remaining = Math.max(0, remaining - stringWidth(part.text))
  }

  return fitted.filter(part => part.text)
}

export function formatWorkspaceHud(
  snapshot: WorkspaceHudSnapshot,
  cols: number,
  statusBarFields: null | ReadonlySet<string> = null
): string {
  const width = Math.max(1, Math.floor(cols))
  const parts = fitWorkspaceHudParts(workspaceHudParts(snapshot, statusBarFields), width)

  if (!parts.length) {
    return ''
  }

  const prefix = width >= stringWidth(HUD_PREFIX) ? HUD_PREFIX : ''

  return truncateDisplay(`${prefix}${parts.map(part => part.text).join(HUD_SEPARATOR)}`, width)
}
