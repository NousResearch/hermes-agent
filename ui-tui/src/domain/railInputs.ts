import { access, readdir, readFile, stat } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'

/** Repo-owned facts rendered by the per-project context rail. */
export interface RailInputs {
  checks: string
  decisions: string[]
  evidence: string
  flow: string
  mtimeMs: number
  product: string
  projectRoot?: string
  sourceFile: string
}

export type ParsedRailInputs = Omit<RailInputs, 'mtimeMs' | 'projectRoot' | 'sourceFile'>

// Context files can contain the full project guide; keep parsing bounded without
// hiding normal guides that are larger than a single screen of instructions.
const MAX_RAIL_INPUT_BYTES = 128 * 1024
const SECTION_RE = /^##\s+Context rail inputs\s*$/i
const NEXT_SECTION_RE = /^##\s/
const TOP_LEVEL_RE = /^-\s+([A-Za-z][A-Za-z0-9_-]*):\s?(.*)$/
const DECISION_ITEM_RE = /^\s{2,}-\s+(.+)$/
const KNOWN_KEYS = new Set(['product', 'flow', 'checks', 'evidence', 'decisions'])
const REQUIRED_KEYS = ['product', 'flow', 'checks', 'evidence', 'decisions'] as const
const HERMES_CONTEXT_NAMES = ['.hermes.md', 'HERMES.md'] as const
const AGENT_CONTEXT_NAMES = ['AGENTS.override.md', 'AGENTS.md', 'agents.md'] as const

const isFile = async (path: string): Promise<boolean> => {
  try {
    return (await stat(path)).isFile()
  } catch {
    return false
  }
}

const findGitRoot = async (startDir: string): Promise<string | null> => {
  let current = resolve(startDir)

  for (;;) {
    try {
      await access(join(current, '.git'))

      return current
    } catch {
      const parent = resolve(current, '..')

      if (parent === current) {
        return null
      }

      current = parent
    }
  }
}

const ancestorsToRoot = (startDir: string, root: string | null): string[] => {
  const start = resolve(startDir)
  const dirs = [start]
  let current = start

  while (root && current !== root) {
    const parent = resolve(current, '..')

    if (parent === current) {
      break
    }

    dirs.push(parent)
    current = parent
  }

  return dirs
}

const firstFile = async (dir: string, names: readonly string[]): Promise<string | null> => {
  for (const name of names) {
    const candidate = join(dir, name)

    if (await isFile(candidate)) {
      return candidate
    }
  }

  return null
}

const cursorRuleFile = async (dir: string): Promise<string | null> => {
  try {
    const names = (await readdir(join(dir, '.cursor', 'rules'))).filter(name => name.endsWith('.mdc')).sort()

    for (const name of names) {
      const candidate = join(dir, '.cursor', 'rules', name)

      if (await isFile(candidate)) {
        return candidate
      }
    }
  } catch {
    // No Cursor rules directory.
  }

  return null
}

const contextCandidates = async (startDir: string): Promise<string[]> => {
  const start = resolve(startDir)
  const gitRoot = await findGitRoot(start)
  let hermesHit: string | null = null

  for (const dir of ancestorsToRoot(start, gitRoot)) {
    hermesHit = await firstFile(dir, HERMES_CONTEXT_NAMES)

    if (hermesHit) {
      break
    }
  }

  if (hermesHit) {
    return [hermesHit]
  }

  const agentDirs = gitRoot ? ancestorsToRoot(start, gitRoot).reverse() : [start]
  const agents: string[] = []

  for (const dir of agentDirs) {
    const hit = await firstFile(dir, AGENT_CONTEXT_NAMES)

    if (hit) {
      agents.push(hit)
    }
  }

  if (agents.length) {
    return agents
  }

  for (const dir of ancestorsToRoot(start, gitRoot)) {
    const claude = await firstFile(dir, ['CLAUDE.md', 'claude.md'])

    if (claude) {
      return [claude]
    }
  }

  for (const dir of ancestorsToRoot(start, gitRoot)) {
    const cursor = (await firstFile(dir, ['.cursorrules'])) ?? (await cursorRuleFile(dir))

    if (cursor) {
      return [cursor]
    }
  }

  return []
}

/** Locate the context file that wins the host's project-context precedence. */
export const discoverRailInputsFile = async (startDir: string): Promise<string | null> => {
  const candidates = await contextCandidates(startDir)

  return candidates.at(-1) ?? null
}

/**
 * Parse the `## Context rail inputs` block. A malformed block returns null;
 * callers must hide the repo section rather than guessing from partial data.
 */
export const parseRailInputs = (markdown: string): ParsedRailInputs | null => {
  const lines = markdown.split(/\r?\n/)
  const start = lines.findIndex(line => SECTION_RE.test(line))

  if (start === -1) {
    return null
  }

  const body: string[] = []

  for (let i = start + 1; i < lines.length; i++) {
    if (NEXT_SECTION_RE.test(lines[i])) {
      break
    }

    body.push(lines[i])
  }

  const parsed: ParsedRailInputs = { checks: '', decisions: [], evidence: '', flow: '', product: '' }
  const seen = new Set<string>()
  let inDecisions = false

  for (const line of body) {
    if (!line.trim()) {
      continue
    }

    if (inDecisions) {
      const item = DECISION_ITEM_RE.exec(line)

      if (item) {
        parsed.decisions.push(item[1].trim())

        continue
      }

      inDecisions = false
    }

    const top = TOP_LEVEL_RE.exec(line)

    if (!top) {
      return null
    }

    const key = top[1].toLowerCase()

    if (!KNOWN_KEYS.has(key)) {
      if (!top[2].trim()) {
        return null
      }

      continue
    }

    if (seen.has(key)) {
      return null
    }

    seen.add(key)

    if (key === 'decisions') {
      if (top[2].trim()) {
        return null
      }

      inDecisions = true

      continue
    }

    const value = top[2].trim()

    if (!value) {
      return null
    }

    parsed[key as 'checks' | 'evidence' | 'flow' | 'product'] = value
  }

  return REQUIRED_KEYS.every(key => seen.has(key)) ? parsed : null
}

interface ReadResult {
  kind: 'malformed' | 'missing'
  value?: RailInputs
}

const readRailFile = async (sourceFile: string, projectRoot: string | null): Promise<ReadResult> => {
  try {
    const fileStat = await stat(sourceFile)

    if (!fileStat.isFile() || fileStat.size > MAX_RAIL_INPUT_BYTES) {
      return { kind: 'malformed' }
    }

    const markdown = await readFile(sourceFile, 'utf8')
    const hasSection = markdown.split(/\r?\n/).some(line => SECTION_RE.test(line))

    if (!hasSection) {
      return { kind: 'missing' }
    }

    const parsed = parseRailInputs(markdown)

    if (!parsed) {
      return { kind: 'malformed' }
    }

    return {
      kind: 'missing',
      value: {
        ...parsed,
        mtimeMs: fileStat.mtimeMs,
        projectRoot: projectRoot ?? dirname(sourceFile),
        sourceFile
      }
    }
  } catch {
    return { kind: 'malformed' }
  }
}

/** Load the nearest valid repo-owned rail block using the host precedence. */
export const loadRailInputs = async (startDir: string): Promise<RailInputs | null> => {
  try {
    const candidates = await contextCandidates(startDir)
    const projectRoot = await findGitRoot(startDir)

    for (const sourceFile of [...candidates].reverse()) {
      const result = await readRailFile(sourceFile, projectRoot)

      if (result.kind === 'malformed') {
        return null
      }

      if (result.value) {
        return result.value
      }
    }
  } catch {
    return null
  }

  return null
}
