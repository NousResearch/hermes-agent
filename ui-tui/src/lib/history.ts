import { appendFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'

const MAX = 1000
const dir = process.env.HERMES_HOME ?? join(homedir(), '.hermes')
const file = join(dir, '.hermes_history')

let cache: string[] | null = null

export function load() {
  if (cache) {
    return cache
  }

  try {
    if (!existsSync(file)) {
      cache = []

      return cache
    }

    const entries: string[] = []
    let current: string[] = []

    for (const line of readFileSync(file, 'utf8').split('\n')) {
      if (line.startsWith('+')) {
        current.push(line.slice(1))
      } else if (current.length) {
        entries.push(current.join('\n'))
        current = []
      }
    }

    if (current.length) {
      entries.push(current.join('\n'))
    }

    cache = entries.slice(-MAX)
  } catch {
    cache = []
  }

  return cache
}

export function append(line: string) {
  const trimmed = line.trim()

  if (!trimmed) {
    return
  }

  const items = load()

  if (items.at(-1) === trimmed) {
    return
  }

  items.push(trimmed)

  if (items.length > MAX) {
    items.splice(0, items.length - MAX)
  }

  try {
    if (!existsSync(dir)) {
      mkdirSync(dir, { recursive: true })
    }

    const ts = new Date().toISOString().replace('T', ' ').replace('Z', '')

    const encoded = trimmed
      .split('\n')
      .map(l => `+${l}`)
      .join('\n')

    appendFileSync(file, `\n# ${ts}\n${encoded}\n`)
  } catch {
    void 0
  }
}
