import { existsSync, mkdirSync, readFileSync, appendFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'

const MAX = 1000
const dir = join(process.env.HERMES_HOME ?? join(homedir(), '.hermes'))
const file = join(dir, 'tui_history')

let cache: string[] | null = null

function encode(s: string): string {
  return s.replace(/\\/g, '\\\\').replace(/\n/g, '\\n')
}

function decode(s: string): string {
  return s.replace(/\\n/g, '\n').replace(/\\\\/g, '\\')
}

export function load(): string[] {
  if (cache) return cache
  try {
    if (existsSync(file)) {
      cache = readFileSync(file, 'utf8')
        .split('\n')
        .filter(Boolean)
        .map(decode)
        .slice(-MAX)
    } else {
      cache = []
    }
  } catch {
    cache = []
  }
  return cache
}

export function append(line: string): void {
  const trimmed = line.trim()
  if (!trimmed) return
  const items = load()
  if (items.at(-1) === trimmed) return
  items.push(trimmed)
  if (items.length > MAX) items.splice(0, items.length - MAX)
  try {
    if (!existsSync(dir)) mkdirSync(dir, { recursive: true })
    appendFileSync(file, encode(trimmed) + '\n')
  } catch { /* ignore */ }
}

export function all(): string[] {
  return load()
}
