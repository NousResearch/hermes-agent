import { closeSync, constants, existsSync, fchmodSync, fstatSync, mkdirSync, openSync, readSync, writeSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'

const MAX = 1000
// Only the tail is read: MAX entries never need more, and a years-old file stays cheap to open.
const READ_TAIL_BYTES = 1 << 20
const dir = process.env.HERMES_HOME ?? join(homedir(), '.hermes')
const file = join(dir, '.hermes_history')

// Prompts can carry pasted secrets: the file is owner-only, and a symlink planted at its path is
// refused rather than followed (O_NOFOLLOW is undefined on Windows, where it is a no-op).
const PRIVATE_MODE = 0o600
const NOFOLLOW = constants.O_NOFOLLOW ?? 0

let cache: string[] | null = null

function readTail(): string {
  const fd = openSync(file, constants.O_RDONLY | NOFOLLOW)

  try {
    const size = fstatSync(fd).size
    const length = Math.min(size, READ_TAIL_BYTES)
    const buf = Buffer.alloc(length)
    readSync(fd, buf, 0, length, size - length)
    const text = buf.toString('utf8')

    if (length === size) {
      return text
    }

    // A cut tail starts mid-entry: drop everything before the first complete record. No header at
    // all means one record outgrew the window; its tail alone is not the prompt, so recall nothing.
    const boundary = text.indexOf('\n#')

    return boundary === -1 ? '' : text.slice(boundary + 1)
  } finally {
    closeSync(fd)
  }
}

export function parse(text: string): string[] {
  const entries: string[] = []
  let current: string[] = []

  for (const line of text.split('\n')) {
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

  return entries.slice(-MAX)
}

export function load() {
  if (cache) {
    return cache
  }

  try {
    cache = existsSync(file) ? parse(readTail()) : []
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

    const fd = openSync(file, constants.O_WRONLY | constants.O_APPEND | constants.O_CREAT | NOFOLLOW, PRIVATE_MODE)

    try {
      // Tighten a file an older build created world-readable.
      fchmodSync(fd, PRIVATE_MODE)
      writeSync(fd, `\n# ${ts}\n${encoded}\n`)
    } finally {
      closeSync(fd)
    }
  } catch {
    void 0
  }
}
