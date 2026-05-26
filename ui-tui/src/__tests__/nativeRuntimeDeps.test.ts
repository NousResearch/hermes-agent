import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs'
import { extname, join, relative } from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

const ROOT = fileURLToPath(new URL('../..', import.meta.url))
const SOURCE_EXTENSIONS = new Set(['.js', '.json', '.mjs', '.ts', '.tsx'])
const SKIP_DIRS = new Set(['__tests__', 'dist', 'node_modules'])
const FORBIDDEN_MARKERS = ['@opentui/', 'libopentui', 'opentui']

function collectFiles(target: string): string[] {
  if (!existsSync(target)) {
    return []
  }

  const stat = statSync(target)
  if (stat.isFile()) {
    return SOURCE_EXTENSIONS.has(extname(target)) ? [target] : []
  }

  if (!stat.isDirectory()) {
    return []
  }

  return readdirSync(target, { withFileTypes: true }).flatMap(entry => {
    if (entry.isDirectory() && SKIP_DIRS.has(entry.name)) {
      return []
    }

    return collectFiles(join(target, entry.name))
  })
}

describe('native TUI runtime dependencies', () => {
  it('does not reintroduce the OpenTUI native shared-object runtime', () => {
    const targets = [
      'package.json',
      'package-lock.json',
      'packages/hermes-ink/package.json',
      'packages/hermes-ink/package-lock.json',
      'packages/hermes-ink/src',
      'src'
    ]

    const matches = targets
      .flatMap(target => collectFiles(join(ROOT, target)))
      .flatMap(file => {
        const text = readFileSync(file, 'utf8').toLowerCase()

        return FORBIDDEN_MARKERS.filter(marker => text.includes(marker)).map(marker => ({
          marker,
          path: relative(ROOT, file)
        }))
      })

    expect(matches).toEqual([])
  })
})
