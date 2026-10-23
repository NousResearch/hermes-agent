#!/usr/bin/env -S npx tsx
// Export the bundled TUI English catalog as a flat {key: template} JSON for translators.
// Function leaves are rendered with positional `{0}`, `{1}`… placeholders (the pack format the
// runtime formats with); leaves whose function branches on its argument are flagged so a
// translator can eyeball both branches in the .ts source.
import { writeFileSync } from 'node:fs'

import { en } from '../src/i18n/en.js'

const out: Record<string, string> = {}
const flagged: string[] = []

function walk(node: unknown, prefix: string) {
  if (typeof node === 'string') {
    out[prefix] = node
  } else if (typeof node === 'function') {
    const fn = node as (...a: unknown[]) => string
    const args = Array.from({ length: Math.max(fn.length, 0) }, (_, i) => `{${i}}`)
    let rendered: string
    try {
      rendered = String(fn(...args))
    } catch {
      rendered = ''
    }
    // Detect argument-branching functions: numeric arg 1 vs 2 gives different shapes.
    if (fn.length > 0) {
      try {
        const a = String(fn(...args.map((_, i) => (i === 0 ? 1 : `{${i}}`))))
        const b = String(fn(...args.map((_, i) => (i === 0 ? 2 : `{${i}}`))))
        if (a.replace('1', 'N') !== b.replace('2', 'N')) flagged.push(prefix)
      } catch {
        /* non-numeric first arg */
      }
    }
    out[prefix] = rendered
  } else if (node && typeof node === 'object') {
    for (const [k, v] of Object.entries(node as Record<string, unknown>)) walk(v, prefix ? `${prefix}.${k}` : k)
  }
}

walk(en, '')
const target = process.argv[2]
writeFileSync(target, JSON.stringify({ templates: out, branching: flagged }, null, 2) + '\n')
console.log(`wrote ${Object.keys(out).length} templates (${flagged.length} branching) to ${target}`)
