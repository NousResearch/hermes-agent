#!/usr/bin/env -S npx tsx
// Emit <repo>/locales/_keys.tui.json: the sorted flat key list of the bundled
// TUI English catalog. `hermes plugins validate` checks a pack's `<lang>.tui.yaml`
// keys against this file, so it is regenerated on every `npm run build` and
// committed. Run directly: `npm run i18n:keys` (add `--check` to verify only).

import { readFileSync, writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { en } from '../src/i18n/en.js'
import { flattenKeys } from '../src/i18n/keys.js'

const here = dirname(fileURLToPath(import.meta.url))
const target = resolve(here, '../../locales/_keys.tui.json')
const keys = flattenKeys(en)
const body = JSON.stringify(keys, null, 2) + '\n'

if (process.argv.includes('--check')) {
  let current = ''

  try {
    current = readFileSync(target, 'utf8')
  } catch {
    current = ''
  }

  if (current !== body) {
    console.error(`${target} is stale — run \`npm run i18n:keys\` in ui-tui/`)
    process.exit(1)
  }

  console.log(`${target} is fresh (${keys.length} keys)`)
} else {
  writeFileSync(target, body)
  console.log(`wrote ${keys.length} keys to ${target}`)
}
