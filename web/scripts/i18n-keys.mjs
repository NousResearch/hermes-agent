#!/usr/bin/env node
// Emit the web dashboard catalog's key set for language-pack validation.
//
// `hermes plugins validate` checks a pack's `<lang>.web.yaml` keys against
// `locales/_keys.web.json` (repo root, committed). This script derives that
// file from `src/i18n/en.ts` with the same `flattenMessageKeys` the renderer's
// pack merge relies on — every leaf's dotted path, sorted; function-valued
// entries are keys too (a pack overrides them with a positional `{0}`/`{1}`
// string). Run via `pnpm run i18n:keys` and commit the result (`--check`
// verifies; `src/i18n/keys.test.ts` pins it in CI). Node ≥ 26 imports the
// TypeScript sources directly (both files carry only type imports).
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { parseArgs } from 'node:util'

export const SURFACE = 'web'

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../..')

export function renderKeysFile(keys) {
  return `${JSON.stringify({ surface: SURFACE, keys }, null, 2)}\n`
}

export async function emitWebKeys({ source = repoRoot, out, check = false } = {}) {
  source = resolve(source)
  out = resolve(out ?? join(source, 'locales', `_keys.${SURFACE}.json`))
  const { en } = await import(pathToFileURL(join(source, 'web/src/i18n/en.ts')).href)
  const { flattenMessageKeys } = await import(pathToFileURL(join(source, 'apps/shared/src/i18n.ts')).href)
  const rendered = renderKeysFile(flattenMessageKeys(en))
  if (check) {
    let current = ''
    try {
      current = readFileSync(out, 'utf8')
    } catch {
      // A missing file is drift too.
    }
    if (current !== rendered) {
      throw new Error(`${out} is out of date with web/src/i18n/en.ts — run \`pnpm run i18n:keys\` in web and commit it`)
    }
    return out
  }
  mkdirSync(dirname(out), { recursive: true })
  writeFileSync(out, rendered)
  return out
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const { values } = parseArgs({
    options: { source: { type: 'string' }, out: { type: 'string' }, check: { type: 'boolean', default: false } }
  })
  const out = await emitWebKeys(values)
  console.log(`${values.check ? 'verified' : 'wrote'} ${out}`)
}
