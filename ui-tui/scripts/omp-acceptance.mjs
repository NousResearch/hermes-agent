#!/usr/bin/env node
import { readFileSync } from 'node:fs'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

import { normalizeIndicatorStyle } from '../src/app/useConfigSync.ts'
import { composerFrameChromeCols } from '../src/lib/inputMetrics.ts'
import { toolCardCollapsedByDefault } from '../src/lib/text.ts'

const here = dirname(fileURLToPath(import.meta.url))
const uiRoot = resolve(here, '..')
const repoRoot = resolve(uiRoot, '..')
const readSrc = rel => readFileSync(resolve(uiRoot, 'src', rel), 'utf8')

const failures = []

const assert = (cond, msg) => {
  if (!cond) failures.push(msg)
}

assert(normalizeIndicatorStyle(undefined) === 'unicode', 'normalizeIndicatorStyle default')
assert(/"tui_status_indicator":\s*"unicode"/.test(readFileSync(resolve(repoRoot, 'hermes_cli/config_defaults.py'), 'utf8')), 'python default unicode')

for (const name of ['terminal', 'web_search', 'read_file', 'patch']) {
  assert(toolCardCollapsedByDefault(name) === true, `collapse default for ${name}`)
}

assert(composerFrameChromeCols(true) === 2, 'compact frame chrome')
assert(composerFrameChromeCols(false) === 4, 'comfort frame chrome')

const layout = readSrc('components/appLayout.tsx')
assert(/borderStyle=\{ui\.compact \? 'single' : 'round'\}/.test(layout), 'framed composer border')
assert(/compact=\{firstUserIdx >= 0\}/.test(layout), 'compact intro toggle')

const branding = readSrc('components/branding.tsx')
assert(/useState\(false\)/.test(branding), 'tools accordion starts closed')

const prompts = readSrc('components/prompts.tsx')
assert(!/`\$\{i \+ 1\}\./.test(prompts), 'clarify rows without numeric prefix')

const thinking = readSrc('components/thinking.tsx')
assert(/toolCardCollapsedByDefault/.test(thinking), 'tool cards use collapse helper')

if (failures.length) {
  console.error('OMP acceptance failed:')
  for (const item of failures) console.error(` - ${item}`)
  process.exit(1)
}

console.log(`OMP acceptance passed (${7 - failures.length} checks)`)
