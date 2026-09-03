#!/usr/bin/env node
import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const uiRoot = resolve(here, '..')
const repoRoot = resolve(uiRoot, '..')
const readSrc = rel => readFileSync(resolve(uiRoot, 'src', rel), 'utf8')

const failures = []
let checks = 0
const assert = (cond, msg) => {
  checks += 1
  if (!cond) failures.push(msg)
}

const frameChromeCols = compact => 2 + (compact ? 0 : 2)

assert(frameChromeCols(true) === 2, 'compact frame chrome')
assert(frameChromeCols(false) === 4, 'comfort frame chrome')
assert(/composerFrameChromeCols = \(compact: boolean\) => 2 \+ \(compact \? 0 : 2\)/.test(readSrc('lib/inputMetrics.ts')), 'composerFrameChromeCols formula')

const interfaces = readSrc('app/interfaces.ts')
assert(/DEFAULT_INDICATOR_STYLE[^'\n]*'unicode'/.test(interfaces), 'DEFAULT_INDICATOR_STYLE is unicode')
assert(/typeof raw !== 'string'/.test(readSrc('app/useConfigSync.ts')), 'normalizeIndicatorStyle falls back for non-strings')

const defaults = readFileSync(resolve(repoRoot, 'hermes_cli/config_defaults.py'), 'utf8')
assert(/"tui_status_indicator":\s*"unicode"/.test(defaults), 'python default unicode')

const layout = readSrc('components/appLayout.tsx')
assert(/borderStyle=\{ui\.compact \? 'single' : 'round'\}/.test(layout), 'framed composer border')
assert(/compact=\{firstUserIdx >= 0\}/.test(layout), 'compact intro toggle')

const branding = readSrc('components/branding.tsx')
assert(/useState\(false\)/.test(branding), 'tools accordion starts closed')

const prompts = readSrc('components/prompts.tsx')
assert(!/`\$\{i \+ 1\}\./.test(prompts), 'clarify rows without numeric prefix')

const thinking = readSrc('components/thinking.tsx')
assert(/collapsedDefault: true/.test(thinking), 'tool cards default collapsed')
assert(!/toolCardCollapsedByDefault/.test(thinking), 'no toolCardCollapsedByDefault stub')
const toolCardBlock = thinking.match(/function ToolCard\([\s\S]*?\n\}/)?.[0] ?? ''
assert(toolCardBlock.length > 0, 'ToolCard component present')
assert(!/●/.test(toolCardBlock), 'ToolCard has no tree bullet glyph')

if (failures.length) {
  console.error('OMP acceptance failed:')
  for (const item of failures) console.error(` - ${item}`)
  process.exit(1)
}

console.log(`OMP acceptance passed (${checks} checks)`)
