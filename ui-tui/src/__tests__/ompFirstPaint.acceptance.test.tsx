import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

import { normalizeIndicatorStyle } from '../app/useConfigSync.js'
import { composerFrameChromeCols } from '../lib/inputMetrics.js'
import { toolCardCollapsedByDefault } from '../lib/text.js'

const REPO_ROOT = resolve(import.meta.dirname, '../../..')
const readSrc = (rel: string) => readFileSync(resolve(import.meta.dirname, rel), 'utf8')

describe('OMP first-paint acceptance', () => {
  it('ships unicode as the default busy indicator style', () => {
    expect(normalizeIndicatorStyle(undefined)).toBe('unicode')

    const defaults = readFileSync(resolve(REPO_ROOT, 'hermes_cli/config_defaults.py'), 'utf8')
    expect(defaults).toMatch(/"tui_status_indicator":\s*"unicode"/)
  })

  it('collapses every tool card by default', () => {
    for (const name of ['terminal', 'web_search', 'read_file', 'patch', 'delegate']) {
      expect(toolCardCollapsedByDefault(name)).toBe(true)
    }

    expect(readSrc('../lib/text.ts')).toMatch(/toolCardCollapsedByDefault = \(_name: string\) => true/)
  })

  it('reserves framed composer chrome in compact density', () => {
    expect(composerFrameChromeCols(true)).toBe(2)
    expect(composerFrameChromeCols(false)).toBe(4)

    const layout = readSrc('../components/appLayout.tsx')
    expect(layout).toMatch(/borderStyle=\{ui\.compact \? 'single' : 'round'\}/)
  })

  it('compacts the intro panel after the first user message', () => {
    const layout = readSrc('../components/appLayout.tsx')
  const branding = readSrc('../components/branding.tsx')

    expect(layout).toMatch(/compact=\{firstUserIdx >= 0\}/)
    expect(layout).toMatch(/firstUserIdx < 0 \? <Banner/)
    expect(branding).toMatch(/compact\?: boolean/)
    expect(branding).toMatch(/useState\(false\)/)
  })

  it('uses OMP-style clarify rows without numbered prefixes', () => {
    const prompts = readSrc('../components/prompts.tsx')
    expect(prompts).not.toMatch(/`\$\{i \+ 1\}\./)
    expect(prompts).not.toMatch(/\$\{index \+ 1\}\./)
  })

  it('flattens tool card rows (no tree bullet)', () => {
    const thinking = readSrc('../components/thinking.tsx')
    expect(thinking).not.toMatch(/●/)
    expect(thinking).toMatch(/toolCardCollapsedByDefault/)
  })
})
