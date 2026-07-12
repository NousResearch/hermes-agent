import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'

describe('confirm prompt terminal-safe glyphs', () => {
  it('does not use the ambiguous-width warning sign in bordered prompts', () => {
    const source = readFileSync(fileURLToPath(new URL('../components/prompts.tsx', import.meta.url)), 'utf8')

    expect(source).not.toContain('⚠')
  })
})
