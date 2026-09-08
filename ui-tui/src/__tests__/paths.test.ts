import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { composeTabTitle, fmtCwdBranch, fmtProjectCwdBranch, shortCwd, shortProject } from '../domain/paths.js'

describe('shortCwd', () => {
  const origHome = process.env.HOME

  beforeEach(() => {
    process.env.HOME = '/Users/bb'
  })

  afterEach(() => {
    process.env.HOME = origHome
  })

  it('collapses HOME to ~', () => {
    expect(shortCwd('/Users/bb/proj/repo')).toBe('~/proj/repo')
  })

  it('leaves non-HOME paths alone', () => {
    expect(shortCwd('/tmp/work')).toBe('/tmp/work')
  })

  it('truncates long paths from the left with ellipsis', () => {
    const out = shortCwd('/var/long/deeply/nested/workspace/here', 10)
    expect(out.startsWith('…')).toBe(true)
    expect(out.length).toBe(10)
    expect('/var/long/deeply/nested/workspace/here'.endsWith(out.slice(1))).toBe(true)
  })

  it('keeps paths shorter than max intact', () => {
    expect(shortCwd('/a/b', 10)).toBe('/a/b')
  })
})

describe('fmtCwdBranch', () => {
  const origHome = process.env.HOME

  beforeEach(() => {
    process.env.HOME = '/Users/bb'
  })

  afterEach(() => {
    process.env.HOME = origHome
  })

  it('returns bare cwd when branch is null', () => {
    expect(fmtCwdBranch('/Users/bb/proj', null)).toBe('~/proj')
  })

  it('returns bare cwd when branch is empty', () => {
    expect(fmtCwdBranch('/Users/bb/proj', '')).toBe('~/proj')
  })

  it('appends branch in parens', () => {
    expect(fmtCwdBranch('/Users/bb/proj', 'main')).toBe('~/proj (main)')
  })

  it('truncates the path to keep the branch tag readable', () => {
    const out = fmtCwdBranch('/Users/bb/very/deeply/nested/project/folder', 'feature-branch', 30)
    expect(out).toMatch(/ \(feature-branch\)$/)
    expect(out.length).toBeLessThanOrEqual(30)
  })

  it('truncates very long branch names from the right', () => {
    const out = fmtCwdBranch('/Users/bb/p', 'a-very-long-feature-branch-name')
    expect(out).toMatch(/^~\/p \(…/)
    expect(out).toContain(')')
  })
})

describe('shortProject', () => {
  it('trims whitespace', () => {
    expect(shortProject('  website  ')).toBe('website')
  })

  it('truncates long project names from the right', () => {
    expect(shortProject('a-very-long-project-name', 10)).toBe('a-very-lo…')
  })
})

describe('fmtProjectCwdBranch', () => {
  const origHome = process.env.HOME

  beforeEach(() => {
    process.env.HOME = '/Users/bb'
  })

  afterEach(() => {
    process.env.HOME = origHome
  })

  it('prefixes the cwd/branch label with the project name', () => {
    expect(fmtProjectCwdBranch('/Users/bb/proj', 'main', 'website', 28)).toBe('website · ~/proj (main)')
  })

  it('falls back to the cwd/branch label when no project is known', () => {
    expect(fmtProjectCwdBranch('/Users/bb/proj', 'main', null, 28)).toBe('~/proj (main)')
  })

  it('keeps the project visible when space is tight', () => {
    expect(fmtProjectCwdBranch('/Users/bb/proj', 'main', 'hermes-agent', 12)).toBe('hermes-agent')
  })
})

describe('composeTabTitle', () => {
  it('joins marker, name, model, and cwd in order', () => {
    expect(composeTabTitle('✓', 'auth refactor', 'opus-4', '~/proj')).toBe('✓ auth refactor · opus-4 · ~/proj')
  })

  it('glues the marker to the first segment with a space, not a separator', () => {
    expect(composeTabTitle('⏳', 'my session', 'opus-4', '~/proj').startsWith('⏳ my session')).toBe(true)
  })

  it('omits the session name when empty (matches the pre-name format)', () => {
    expect(composeTabTitle('✓', '', 'opus-4', '~/proj')).toBe('✓ opus-4 · ~/proj')
  })

  it('treats a whitespace-only name as absent', () => {
    expect(composeTabTitle('✓', '   ', 'opus-4', '~/proj')).toBe('✓ opus-4 · ~/proj')
  })

  it('omits the cwd when empty', () => {
    expect(composeTabTitle('✓', 'my session', 'opus-4', '')).toBe('✓ my session · opus-4')
  })

  it('falls back to just the marker when only the marker is present', () => {
    expect(composeTabTitle('✓', '', '', '')).toBe('✓')
  })

  it('truncates an over-long session name with an ellipsis', () => {
    const long = 'a'.repeat(40)
    const out = composeTabTitle('✓', long, 'opus-4', '', 28)
    const namePart = out.slice('✓ '.length).split(' · ')[0]
    expect(namePart.endsWith('…')).toBe(true)
    expect(namePart.length).toBe(28)
  })

  it('keeps a name at the boundary length intact', () => {
    const name = 'b'.repeat(28)
    const out = composeTabTitle('✓', name, 'opus-4', '', 28)
    expect(out).toBe(`✓ ${name} · opus-4`)
  })
})
