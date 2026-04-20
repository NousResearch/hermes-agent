import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { fmtCwdBranch, shortCwd } from '../domain/paths.js'

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
