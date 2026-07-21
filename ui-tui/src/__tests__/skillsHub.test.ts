import { describe, expect, it } from 'vitest'

import { classifySkillsListKey, rankSkillCategories, rankSkills } from '../components/skillsHub.js'

describe('skills hub fuzzy filtering', () => {
  const skillsByCat = {
    github: ['pr-review', 'issue-triage'],
    mlops: ['train-model', 'deploy-endpoint'],
    security: ['threat-model', 'secret-scan']
  }

  const cats = Object.keys(skillsByCat).sort()

  it('keeps the full category list for an empty query', () => {
    expect(rankSkillCategories(cats, skillsByCat, '')).toEqual(cats)
    expect(rankSkillCategories(cats, skillsByCat, '   ')).toEqual(cats)
  })

  it('ranks categories by name, best match first', () => {
    const ranked = rankSkillCategories(cats, skillsByCat, 'sec')

    expect(ranked[0]).toBe('security')
  })

  it('surfaces a category by the names of the skills it contains', () => {
    // "deploy" is a skill in mlops, not a category name.
    expect(rankSkillCategories(cats, skillsByCat, 'deploy')).toContain('mlops')
  })

  it('drops categories that do not match', () => {
    expect(rankSkillCategories(cats, skillsByCat, 'zzzz')).toEqual([])
  })

  it('filters and ranks skills within a category', () => {
    const skills = skillsByCat.security

    expect(rankSkills(skills, '')).toEqual(skills)
    expect(rankSkills(skills, 'scan')).toEqual(['secret-scan'])
    expect(rankSkills(skills, 'zzzz')).toEqual([])
  })
})

describe('skills hub list-stage key gating', () => {
  it('fires reserved single-key shortcuts only when no filter is active', () => {
    expect(classifySkillsListKey('q', {}, false)).toEqual({ kind: 'close' })
    expect(classifySkillsListKey('1', {}, false)).toEqual({ kind: 'quick', n: 1 })
    expect(classifySkillsListKey('0', {}, false)).toEqual({ kind: 'quick', n: 10 })
  })

  it('types the same reserved keys into the filter once it is active', () => {
    expect(classifySkillsListKey('q', {}, true)).toEqual({ ch: 'q', kind: 'append' })
    expect(classifySkillsListKey('1', {}, true)).toEqual({ ch: '1', kind: 'append' })
  })

  it('starts a filter when a non-reserved printable key is pressed', () => {
    expect(classifySkillsListKey('a', {}, false)).toEqual({ ch: 'a', kind: 'append' })
  })

  it('does not start a filter on a leading space, but allows spaces mid-query', () => {
    expect(classifySkillsListKey(' ', {}, false)).toEqual({ kind: 'ignore' })
    expect(classifySkillsListKey(' ', {}, true)).toEqual({ ch: ' ', kind: 'append' })
  })

  it('clears a non-empty filter on Esc before navigating back', () => {
    expect(classifySkillsListKey('', { escape: true }, true)).toEqual({ kind: 'clearFilter' })
    expect(classifySkillsListKey('', { escape: true }, false)).toEqual({ kind: 'escape' })
  })

  it('edits the filter with Backspace / Ctrl+U only while filtering', () => {
    expect(classifySkillsListKey('', { backspace: true }, true)).toEqual({ kind: 'backspace' })
    expect(classifySkillsListKey('', { backspace: true }, false)).toEqual({ kind: 'ignore' })
    expect(classifySkillsListKey('u', { ctrl: true }, true)).toEqual({ kind: 'clearFilter' })
    expect(classifySkillsListKey('u', { ctrl: true }, false)).toEqual({ kind: 'ignore' })
  })

  it('routes navigation keys regardless of filter state', () => {
    expect(classifySkillsListKey('', { upArrow: true }, true)).toEqual({ kind: 'up' })
    expect(classifySkillsListKey('', { downArrow: true }, false)).toEqual({ kind: 'down' })
    expect(classifySkillsListKey('', { return: true }, true)).toEqual({ kind: 'select' })
  })

  it('does not treat modified chords as filter input', () => {
    expect(classifySkillsListKey('g', { ctrl: true }, true)).toEqual({ kind: 'ignore' })
    expect(classifySkillsListKey('g', { meta: true }, true)).toEqual({ kind: 'ignore' })
  })
})
