import { describe, expect, it } from 'vitest'

import { isSectionName, parseDetailsMode, resolveSections, sectionMode, SECTION_NAMES } from '../domain/details.js'

describe('parseDetailsMode', () => {
  it('accepts the canonical modes case-insensitively', () => {
    expect(parseDetailsMode('hidden')).toBe('hidden')
    expect(parseDetailsMode(' COLLAPSED ')).toBe('collapsed')
    expect(parseDetailsMode('Expanded')).toBe('expanded')
  })

  it('rejects junk', () => {
    expect(parseDetailsMode('truncated')).toBeNull()
    expect(parseDetailsMode('')).toBeNull()
    expect(parseDetailsMode(undefined)).toBeNull()
    expect(parseDetailsMode(42)).toBeNull()
  })
})

describe('isSectionName', () => {
  it('only lets the four canonical sections through', () => {
    expect(isSectionName('thinking')).toBe(true)
    expect(isSectionName('tools')).toBe(true)
    expect(isSectionName('subagents')).toBe(true)
    expect(isSectionName('activity')).toBe(true)

    expect(isSectionName('Thinking')).toBe(false) // case-sensitive on purpose
    expect(isSectionName('bogus')).toBe(false)
    expect(isSectionName('')).toBe(false)
    expect(isSectionName(7)).toBe(false)
  })

  it('SECTION_NAMES exposes them all', () => {
    expect([...SECTION_NAMES].sort()).toEqual(['activity', 'subagents', 'thinking', 'tools'])
  })
})

describe('resolveSections', () => {
  it('parses a well-formed sections object', () => {
    expect(
      resolveSections({
        thinking: 'expanded',
        tools: 'expanded',
        subagents: 'collapsed',
        activity: 'hidden'
      })
    ).toEqual({
      thinking: 'expanded',
      tools: 'expanded',
      subagents: 'collapsed',
      activity: 'hidden'
    })
  })

  it('drops unknown section names and unknown modes', () => {
    expect(
      resolveSections({
        thinking: 'expanded',
        tools: 'maximised',
        bogus: 'hidden',
        activity: 'hidden'
      })
    ).toEqual({ thinking: 'expanded', activity: 'hidden' })
  })

  it('treats nullish/non-objects as empty overrides', () => {
    expect(resolveSections(undefined)).toEqual({})
    expect(resolveSections(null)).toEqual({})
    expect(resolveSections('hidden')).toEqual({})
    expect(resolveSections([])).toEqual({})
  })
})

describe('sectionMode', () => {
  it('falls back to the global mode for sections without a built-in default', () => {
    expect(sectionMode('subagents', 'collapsed', {})).toBe('collapsed')
    expect(sectionMode('subagents', 'expanded', undefined)).toBe('expanded')
    expect(sectionMode('subagents', 'hidden', {})).toBe('hidden')
  })

  it('streams thinking + tools expanded by default regardless of global mode', () => {
    expect(sectionMode('thinking', 'collapsed', {})).toBe('expanded')
    expect(sectionMode('thinking', 'hidden', undefined)).toBe('expanded')
    expect(sectionMode('tools', 'collapsed', {})).toBe('expanded')
    expect(sectionMode('tools', 'hidden', undefined)).toBe('expanded')
  })

  it('hides the activity panel by default regardless of global mode', () => {
    expect(sectionMode('activity', 'collapsed', {})).toBe('hidden')
    expect(sectionMode('activity', 'expanded', undefined)).toBe('hidden')
    expect(sectionMode('activity', 'hidden', {})).toBe('hidden')
  })

  it('honours per-section overrides over both the section default and global mode', () => {
    expect(sectionMode('thinking', 'collapsed', { thinking: 'collapsed' })).toBe('collapsed')
    expect(sectionMode('tools', 'collapsed', { tools: 'hidden' })).toBe('hidden')
    expect(sectionMode('activity', 'collapsed', { activity: 'expanded' })).toBe('expanded')
    expect(sectionMode('activity', 'expanded', { activity: 'collapsed' })).toBe('collapsed')
  })

  it('lets per-section overrides escape the global hidden mode', () => {
    // Regression for the case where global details_mode: hidden used to
    // short-circuit the entire accordion and prevent overrides from
    // surfacing — `sections.tools: expanded` must still resolve to expanded.
    expect(sectionMode('subagents', 'hidden', { subagents: 'expanded' })).toBe('expanded')
    expect(sectionMode('thinking', 'hidden', { thinking: 'collapsed' })).toBe('collapsed')
    expect(sectionMode('activity', 'hidden', { activity: 'expanded' })).toBe('expanded')
  })
})
