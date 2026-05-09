import { describe, expect, it } from 'vitest'

import { TOOL_VERBS, VERBS } from '../content/verbs.js'

describe('TOOL_VERBS', () => {
  it('exposes a verb for every known tool name used in the TUI', () => {
    const required = [
      'browser',
      'clarify',
      'create_file',
      'delegate_task',
      'delete_file',
      'execute_code',
      'image_generate',
      'list_files',
      'memory',
      'patch',
      'read_file',
      'run_command',
      'search_code',
      'search_files',
      'terminal',
      'web_extract',
      'web_search',
      'write_file'
    ]

    for (const tool of required) {
      expect(TOOL_VERBS[tool], `expected verb for ${tool}`).toBeTypeOf('string')
    }
  })

  it('all verbs are non-empty lowercase strings', () => {
    for (const [tool, verb] of Object.entries(TOOL_VERBS)) {
      expect(verb.length, `${tool} verb`).toBeGreaterThan(0)
      expect(verb, `${tool} verb`).toBe(verb.toLowerCase())
    }
  })

  it('most verbs end in "ing" gerund form', () => {
    const ingCount = Object.values(TOOL_VERBS).filter(v => v.endsWith('ing')).length

    expect(ingCount).toBeGreaterThanOrEqual(Object.keys(TOOL_VERBS).length - 2)
  })
})

describe('VERBS thinking pool', () => {
  it('has at least 10 entries', () => {
    expect(VERBS.length).toBeGreaterThanOrEqual(10)
  })

  it('all entries are unique', () => {
    expect(new Set(VERBS).size).toBe(VERBS.length)
  })

  it('all entries are non-empty lowercase strings', () => {
    for (const v of VERBS) {
      expect(v.length).toBeGreaterThan(0)
      expect(v).toBe(v.toLowerCase())
    }
  })

  it('all entries are gerund (ing) form', () => {
    for (const v of VERBS) {
      expect(v).toMatch(/ing$/)
    }
  })
})
