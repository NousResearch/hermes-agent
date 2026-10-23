import { describe, expect, it } from 'vitest'

import { sessionCreateParams, temporaryCreateOption } from '../app/sessionCreateParams.js'

describe('sessionCreateParams', () => {
  it('a normal TUI session asks for nothing temporary', () => {
    expect(sessionCreateParams(96, { temporary: false, workspaceCwd: '' })).toEqual({ cols: 96 })
  })

  it('hermes --temporary makes every created session temporary', () => {
    expect(sessionCreateParams(80, { temporary: true, workspaceCwd: '/w' })).toEqual({
      cols: 80,
      cwd: '/w',
      temporary: true
    })
  })
})

describe('temporaryCreateOption (canonical gateway create)', () => {
  it('adds temporary only for a hermes --temporary launch', () => {
    expect(temporaryCreateOption(true)).toEqual({ temporary: true })
    expect(temporaryCreateOption(false)).toEqual({})
  })
})

describe('temporaryRefusal', () => {
  it('refuses session switching and branching only in temporary mode', async () => {
    const { temporaryRefusal } = await import('../app/slash/temporary.js')

    expect(temporaryRefusal('sessions', true)).toContain('/sessions')
    expect(temporaryRefusal('branch', true)).toContain('/branch')
    expect(temporaryRefusal('model', true)).toBeNull()
    expect(temporaryRefusal('sessions', false)).toBeNull()
  })
})
