import { describe, expect, it, vi } from 'vitest'

import { patchUiState, resetUiState } from '../app/uiStore.js'
import { completionRequestForInput } from '../hooks/useCompletion.js'

describe('completionRequestForInput', () => {
  it('routes real slash commands to slash completion', () => {
    expect(completionRequestForInput('/help')).toMatchObject({
      method: 'complete.slash',
      params: { text: '/help' },
      replaceFrom: 1
    })
  })

  it('does not route absolute paths through slash completion', () => {
    expect(
      completionRequestForInput('/home/d/Desktop/agenda/CrimsonRed/.hermes/plans/2026-05-04-HANDOFF-NEXT.md')
    ).toMatchObject({
      method: 'complete.path',
      params: { word: '/home/d/Desktop/agenda/CrimsonRed/.hermes/plans/2026-05-04-HANDOFF-NEXT.md' },
      replaceFrom: 0
    })
  })

  it('keeps path completion for trailing absolute path tokens', () => {
    expect(completionRequestForInput('read /home/d/Desktop/file.md')).toMatchObject({
      method: 'complete.path',
      params: { word: '/home/d/Desktop/file.md' },
      replaceFrom: 5
    })
  })

  it('leaves plain text alone', () => {
    expect(completionRequestForInput('hello there')).toBeNull()
  })

  // The shared gateway's own cwd is its daemon folder: the listing must name the session's.
  it('lists paths in the session folder, else the launch folder', () => {
    try {
      vi.stubEnv('HERMES_CWD', '/launch/dir')
      patchUiState({ info: { cwd: '/work/repo' } as never })
      expect(completionRequestForInput('read @file:sr')).toMatchObject({
        method: 'complete.path',
        params: { word: '@file:sr', cwd: '/work/repo' }
      })
      resetUiState()
      expect(completionRequestForInput('./sr')).toMatchObject({ params: { word: './sr', cwd: '/launch/dir' } })
    } finally {
      resetUiState()
      vi.unstubAllEnvs()
    }
  })

  it('names the asking session, whose folder the shared gateway lists', () => {
    try {
      patchUiState({ sid: 'sess-1' })
      expect(completionRequestForInput('read @file:sr')).toMatchObject({
        method: 'complete.path',
        params: { word: '@file:sr', session_id: 'sess-1' }
      })
      resetUiState()
      expect(completionRequestForInput('./sr')?.params).not.toHaveProperty('session_id')
    } finally {
      resetUiState()
      vi.unstubAllEnvs()
    }
  })
})
