// @vitest-environment jsdom

import { act, renderHook, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import type { GatewayClient } from '../gatewayClient.js'
import { firstCandidateText, ghostVisible, useGhostSuggestion } from '../hooks/useGhostSuggestion.js'

describe('firstCandidateText', () => {
  it('returns the first candidate from a result envelope', () => {
    const raw = {
      candidates: [
        { kind: 'path', text: '~/Downloads/report.xlsx' },
        { kind: 'confirm', text: 'Yes, go ahead' }
      ],
      history_version: 12
    }

    expect(firstCandidateText(raw)).toBe('~/Downloads/report.xlsx')
  })

  it('returns empty string for empty or malformed results', () => {
    expect(firstCandidateText({ candidates: [] })).toBe('')
    expect(firstCandidateText({})).toBe('')
    expect(firstCandidateText(null)).toBe('')
    expect(firstCandidateText({ candidates: [{ kind: 'path' }] })).toBe('')
  })

  it('trims candidate whitespace', () => {
    expect(firstCandidateText({ candidates: [{ kind: 'confirm', text: '  yes  ' }] })).toBe('yes')
  })
})

describe('ghostVisible', () => {
  const base = { blocked: false, dismissed: false, ghost: 'Yes, go ahead', input: '' }

  it('shows only in an idle, empty, undismissed composer with a ghost', () => {
    expect(ghostVisible(base)).toBe(true)
  })

  it('hides while the agent is busy', () => {
    expect(ghostVisible({ ...base, blocked: true })).toBe(false)
  })

  it('hides as soon as the user has typed anything', () => {
    expect(ghostVisible({ ...base, input: 'n' })).toBe(false)
  })

  it('stays hidden after an explicit dismiss', () => {
    expect(ghostVisible({ ...base, dismissed: true })).toBe(false)
  })

  it('never shows an empty ghost', () => {
    expect(ghostVisible({ ...base, ghost: '' })).toBe(false)
  })
})

describe('useGhostSuggestion', () => {
  it('drops a delayed response after the selected session changes', async () => {
    let resolveRequest: (value: unknown) => void = () => {}
    let sessionId = 'session-a'

    const gateway = {
      request: vi.fn(
        () =>
          new Promise(resolve => {
            resolveRequest = resolve
          })
      )
    } as unknown as GatewayClient

    const getSessionId = () => sessionId

    const { result, rerender } = renderHook(
      ({ blocked }) => useGhostSuggestion('', blocked, gateway, getSessionId),
      { initialProps: { blocked: true } }
    )

    rerender({ blocked: false })
    expect(gateway.request).toHaveBeenCalledWith('complete.suggest', { session_id: 'session-a' })

    sessionId = 'session-b'
    await act(async () => {
      resolveRequest({ candidates: [{ kind: 'confirm', text: 'Yes, go ahead' }] })
      await Promise.resolve()
    })

    expect(result.current.ghost).toBe('')
  })

  it('does not resurrect a suggestion after typing and deleting', async () => {
    const gateway = {
      request: vi.fn().mockResolvedValue({
        candidates: [{ kind: 'confirm', text: 'Yes, go ahead' }]
      })
    } as unknown as GatewayClient

    const getSessionId = () => 'session-a'

    const { result, rerender } = renderHook(
      ({ blocked, input }) => useGhostSuggestion(input, blocked, gateway, getSessionId),
      { initialProps: { blocked: true, input: '' } }
    )

    rerender({ blocked: false, input: '' })
    await waitFor(() => expect(result.current.ghost).toBe('Yes, go ahead'))

    rerender({ blocked: false, input: 'n' })
    expect(result.current.ghost).toBe('')

    rerender({ blocked: false, input: '' })
    expect(result.current.ghost).toBe('')
  })
})
