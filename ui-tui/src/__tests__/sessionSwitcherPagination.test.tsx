import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import type { SessionListRow } from '@hermes/shared/gateway-events'
import React from 'react'
import { describe, expect, it, vi } from 'vitest'

import { ActiveSessionSwitcher } from '../components/activeSessionSwitcher.js'
import {
  buildSwitcherWindowedRows,
  MAX_SAFE_INTEGER,
  resolveLivePollSelection,
  resolvePageChangeSelection,
  validateSessionPageResponse
} from '../components/sessionPagination.js'
import type { SessionActiveItem } from '../gatewayTypes.js'
import { messages } from '../i18n/runtime.js'
import { DEFAULT_THEME } from '../theme.js'

const WAIT = { timeout: 10_000 }
const TEST_TIMEOUT = 30_000

function makeMockSessions(prefix: string, count: number, startId = 0): SessionListRow[] {
  return Array.from({ length: count }, (_, i) => ({
    id: `${prefix}-${startId + i}`,
    message_count: 1,
    model: 'test-model',
    preview: `Preview ${prefix} ${startId + i}`,
    running: false,
    session_id: `${prefix}-${startId + i}`,
    source: 'tui',
    started_at: 1000 + startId + i,
    title: `Session ${prefix} ${startId + i}`
  }))
}

describe('sessionPagination pure unit tests', () => {
  it('validates canonical and legacy metadata envelopes according to spec', () => {
    // Valid initial / later page
    const validRaw = {
      has_more: true,
      limit: 200,
      offset: 0,
      sessions: makeMockSessions('s', 200)
    }
    expect(validateSessionPageResponse({ sessions: makeMockSessions('old', 201) }, 0, true).kind).toBe('malformed')
    const res = validateSessionPageResponse(validRaw, 0, true)
    expect(res.kind).toBe('valid')
    if (res.kind === 'valid') {
      expect(res.meta).toEqual({ has_more: true, limit: 200, offset: 0 })
      expect(res.sessions.length).toBe(200)
    }

    // Older gateway without metadata on initial request
    const legacyRaw = {
      sessions: makeMockSessions('s', 10)
    }
    expect(validateSessionPageResponse(legacyRaw, 0, true)).toEqual({
      kind: 'legacy-unpaged',
      sessions: legacyRaw.sessions
    })

    // Older gateway without metadata on later request (offset > 0) -> malformed
    expect(validateSessionPageResponse(legacyRaw, 200, false).kind).toBe('malformed')

    // Missing metadata on later request back to offset 0 (Newer0) must not downgrade to legacy-unpaged
    expect(validateSessionPageResponse(legacyRaw, 0, false).kind).toBe('malformed')

    // JSON-RPC result envelope support
    const rpcEnvelope = {
      result: {
        has_more: true,
        limit: 200,
        offset: 0,
        sessions: makeMockSessions('s', 200)
      }
    }
    const rpcRes = validateSessionPageResponse(rpcEnvelope, 0, true)
    expect(rpcRes.kind).toBe('valid')

    // Wrong offset
    expect(validateSessionPageResponse({ ...validRaw, offset: 100 }, 200).kind).toBe('malformed')

    // Out of bounds limit (0 or >200 or float)
    expect(validateSessionPageResponse({ ...validRaw, limit: 0 }, 0).kind).toBe('malformed')
    expect(validateSessionPageResponse({ ...validRaw, limit: 201 }, 0).kind).toBe('malformed')
    expect(validateSessionPageResponse({ ...validRaw, limit: 50.5 }, 0).kind).toBe('malformed')

    // Non-boolean has_more
    expect(validateSessionPageResponse({ ...validRaw, has_more: 'true' as unknown as boolean }, 0).kind).toBe(
      'malformed'
    )

    // Sessions over limit
    expect(
      validateSessionPageResponse({ ...validRaw, limit: 10, sessions: makeMockSessions('s', 11) }, 0).kind
    ).toBe('malformed')

    // Empty page with has_more: true -> malformed
    expect(validateSessionPageResponse({ has_more: true, limit: 200, offset: 200, sessions: [] }, 200).kind).toBe(
      'malformed'
    )

    // Unsafe next offset > MAX_SAFE_INTEGER is validated ONLY if has_more is true
    expect(
      validateSessionPageResponse(
        { has_more: false, limit: 200, offset: MAX_SAFE_INTEGER - 10, sessions: makeMockSessions('s', 5) },
        MAX_SAFE_INTEGER - 10
      ).kind
    ).toBe('valid')
    expect(
      validateSessionPageResponse(
        { has_more: true, limit: 200, offset: MAX_SAFE_INTEGER - 10, sessions: makeMockSessions('s', 5) },
        MAX_SAFE_INTEGER - 10
      ).kind
    ).toBe('malformed')
  })

  it.each([
    null,
    'not an object',
    [],
    {},
    { id: 7 },
    { id: 'bad', title: {} },
    { id: 'bad', preview: [] },
    { id: 'bad', started_at: 'yesterday' },
    { id: 'bad', message_count: {} }
  ])('rejects malformed rows before accepting paged or legacy responses: %j', row => {
    const error = messages().pickers.common.invalidResponse('session.list')
    expect(validateSessionPageResponse({ sessions: [row] }, 0, true)).toEqual({ error, kind: 'malformed' })
    expect(validateSessionPageResponse({ has_more: false, limit: 200, offset: 200, sessions: [row] }, 200)).toEqual({
      error,
      kind: 'malformed'
    })
  })

  it('builds windowed rows with dedicated pagination controls', () => {
    const live: SessionActiveItem[] = [
      {
        current: true,
        id: 'live-1',
        last_active: 1,
        message_count: 1,
        model: 'm',
        preview: 'p',
        session_key: 'k',
        started_at: 1,
        status: 'idle',
        title: 'Live 1'
      }
    ]
    const hist = makeMockSessions('h', 5)

    // Page 0 with has_more: false -> no Older/Newer
    const rows0 = buildSwitcherWindowedRows(live, hist, 0, 200, false)
    expect(rows0.map(r => r.kind)).toEqual(['live', 'history', 'history', 'history', 'history', 'history'])

    // Page 0 with has_more: true -> Older at end
    const rows1 = buildSwitcherWindowedRows(live, hist, 0, 200, true)
    expect(rows1.map(r => r.kind)).toEqual([
      'live',
      'history',
      'history',
      'history',
      'history',
      'history',
      'pagination-older'
    ])

    // Page 1 (offset 200) with has_more: false -> Newer at top of history
    const rows2 = buildSwitcherWindowedRows(live, hist, 200, 200, false)
    expect(rows2.map(r => r.kind)).toEqual([
      'live',
      'pagination-newer',
      'history',
      'history',
      'history',
      'history',
      'history'
    ])

    // Page 1 (offset 200) with has_more: true -> Newer at top, Older at end
    const rows3 = buildSwitcherWindowedRows(live, hist, 200, 200, true)
    expect(rows3.map(r => r.kind)).toEqual([
      'live',
      'pagination-newer',
      'history',
      'history',
      'history',
      'history',
      'history',
      'pagination-older'
    ])
  })

  it('resolves selection fallback and preserves selection across page and live changes', () => {
    const live: SessionActiveItem[] = [
      {
        current: false,
        id: 'l1',
        last_active: 1,
        message_count: 1,
        model: 'm',
        preview: 'p',
        session_key: 'k',
        started_at: 1,
        status: 'idle',
        title: 'L1'
      }
    ]
    const hist1 = makeMockSessions('h1', 3)
    const rows1 = buildSwitcherWindowedRows(live, hist1, 200, 200, false)

    // Fallback: previous history ID not in new page -> selects first history row (index in list is 2 -> sel 3)
    const selFallback = resolvePageChangeSelection('history', 'old-id-not-in-page', live, hist1, rows1)
    expect(selFallback).toBe(3) // 0=new, 1=l1, 2=pagination-newer, 3=first history

    // When no history on new page -> fallback to new row (sel 0)
    const emptyRows = buildSwitcherWindowedRows(live, [], 200, 200, false)
    const selEmpty = resolvePageChangeSelection('history', 'missing', live, [], emptyRows)
    expect(selEmpty).toBe(0)

    // User was on new row -> stays on new row
    expect(resolvePageChangeSelection('new', null, live, hist1, rows1)).toBe(0)

    // User was on live row -> stays on live row (index 0 in rows1 -> sel 1)
    expect(resolvePageChangeSelection('live', 'l1', live, hist1, rows1)).toBe(1)

    // Live poll preserves history selection by ID
    const selLivePoll = resolveLivePollSelection(3, 'history', 'h1-1', rows1)
    expect(selLivePoll).toBe(4) // h1-1 is index 3 in rows1 (after live and pagination-newer) -> sel 4
  })
})

describe('Mounted ActiveSessionSwitcher pagination', () => {
  it.each(
    [true, false].flatMap(canonical =>
      ['transport', 'malformed'].flatMap(failure =>
        ['history', 'newer', 'older'].map(selected => ({ canonical, failure, selected }))
      )
    )
  )(
    'preserves committed $selected selection after failed Ctrl+R ($failure, canonical=$canonical)',
    async ({ canonical, failure, selected }) => {
      let refresh = 'initial'
      let liveRows: SessionListRow[] = []
      const requests: Array<Record<string, unknown>> = []
      const pageA = makeMockSessions('refresh-A', 1)
      const pageB = makeMockSessions('refresh-B', 1)
      const request = vi.fn(async (method: string, params: Record<string, unknown>) => {
        if (method === 'session.active_list' || (canonical && params.scope !== 'stored')) {
          return { sessions: liveRows }
        }
        requests.push({ ...params })
        if (params.offset === undefined && refresh === 'failed') {
          if (failure === 'transport') {throw new Error('refresh-offline')}
          return { has_more: true, limit: 1, offset: 0, sessions: [null] }
        }
        const offset = (params.offset as number) ?? 0
        return { has_more: true, limit: 1, offset, sessions: offset === 0 ? pageA : pageB }
      })
      const stdin = Object.assign(new PassThrough(), { isTTY: true, ref: vi.fn(), setRawMode: vi.fn(), unref: vi.fn() })
      const stdout = Object.assign(new PassThrough(), { columns: 120, isTTY: false, rows: 30 })
      const out = { text: '' }
      stdout.on('data', chunk => { out.text += chunk.toString() })
      const onResume = vi.fn()
      const onSelect = vi.fn()
      const instance = renderSync(
        <ActiveSessionSwitcher
          currentSessionId={null}
          gw={{ isCanonical: canonical, request } as any}
          onCancel={vi.fn()}
          onClose={vi.fn()}
          onNew={vi.fn()}
          onNewPrompt={vi.fn()}
          onResume={onResume}
          onSelect={onSelect}
          t={DEFAULT_THEME}
        />,
        { patchConsole: false, stderr: new PassThrough() as any, stdin: stdin as any, stdout: stdout as any }
      )
      try {
        await vi.waitFor(() => expect(out.text).toContain('Session refresh-A 0'), WAIT)
        await vi.waitFor(() => {
          if (!out.text.includes('Page: Enter load page')) {stdin.write('\u001b[B')}
          expect(out.text).toContain('Page: Enter load page')
        }, WAIT)
        out.text = ''
        stdin.write('\r')
        await vi.waitFor(() => expect(out.text).toMatch(/▸[^\n]*Session refresh-B 0/), WAIT)
        expect(requests[1]).toEqual({ limit: 1, offset: 1, ...(canonical ? { scope: 'stored' } : {}) })

        // Positive control: the committed history row really resumes before refresh.
        stdin.write('\r')
        await vi.waitFor(() => expect(onResume.mock.calls).toEqual([[pageB[0]!.id]]), WAIT)
        onResume.mockClear()
        if (selected !== 'history') {
          out.text = ''
          stdin.write(selected === 'newer' ? '\u001b[A' : '\u001b[B')
          await vi.waitFor(() => expect(out.text).toMatch(
            selected === 'newer' ? /▸[^\n]*Newer sessions/ : /▸[^\n]*Older sessions/
          ), WAIT)
        }

        // Live data still updates, but failed history cannot stage a first-page layout.
        refresh = 'failed'
        liveRows = makeMockSessions('refresh-live', 1)
        out.text = ''
        stdin.write('\u0012')
        await vi.waitFor(() => {
          expect(out.text).toContain(failure === 'transport'
            ? 'refresh-offline' : messages().pickers.common.invalidResponse('session.list'))
          expect(out.text).toContain('1 live · 1 resumable')
          expect(out.text).toContain('Session refresh-B 0')
          expect(out.text).toContain('Newer sessions')
          expect(out.text).toContain('Older sessions')
          expect(out.text).not.toContain('Session refresh-A 0')
          expect(out.text).toMatch(selected === 'history' ? /▸[^\n]*Session refresh-B 0/
            : selected === 'newer' ? /▸[^\n]*Newer sessions/ : /▸[^\n]*Older sessions/)
        }, WAIT)
        expect(requests[2]).toEqual({ limit: 1, ...(canonical ? { scope: 'stored' } : {}) })
        expect(onResume).not.toHaveBeenCalled()
        const requestsBeforeEnter = requests.length
        stdin.write('\r')
        if (selected === 'history') {
          await vi.waitFor(() => expect(onResume.mock.calls).toEqual([[pageB[0]!.id]]), WAIT)
          expect(requests).toHaveLength(requestsBeforeEnter)
        } else {
          await vi.waitFor(() => expect(requests).toHaveLength(requestsBeforeEnter + 1), WAIT)
          expect(requests.at(-1)).toEqual({
            limit: 1, offset: selected === 'newer' ? 0 : 2, ...(canonical ? { scope: 'stored' } : {})
          })
          expect(onResume).not.toHaveBeenCalled()
          await vi.waitFor(() => expect(out.text).toMatch(
            selected === 'newer' ? /▸[^\n]*Session refresh-A 0/ : /▸[^\n]*Session refresh-B 0/
          ), WAIT)
        }
        expect(onSelect).not.toHaveBeenCalled()

        // A successful explicit refresh must still reset history to the first page.
        refresh = 'success'
        onResume.mockClear()
        out.text = ''
        stdin.write('\u0012')
        await vi.waitFor(() => {
          expect(out.text).toContain('Session refresh-A 0')
          expect(out.text).toContain('Older sessions')
          expect(out.text).not.toContain('Newer sessions')
          expect(out.text).not.toContain('Session refresh-B 0')
          expect(out.text).not.toContain('refresh-offline')
          expect(out.text).not.toContain(messages().pickers.common.invalidResponse('session.list'))
          expect(out.text).toMatch(/▸[^\n]*Session refresh-A 0/)
        }, WAIT)
        expect(requests.at(-1)).toEqual({ limit: 1, ...(canonical ? { scope: 'stored' } : {}) })
        stdin.write('\r')
        await vi.waitFor(() => expect(onResume.mock.calls).toEqual([[pageA[0]!.id]]), WAIT)
      } finally {
        instance.unmount()
      }
    },
    TEST_TIMEOUT
  )

  it.each([true, false])(
    'keeps first history IDs per view across Older/Newer pages and live polls (canonical=%s)',
    async canonical => {
      const shared = makeMockSessions('shared', 1)[0]!
      const liveRow = makeMockSessions('live', 1)[0]!
      const unique0 = makeMockSessions('unique0', 1)[0]!
      const unique1 = makeMockSessions('unique1', 1)[0]!
      const page0 = [
        { ...shared, title: 'SHARED-INITIAL-FIRST' },
        { ...shared, title: 'SHARED-INITIAL-DUPLICATE' },
        { ...liveRow, title: 'HISTORY-LIVE-DUPLICATE' },
        unique0
      ]
      const page1 = [
        { ...shared, title: 'SHARED-OLDER-FIRST' },
        { ...shared, title: 'SHARED-OLDER-DUPLICATE' },
        { ...liveRow, title: 'HISTORY-LIVE-DUPLICATE' },
        unique1
      ]
      let liveRows = [liveRow]
      let livePolls = 0
      const requests: Array<Record<string, unknown>> = []
      const request = vi.fn(async (method: string, params: Record<string, unknown>) => {
        if (method === 'session.active_list' || (canonical && params.scope !== 'stored')) {
          livePolls++
          return { sessions: liveRows }
        }
        requests.push(params)
        const offset = (params.offset as number) ?? 0
        return { has_more: offset === 0, limit: 4, offset, sessions: offset === 0 ? page0 : page1 }
      })
      const stdin = Object.assign(new PassThrough(), { isTTY: true, ref: vi.fn(), setRawMode: vi.fn(), unref: vi.fn() })
      const stdout = Object.assign(new PassThrough(), { columns: 120, isTTY: false, rows: 30 })
      const stderr = new PassThrough()
      const out = { text: '', error: '' }
      stdout.on('data', chunk => { out.text += chunk.toString() })
      stderr.on('data', chunk => { out.error += chunk.toString() })
      const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {})
      const onResume = vi.fn()
      const onSelect = vi.fn()
      const element = (
        <ActiveSessionSwitcher
          currentSessionId={null}
          gw={{ isCanonical: canonical, request } as any}
          onCancel={vi.fn()}
          onClose={vi.fn()}
          onNew={vi.fn()}
          onNewPrompt={vi.fn()}
          onResume={onResume}
          onSelect={onSelect}
          t={DEFAULT_THEME}
        />
      )
      const instance = renderSync(element, {
        patchConsole: false, stderr: stderr as any, stdin: stdin as any, stdout: stdout as any
      })
      try {
        await vi.waitFor(() => {
          expect(out.text).toContain('1 live · 2 resumable')
          expect(out.text).toContain('SHARED-INITIAL-FIRST')
          expect(out.text).not.toContain('SHARED-INITIAL-DUPLICATE')
          expect(out.text).not.toContain('HISTORY-LIVE-DUPLICATE')
        }, WAIT)
        expect(requests).toEqual([{ limit: 200, ...(canonical ? { scope: 'stored' } : {}) }])

        // Select shared by ID; a live poll inserts another live row before it.
        stdin.write('\u001b[B')
        await vi.waitFor(() => expect(out.text).toMatch(/▸[^\n]*SHARED-INITIAL-FIRST/), WAIT)
        const pollsBefore = livePolls
        out.text = ''
        liveRows = [liveRow, unique0]
        await vi.waitFor(() => {
          expect(livePolls).toBeGreaterThan(pollsBefore)
          expect(out.text).toContain('2 live · 1 resumable')
          expect(out.text).toMatch(/▸[^\n]*SHARED-INITIAL-FIRST/)
          expect(out.text).not.toContain('SHARED-INITIAL-DUPLICATE')
        }, WAIT)
        stdin.write('\r')
        await vi.waitFor(() => expect(onResume).toHaveBeenCalledWith(shared.id), WAIT)
        onResume.mockClear()

        // Deduped visible length is one, but Older must advance by raw limit four.
        out.text = ''
        stdin.write('\u001b[B')
        await vi.waitFor(() => expect(out.text).toContain('Page: Enter load page'), WAIT)
        out.text = ''
        stdin.write('\r')
        await vi.waitFor(() => {
          expect(out.text).toContain('SHARED-OLDER-FIRST')
          expect(out.text).toContain('2 live · 2 resumable')
          expect(out.text).toContain('Newer sessions')
          expect(out.text).not.toContain('Older sessions')
          expect(out.text).not.toContain('SHARED-OLDER-DUPLICATE')
          expect(out.text).not.toContain('SHARED-INITIAL-FIRST')
          expect(out.text).not.toContain('HISTORY-LIVE-DUPLICATE')
        }, WAIT)
        expect(requests[1]).toEqual({ limit: 4, offset: 4, ...(canonical ? { scope: 'stored' } : {}) })
        expect(onResume).not.toHaveBeenCalled()
        expect(onSelect).not.toHaveBeenCalled()

        // Overlap across pages is not a permanent seen-ID cache: Newer restores shared.
        out.text = ''
        stdin.write('\u001b[A')
        await vi.waitFor(() => expect(out.text).toContain('Page: Enter load page'), WAIT)
        out.text = ''
        stdin.write('\r')
        await vi.waitFor(() => {
          expect(out.text).toContain('SHARED-INITIAL-FIRST')
          expect(out.text).toContain('2 live · 1 resumable')
          expect(out.text).toContain('Older sessions')
          expect(out.text).not.toContain('Newer sessions')
          expect(out.text).not.toContain('SHARED-INITIAL-DUPLICATE')
          expect(out.text).not.toContain('SHARED-OLDER-FIRST')
        }, WAIT)
        expect(requests[2]).toEqual({ limit: 4, offset: 0, ...(canonical ? { scope: 'stored' } : {}) })
        expect(page0).toHaveLength(4)
        expect(page1).toHaveLength(4)
        expect(out.error).not.toContain('same key')
        expect(consoleError.mock.calls.flat().join(' ')).not.toContain('same key')
        expect(onResume).not.toHaveBeenCalled()
        expect(onSelect).not.toHaveBeenCalled()
      } finally {
        instance.unmount()
        consoleError.mockRestore()
      }
    },
    TEST_TIMEOUT
  )

  it.each(['null', 'non-object', 'missing id', 'projection failure'])(
    'retains good rows and metadata and retries offset 200 after a malformed %s row',
    async kind => {
      let attempts = 0
      let idReads = 0
      let livePolls = 0
      const malformedRow =
        kind === 'null'
          ? null
          : kind === 'non-object'
            ? 'not an object'
            : kind === 'missing id'
              ? { title: 'Missing ID' }
              : {
                  get id() {
                    if (++idReads > 1) {throw new Error('projection failed')}
                    return 'bad-projection'
                  }
                }
      const requests: Array<Record<string, unknown>> = []
      const request = vi.fn(async (_method: string, params: Record<string, unknown>) => {
        if (params.scope !== 'stored') {
          livePolls++
          return { sessions: [] }
        }
        requests.push(params)
        if (params.offset === 200) {
          attempts++
          return attempts === 1
            ? { has_more: false, limit: 1, offset: 200, sessions: [malformedRow] }
            : { has_more: false, limit: 200, offset: 200, sessions: makeMockSessions('good-after', 1) }
        }
        return { has_more: true, limit: 200, offset: 0, sessions: makeMockSessions('good-before', 1) }
      })
      const stdin = Object.assign(new PassThrough(), { isTTY: true, ref: vi.fn(), setRawMode: vi.fn(), unref: vi.fn() })
      const stdout = Object.assign(new PassThrough(), { columns: 120, isTTY: false, rows: 30 })
      const out = { text: '' }
      stdout.on('data', chunk => {
        out.text += chunk.toString()
      })
      const onResume = vi.fn()
      const onSelect = vi.fn()
      const element = (
        <ActiveSessionSwitcher
          currentSessionId={null}
          gw={{ isCanonical: true, request } as any}
          onCancel={vi.fn()}
          onClose={vi.fn()}
          onNew={vi.fn()}
          onNewPrompt={vi.fn()}
          onResume={onResume}
          onSelect={onSelect}
          t={DEFAULT_THEME}
        />
      )
      const instance = renderSync(element, {
        patchConsole: false,
        stderr: new PassThrough() as any,
        stdin: stdin as any,
        stdout: stdout as any
      })
      try {
        await vi.waitFor(() => expect(out.text).toContain('Session good-before 0'), WAIT)
        await vi.waitFor(() => {
          if (!out.text.includes('Page: Enter load page')) {stdin.write('\u001b[B')}
          expect(out.text).toContain('Page: Enter load page')
        }, WAIT)
        out.text = ''
        stdin.write('\r')
        await vi.waitFor(() => {
          expect(attempts).toBe(1)
          expect(out.text).toContain(messages().pickers.common.invalidResponse('session.list'))
          expect(out.text).toContain('retry')
        }, WAIT)
        const pollsBefore = livePolls
        await vi.waitFor(() => expect(livePolls).toBeGreaterThan(pollsBefore), WAIT)
        out.text = ''
        instance.rerender(React.cloneElement(element))
        await vi.waitFor(() => {
          expect(out.text).toContain('Session good-before 0')
          expect(out.text).toContain('Older sessions')
          expect(out.text).not.toContain('Newer sessions')
        }, WAIT)
        stdin.write('\r')
        await vi.waitFor(() => expect(out.text).toContain('Session good-after 0'), WAIT)
        expect(attempts).toBe(2)
        expect(requests.filter(params => params.offset === 200)).toEqual([
          { limit: 200, offset: 200, scope: 'stored' },
          { limit: 200, offset: 200, scope: 'stored' }
        ])
        expect(out.text).toContain('Newer sessions')
        expect(onResume).not.toHaveBeenCalled()
        expect(onSelect).not.toHaveBeenCalled()
      } finally {
        instance.unmount()
      }
    },
    TEST_TIMEOUT
  )

  it(
    'traverses >200 sessions across page boundaries, verifies bounded requests, and stops at exhaustion',
    async () => {
      const page0Sessions = makeMockSessions('store', 200, 0)
      const page1Sessions = makeMockSessions('store', 50, 200)

      const requests: Array<{ method: string; params: Record<string, unknown> }> = []
      const request = vi.fn(async (method: string, params: Record<string, unknown>) => {
        requests.push({ method, params })

        if (method === 'session.list') {
          if (params.scope === 'stored') {
            const offset = (params.offset as number) ?? 0

            if (offset === 0) {
              return {
                has_more: true,
                limit: 200,
                offset: 0,
                scope: 'stored',
                sessions: page0Sessions
              }
            }

            if (offset === 200) {
              return {
                has_more: false,
                limit: 200,
                offset: 200,
                scope: 'stored',
                sessions: page1Sessions
              }
            }
          }

          // Canonical live sessions
          return { scope: 'live', sessions: [] }
        }

        throw new Error(`unexpected method: ${method}`)
      })

      const stdin = Object.assign(new PassThrough(), {
        isTTY: true,
        ref: vi.fn(),
        setRawMode: vi.fn(),
        unref: vi.fn()
      })
      const stdout = Object.assign(new PassThrough(), { columns: 120, isTTY: false, rows: 30 })
      const out = { text: '' }

      stdout.on('data', chunk => {
        out.text += chunk.toString()
      })

      const onResume = vi.fn()
      const onSelect = vi.fn()

      const element = (
        <ActiveSessionSwitcher
          currentSessionId={null}
          gw={{ isCanonical: true, request } as any}
          onCancel={vi.fn()}
          onClose={vi.fn()}
          onNew={vi.fn()}
          onNewPrompt={vi.fn()}
          onResume={onResume}
          onSelect={onSelect}
          t={DEFAULT_THEME}
        />
      )

      const instance = renderSync(element, {
        patchConsole: false,
        stderr: new PassThrough() as any,
        stdin: stdin as any,
        stdout: stdout as any
      })

      try {
        // Initial request omitted offset!
        await vi.waitFor(() => {
          const storedCalls = requests.filter(r => r.method === 'session.list' && r.params.scope === 'stored')
          expect(storedCalls.length).toBeGreaterThan(0)
          expect(storedCalls[0]!.params).toEqual({ limit: 200, scope: 'stored' })
          expect(storedCalls[0]!.params.offset).toBeUndefined()
        }, WAIT)

        await vi.waitFor(() => expect(out.text).toContain('200 resumable'), WAIT)

        // Scroll down quickly to Older sessions
        await vi.waitFor(() => {
          if (!out.text.includes('Page: Enter load page')) {
            stdin.write('\u001B[B'.repeat(25))
          }
          expect(out.text).toContain('Page: Enter load page')
        }, WAIT)

        // Now that the Older sessions control row is selected, press Enter
        await vi.waitFor(() => {
          if (!requests.some(r => r.params.offset === 200)) {
            stdin.write('\r')
          }
          expect(requests.some(r => r.params.offset === 200)).toBe(true)
        }, WAIT)

        out.text = ''

        // Verify that control row did NOT resume or select
        expect(onResume).not.toHaveBeenCalled()
        expect(onSelect).not.toHaveBeenCalled()

        // Page 1 is now committed!
        await vi.waitFor(() => {
          instance.rerender(React.cloneElement(element))
          expect(out.text).toContain('Newer sessions')
          // Page 1 has has_more: false -> Older sessions must NOT be present
          expect(out.text).not.toContain('Older sessions')
          // Explicit page fallback: first history row of page 1 is selected
          expect(out.text).toContain('Session store 200')
        }, WAIT)

        // Now select Newer sessions (at row 1, right above Session store 200)
        await vi.waitFor(() => {
          if (!out.text.includes('▲ newer') || !out.text.includes('Page: Enter load page')) {
            stdin.write('\u001B[A')
          }
          expect(out.text).toContain('Page: Enter load page')
        }, WAIT)

        await vi.waitFor(() => {
          if (!requests.some(r => r.params.offset === 0)) {
            stdin.write('\r')
          }
          expect(requests.some(r => r.params.offset === 0)).toBe(true)
        }, WAIT)

        // Page 0 restored
        await vi.waitFor(() => {
          instance.rerender(React.cloneElement(element))
          expect(out.text).toContain('200 resumable')
        }, WAIT)
      } finally {
        instance.unmount()
      }
    },
    TEST_TIMEOUT
  )

  it(
    'displays older gateway notice and disables traversal when metadata is omitted on initial request',
    async () => {
      const stored = makeMockSessions('old', 5)
      const request = vi.fn(async (method: string, params: Record<string, unknown>) => {
        if (method === 'session.list') {
          if (params.scope === 'stored') {
            // Older gateway payload: no offset, limit, or has_more
            return {
              scope: 'stored',
              sessions: stored
            }
          }

          return { scope: 'live', sessions: [] }
        }

        throw new Error(`unexpected method: ${method}`)
      })

      const stdin = Object.assign(new PassThrough(), { isTTY: true, setRawMode: vi.fn(), ref: vi.fn(), unref: vi.fn() })
      const stdout = Object.assign(new PassThrough(), { columns: 120, rows: 30, isTTY: false })
      let text = ''

      stdout.on('data', chunk => {
        text += chunk.toString()
      })

      const element = (
        <ActiveSessionSwitcher
          currentSessionId={null}
          gw={{ isCanonical: true, request } as any}
          onCancel={vi.fn()}
          onClose={vi.fn()}
          onNew={vi.fn()}
          onNewPrompt={vi.fn()}
          onResume={vi.fn()}
          onSelect={vi.fn()}
          t={DEFAULT_THEME}
        />
      )

      const instance = renderSync(element, {
        patchConsole: false,
        stderr: new PassThrough() as any,
        stdin: stdin as any,
        stdout: stdout as any
      })

      try {
        await vi.waitFor(() => {
          expect(text).toContain('Session old 0')
          expect(text).toContain(messages().pickers.session.pagination.olderGatewayNotice)
          expect(text).not.toContain('Older sessions')
          expect(text).not.toContain('Newer sessions')
        }, WAIT)
      } finally {
        instance.unmount()
      }
    },
    TEST_TIMEOUT
  )

  it(
    'pending request lock prevents concurrent page requests during in-flight fetch',
    async () => {
      let resolvePage1!: (val: unknown) => void
      const page1Promise = new Promise(resolve => {
        resolvePage1 = resolve
      })

      const requests: Array<{ offset?: unknown }> = []
      const request = vi.fn(async (method: string, params: Record<string, unknown>) => {
        if (method === 'session.list' && params.scope === 'stored') {
          requests.push({ offset: params.offset })

          if (params.offset === 200) {
            return page1Promise
          }

          return {
            has_more: true,
            limit: 200,
            offset: 0,
            scope: 'stored',
            sessions: makeMockSessions('p0', 5)
          }
        }

        return { scope: 'live', sessions: [] }
      })

      const stdin = Object.assign(new PassThrough(), { isTTY: true, setRawMode: vi.fn(), ref: vi.fn(), unref: vi.fn() })
      const stdout = Object.assign(new PassThrough(), { columns: 120, rows: 30, isTTY: false })
      let text = ''

      stdout.on('data', chunk => {
        text += chunk.toString()
      })

      const element = (
        <ActiveSessionSwitcher
          currentSessionId={null}
          gw={{ isCanonical: true, request } as any}
          onCancel={vi.fn()}
          onClose={vi.fn()}
          onNew={vi.fn()}
          onNewPrompt={vi.fn()}
          onResume={vi.fn()}
          onSelect={vi.fn()}
          t={DEFAULT_THEME}
        />
      )

      const instance = renderSync(element, {
        patchConsole: false,
        stderr: new PassThrough() as any,
        stdin: stdin as any,
        stdout: stdout as any
      })

      try {
        await vi.waitFor(() => expect(text).toContain('Older sessions'), WAIT)

        // Navigate down to Older sessions row (row 6)
        await vi.waitFor(() => {
          if (!text.includes('Page: Enter load page')) {
            stdin.write('\u001B[B')
          }
          expect(text).toContain('Page: Enter load page')
        }, WAIT)

        // First Enter triggers request
        stdin.write('\r')

        await vi.waitFor(() => {
          const calls200 = requests.filter(r => r.offset === 200)
          expect(calls200.length).toBe(1) // In-flight request started
        }, WAIT)

        // Additional Enters while pending must be ignored by the pending lock
        stdin.write('\r')
        stdin.write('\r')
        await new Promise(r => setTimeout(r, 50))
        expect(requests.filter(r => r.offset === 200).length).toBe(1)

        // Resolve page 1
        resolvePage1({
          has_more: false,
          limit: 200,
          offset: 200,
          scope: 'stored',
          sessions: makeMockSessions('p1', 5, 200)
        })

        await vi.waitFor(() => {
          instance.rerender(React.cloneElement(element))
          expect(text).toContain('Newer sessions')
        }, WAIT)
      } finally {
        instance.unmount()
      }
    },
    TEST_TIMEOUT
  )

  it(
    'retains current page on error and retries the exact failed offset successfully',
    async () => {
      let failNext = true
      const requests: Array<Record<string, unknown>> = []

      const request = vi.fn(async (method: string, params: Record<string, unknown>) => {
        if (method === 'session.list' && params.scope === 'stored') {
          requests.push(params)

          if (params.offset === 200) {
            if (failNext) {
              failNext = false
              throw new Error('simulated network fault')
            }

            return {
              has_more: false,
              limit: 200,
              offset: 200,
              scope: 'stored',
              sessions: makeMockSessions('p1', 5, 200)
            }
          }

          return {
            has_more: true,
            limit: 200,
            offset: 0,
            scope: 'stored',
            sessions: makeMockSessions('p0', 5)
          }
        }

        return { scope: 'live', sessions: [] }
      })

      const stdin = Object.assign(new PassThrough(), { isTTY: true, setRawMode: vi.fn(), ref: vi.fn(), unref: vi.fn() })
      const stdout = Object.assign(new PassThrough(), { columns: 120, rows: 30, isTTY: false })
      let text = ''

      stdout.on('data', chunk => {
        text += chunk.toString()
      })

      const element = (
        <ActiveSessionSwitcher
          currentSessionId={null}
          gw={{ isCanonical: true, request } as any}
          onCancel={vi.fn()}
          onClose={vi.fn()}
          onNew={vi.fn()}
          onNewPrompt={vi.fn()}
          onResume={vi.fn()}
          onSelect={vi.fn()}
          t={DEFAULT_THEME}
        />
      )

      const instance = renderSync(element, {
        patchConsole: false,
        stderr: new PassThrough() as any,
        stdin: stdin as any,
        stdout: stdout as any
      })

      try {
        await vi.waitFor(() => expect(text).toContain('Older sessions'), WAIT)

        // Navigate to Older sessions row
        await vi.waitFor(() => {
          if (!text.includes('Page: Enter load page')) {
            stdin.write('\u001B[B')
          }
          expect(text).toContain('Page: Enter load page')
        }, WAIT)

        // Press Enter
        await vi.waitFor(() => {
          if (!requests.some(r => r.offset === 200)) {
            stdin.write('\r')
          }
          expect(requests.some(r => r.offset === 200)).toBe(true)
        }, WAIT)

        // Error message appears, last good page remains visible
        await vi.waitFor(() => {
          instance.rerender(React.cloneElement(element))
          expect(text).toContain('simulated network fault')
          expect(text).toContain('Session p0 0')
        }, WAIT)

        // Press Enter again to retry exact same offset 200
        await vi.waitFor(() => {
          const calls200 = requests.filter(r => r.offset === 200)
          if (calls200.length < 2) {
            stdin.write('\r')
          }
          expect(calls200.length).toBe(2)
        }, WAIT)

        await vi.waitFor(() => {
          instance.rerender(React.cloneElement(element))
          expect(text).toContain('Session p1 200')
        }, WAIT)
      } finally {
        instance.unmount()
      }
    },
    TEST_TIMEOUT
  )

  it(
    'resets pagination state and discards delayed old responses when profile/filter changes',
    async () => {
      let delayedResolve!: (val: unknown) => void
      const delayedPromise = new Promise(resolve => {
        delayedResolve = resolve
      })

      const requests: Array<Record<string, unknown>> = []
      const request = vi.fn(async (method: string, params: Record<string, unknown>) => {
        if (method === 'session.list' && params.scope === 'stored') {
          requests.push(params)

          if (params.offset === 200) {
            return delayedPromise
          }

          return {
            has_more: true,
            limit: 200,
            offset: 0,
            scope: 'stored',
            sessions: makeMockSessions('p0', 5)
          }
        }

        return { scope: 'live', sessions: [] }
      })

      const stdin = Object.assign(new PassThrough(), { isTTY: true, setRawMode: vi.fn(), ref: vi.fn(), unref: vi.fn() })
      const stdout = Object.assign(new PassThrough(), { columns: 120, rows: 30, isTTY: false })
      let text = ''

      stdout.on('data', chunk => {
        text += chunk.toString()
      })

      const element = (
        <ActiveSessionSwitcher
          currentSessionId={null}
          filter=""
          gw={{ isCanonical: true, request } as any}
          onCancel={vi.fn()}
          onClose={vi.fn()}
          onNew={vi.fn()}
          onNewPrompt={vi.fn()}
          onResume={vi.fn()}
          onSelect={vi.fn()}
          profile="work"
          t={DEFAULT_THEME}
        />
      )

      const instance = renderSync(element, {
        patchConsole: false,
        stderr: new PassThrough() as any,
        stdin: stdin as any,
        stdout: stdout as any
      })

      try {
        await vi.waitFor(() => expect(text).toContain('Older sessions'), WAIT)

        // Navigate down to Older sessions row
        await vi.waitFor(() => {
          if (!text.includes('Page: Enter load page')) {
            stdin.write('\u001B[B')
          }
          expect(text).toContain('Page: Enter load page')
        }, WAIT)

        // Trigger request for offset 200
        await vi.waitFor(() => {
          if (!requests.some(r => r.offset === 200)) {
            stdin.write('\r')
          }
          expect(requests.some(r => r.offset === 200)).toBe(true)
        }, WAIT)

        // Now profile changes to 'home'!
        instance.rerender(
          <ActiveSessionSwitcher
            currentSessionId={null}
            filter=""
            gw={{ isCanonical: true, request } as any}
            onCancel={vi.fn()}
            onClose={vi.fn()}
            onNew={vi.fn()}
            onNewPrompt={vi.fn()}
            onResume={vi.fn()}
            onSelect={vi.fn()}
            profile="home"
            t={DEFAULT_THEME}
          />
        )

        // The delayed promise from 'work' profile now resolves late
        delayedResolve({
          has_more: false,
          limit: 200,
          offset: 200,
          scope: 'stored',
          sessions: makeMockSessions('stale-work', 50, 200)
        })

        // Stale work sessions must NOT appear in output!
        await new Promise(r => setTimeout(r, 100))
        instance.rerender(
          <ActiveSessionSwitcher
            currentSessionId={null}
            filter=""
            gw={{ isCanonical: true, request } as any}
            onCancel={vi.fn()}
            onClose={vi.fn()}
            onNew={vi.fn()}
            onNewPrompt={vi.fn()}
            onResume={vi.fn()}
            onSelect={vi.fn()}
            profile="home"
            t={DEFAULT_THEME}
          />
        )
        expect(text).not.toContain('stale-work')
      } finally {
        instance.unmount()
      }
    },
    TEST_TIMEOUT
  )

  it.each([false, true])(
    'live dedup uses raw progression even when history is visually empty (%s)',
    async allLive => {
      // 5 sessions on page 0, but 2 of them are also returned in live sessions
      const live: SessionActiveItem[] = [
        {
          current: false,
          id: 'dup-1',
          last_active: 1,
          message_count: 1,
          model: 'm',
          preview: 'p',
          session_key: 'k',
          started_at: 1,
          status: 'idle',
          title: 'Live Dup 1'
        }
      ]
      const rawStored = [
        {
          id: 'dup-1',
          message_count: 1,
          model: 'm',
          preview: 'p',
          running: false,
          session_id: 'dup-1',
          source: 'tui',
          started_at: 1,
          title: 'Stored Dup 1'
        },
        {
          id: 'unique-1',
          message_count: 1,
          model: 'm',
          preview: 'p',
          running: false,
          session_id: 'unique-1',
          source: 'tui',
          started_at: 1,
          title: 'Unique Stored 1'
        }
      ]

      const request = vi.fn(async (method: string, params: Record<string, unknown>) => {
        if (method === 'session.list') {
          if (params.scope === 'stored') {
            return {
              has_more: true,
              limit: 200,
              offset: 0,
              scope: 'stored',
              sessions: allLive ? rawStored.slice(0, 1) : rawStored
            }
          }

          return {
            scope: 'live',
            sessions: live
          }
        }

        throw new Error(`unexpected: ${method}`)
      })

      const stdin = Object.assign(new PassThrough(), { isTTY: true, setRawMode: vi.fn(), ref: vi.fn(), unref: vi.fn() })
      const stdout = Object.assign(new PassThrough(), { columns: 120, rows: 30, isTTY: false })
      let text = ''

      stdout.on('data', chunk => {
        text += chunk.toString()
      })

      const element = (
        <ActiveSessionSwitcher
          currentSessionId={null}
          gw={{ isCanonical: true, request } as any}
          onCancel={vi.fn()}
          onClose={vi.fn()}
          onNew={vi.fn()}
          onNewPrompt={vi.fn()}
          onResume={vi.fn()}
          onSelect={vi.fn()}
          t={DEFAULT_THEME}
        />
      )

      const instance = renderSync(element, {
        patchConsole: false,
        stderr: new PassThrough() as any,
        stdin: stdin as any,
        stdout: stdout as any
      })

      try {
        await vi.waitFor(() => {
          if (!allLive) {expect(text).toContain('Unique Stored 1')}
          expect(text).toContain('Older sessions')
        }, WAIT)
        await vi.waitFor(() => {
          if (!text.includes('Page: Enter load page')) {stdin.write('\u001B[B')}
          expect(text).toContain('Page: Enter load page')
        }, WAIT)
        stdin.write('\r')
        await vi.waitFor(() => {
          expect(request.mock.calls.some(([, params]) => params.offset === 200)).toBe(true)
        }, WAIT)
      } finally {
        instance.unmount()
      }
    },
    TEST_TIMEOUT
  )

  it(
    'allows quiet live polls during a pending page navigation without clearing or corrupting page state',
    async () => {
      const page0Sessions = makeMockSessions('p0', 200, 0)
      const page1Sessions = makeMockSessions('p1', 50, 200)
      let resolvePage1!: (val: unknown) => void
      const page1Promise = new Promise(res => {
        resolvePage1 = res
      })

      let livePollCount = 0

      const request = vi.fn(async (method: string, params: Record<string, unknown>) => {
        if (method === 'session.list') {
          if (params.scope === 'stored') {
            const offset = (params.offset as number) ?? 0

            if (offset === 0) {
              return { has_more: true, limit: 200, offset: 0, scope: 'stored', sessions: page0Sessions }
            }

            if (offset === 200) {
              await page1Promise

              return { has_more: false, limit: 200, offset: 200, scope: 'stored', sessions: page1Sessions }
            }
          }

          livePollCount++

          return { scope: 'live', sessions: [] }
        }

        throw new Error(`unexpected method: ${method}`)
      })

      const stdin = Object.assign(new PassThrough(), { isTTY: true, ref: vi.fn(), setRawMode: vi.fn(), unref: vi.fn() })
      const stdout = Object.assign(new PassThrough(), { columns: 120, isTTY: false, rows: 30 })
      const out = { text: '' }
      stdout.on('data', chunk => {
        out.text += chunk.toString()
      })

      const instance = renderSync(
        <ActiveSessionSwitcher
          currentSessionId={null}
          gw={{ isCanonical: true, request } as any}
          onCancel={vi.fn()}
          onClose={vi.fn()}
          onNew={vi.fn()}
          onNewPrompt={vi.fn()}
          onResume={vi.fn()}
          onSelect={vi.fn()}
          t={DEFAULT_THEME}
        />,
        { patchConsole: false, stderr: new PassThrough() as any, stdin: stdin as any, stdout: stdout as any }
      )

      try {
        await vi.waitFor(() => expect(out.text).toContain('200 resumable'), WAIT)

        // Select Older sessions row and press Enter to initiate page navigation
        await vi.waitFor(() => {
          if (!out.text.includes('Page: Enter load page')) {
            stdin.write('\u001B[B'.repeat(25))
          }
          expect(out.text).toContain('Page: Enter load page')
        }, WAIT)

        stdin.write('\r')

        // Wait until page request for offset 200 is initiated and pending
        await vi.waitFor(() => {
          expect(request).toHaveBeenCalledWith('session.list', expect.objectContaining({ offset: 200 }))
          expect(out.text).toContain('loading page…')
        }, WAIT)

        // While page 1 is pending, trigger or wait for a quiet live poll
        const initialLivePolls = livePollCount
        await vi.waitFor(() => {
          expect(livePollCount).toBeGreaterThan(initialLivePolls)
        }, { timeout: 4000 })

        // Now resolve the pending page 1 promise
        resolvePage1({ has_more: false, limit: 200, offset: 200, scope: 'stored', sessions: page1Sessions })

        // Page 1 must successfully render!
        await vi.waitFor(() => {
          expect(out.text).toContain('Session p1 200')
          expect(out.text).toContain('Newer sessions')
        }, WAIT)
      } finally {
        instance.unmount()
      }
    },
    TEST_TIMEOUT
  )

  it(
    'immediately resets visible rows and raw history when gateway connection changes',
    async () => {
      const gw1Sessions = makeMockSessions('gw1', 5, 0)
      const gw2Sessions = makeMockSessions('gw2', 5, 0)

      const request1 = vi.fn(async (method: string, params: Record<string, unknown>) => {
        if (method === 'session.list') {
          if (params.scope === 'stored') {
            return { has_more: false, limit: 200, offset: 0, scope: 'stored', sessions: gw1Sessions }
          }

          return { scope: 'live', sessions: [] }
        }

        throw new Error(`unexpected: ${method}`)
      })

      const request2 = vi.fn(async (method: string, params: Record<string, unknown>) => {
        if (method === 'session.list') {
          if (params.scope === 'stored') {
            return { has_more: false, limit: 200, offset: 0, scope: 'stored', sessions: gw2Sessions }
          }

          return { scope: 'live', sessions: [] }
        }

        throw new Error(`unexpected: ${method}`)
      })

      const stdin = Object.assign(new PassThrough(), { isTTY: true, ref: vi.fn(), setRawMode: vi.fn(), unref: vi.fn() })
      const stdout = Object.assign(new PassThrough(), { columns: 120, isTTY: false, rows: 30 })
      const out = { text: '' }
      stdout.on('data', chunk => {
        out.text += chunk.toString()
      })

      const gw1 = { isCanonical: true, request: request1 } as any
      const gw2 = { isCanonical: true, request: request2 } as any

      const instance = renderSync(
        <ActiveSessionSwitcher
          currentSessionId={null}
          gw={gw1}
          onCancel={vi.fn()}
          onClose={vi.fn()}
          onNew={vi.fn()}
          onNewPrompt={vi.fn()}
          onResume={vi.fn()}
          onSelect={vi.fn()}
          t={DEFAULT_THEME}
        />,
        { patchConsole: false, stderr: new PassThrough() as any, stdin: stdin as any, stdout: stdout as any }
      )

      try {
        await vi.waitFor(() => expect(out.text).toContain('Session gw1 0'), WAIT)

        // Clear output buffer so we verify the new render does not contain old connection rows
        out.text = ''

        // Change connection to gw2
        instance.rerender(
          <ActiveSessionSwitcher
            currentSessionId={null}
            gw={gw2}
            onCancel={vi.fn()}
            onClose={vi.fn()}
            onNew={vi.fn()}
            onNewPrompt={vi.fn()}
            onResume={vi.fn()}
            onSelect={vi.fn()}
            t={DEFAULT_THEME}
          />
        )

        // Old rows from gw1 must disappear and gw2 rows must appear
        await vi.waitFor(() => {
          expect(out.text).toContain('Session gw2 0')
          expect(out.text).not.toContain('Session gw1 0')
        }, WAIT)
      } finally {
        instance.unmount()
      }
    },
    TEST_TIMEOUT
  )

  it(
    'displays inline error on malformed page response and successfully retries on Enter',
    async () => {
      const page0Sessions = makeMockSessions('m0', 200, 0)
      const page1Sessions = makeMockSessions('m1', 20, 200)
      let attempt = 0

      const request = vi.fn(async (method: string, params: Record<string, unknown>) => {
        if (method === 'session.list') {
          if (params.scope === 'stored') {
            const offset = (params.offset as number) ?? 0

            if (offset === 0) {
              return { has_more: true, limit: 200, offset: 0, scope: 'stored', sessions: page0Sessions }
            }

            if (offset === 200) {
              attempt++
              if (attempt === 1) {
                // Return malformed response: empty sessions but has_more: true
                return { has_more: true, limit: 200, offset: 200, scope: 'stored', sessions: [] }
              }

              // Second attempt returns valid page
              return { has_more: false, limit: 200, offset: 200, scope: 'stored', sessions: page1Sessions }
            }
          }

          return { scope: 'live', sessions: [] }
        }

        throw new Error(`unexpected method: ${method}`)
      })

      const stdin = Object.assign(new PassThrough(), { isTTY: true, ref: vi.fn(), setRawMode: vi.fn(), unref: vi.fn() })
      const stdout = Object.assign(new PassThrough(), { columns: 120, isTTY: false, rows: 30 })
      const out = { text: '' }
      stdout.on('data', chunk => {
        out.text += chunk.toString()
      })

      const instance = renderSync(
        <ActiveSessionSwitcher
          currentSessionId={null}
          gw={{ isCanonical: true, request } as any}
          onCancel={vi.fn()}
          onClose={vi.fn()}
          onNew={vi.fn()}
          onNewPrompt={vi.fn()}
          onResume={vi.fn()}
          onSelect={vi.fn()}
          t={DEFAULT_THEME}
        />,
        { patchConsole: false, stderr: new PassThrough() as any, stdin: stdin as any, stdout: stdout as any }
      )

      try {
        await vi.waitFor(() => expect(out.text).toContain('200 resumable'), WAIT)

        // Navigate down to Older sessions
        await vi.waitFor(() => {
          if (!out.text.includes('Page: Enter load page')) {
            stdin.write('\u001B[B'.repeat(25))
          }
          expect(out.text).toContain('Page: Enter load page')
        }, WAIT)

        // Press Enter to trigger first attempt (which will fail due to malformed payload)
        stdin.write('\r')

        await vi.waitFor(() => {
          expect(out.text).toContain('retry')
        }, WAIT)

        // Press Enter again to retry
        stdin.write('\r')

        // Must succeed and display valid page
        await vi.waitFor(() => {
          expect(out.text).toContain('Session m1 200')
        }, WAIT)
      } finally {
        instance.unmount()
      }
    },
    TEST_TIMEOUT
  )

  it(
    'cleans up gracefully on unmount during in-flight requests without errors',
    async () => {
      let pendingRequest = true
      const request = vi.fn(async () => {
        await new Promise(resolve => setTimeout(resolve, 100))
        if (!pendingRequest) {
          return { sessions: [] }
        }

        return { has_more: true, limit: 200, offset: 0, scope: 'stored', sessions: makeMockSessions('u', 10) }
      })

      const stdin = Object.assign(new PassThrough(), { isTTY: true, ref: vi.fn(), setRawMode: vi.fn(), unref: vi.fn() })
      const stdout = Object.assign(new PassThrough(), { columns: 120, isTTY: false, rows: 30 })

      const instance = renderSync(
        <ActiveSessionSwitcher
          currentSessionId={null}
          gw={{ isCanonical: true, request } as any}
          onCancel={vi.fn()}
          onClose={vi.fn()}
          onNew={vi.fn()}
          onNewPrompt={vi.fn()}
          onResume={vi.fn()}
          onSelect={vi.fn()}
          t={DEFAULT_THEME}
        />,
        { patchConsole: false, stderr: new PassThrough() as any, stdin: stdin as any, stdout: stdout as any }
      )

      // Immediately unmount while in flight
      instance.unmount()
      pendingRequest = false

      await new Promise(resolve => setTimeout(resolve, 200))
    },
    TEST_TIMEOUT
  )
})
