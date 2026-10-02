import {
  closeSync,
  mkdirSync,
  mkdtempSync,
  openSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync
} from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { writeActiveSessionFile } from '../app/activeSessionFile.js'
import { turnController } from '../app/turnController.js'
import { getTurnState, resetTurnState } from '../app/turnStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import {
  hydrateLiveSessionInflight,
  liveSessionInflightMessages,
  signalFreshSessionBoundary
} from '../app/useSessionLifecycle.js'

describe('fresh session boundary', () => {
  it('signals only when a live session is replaced by a different session', () => {
    const onFreshSessionStarted = vi.fn()

    expect(signalFreshSessionBoundary('old-session', 'new-session', onFreshSessionStarted)).toBe(true)
    expect(signalFreshSessionBoundary(null, 'first-session', onFreshSessionStarted)).toBe(false)
    expect(signalFreshSessionBoundary('same-session', 'same-session', onFreshSessionStarted)).toBe(false)
    expect(signalFreshSessionBoundary('old-session', null, onFreshSessionStarted)).toBe(false)
    expect(signalFreshSessionBoundary('old-session', 'new-session')).toBe(false)
    expect(onFreshSessionStarted).toHaveBeenCalledOnce()
    expect(onFreshSessionStarted).toHaveBeenCalledWith('new-session')
  })
})

describe('writeActiveSessionFile', () => {
  let dir = ''

  afterEach(() => {
    if (dir) {
      rmSync(dir, { force: true, recursive: true })
      dir = ''
    }
  })

  it('writes the actual resumed session id for the shell exit summary', () => {
    dir = mkdtempSync(join(tmpdir(), 'hermes-tui-active-'))
    const path = join(dir, 'active.json')

    writeActiveSessionFile('actual_session', path)

    expect(JSON.parse(readFileSync(path, 'utf8'))).toEqual({ session_id: 'actual_session' })
  })

  it('replaces the breadcrumb without changing an open reader or leaving staging files', () => {
    dir = mkdtempSync(join(tmpdir(), 'hermes-tui-active-'))
    const path = join(dir, 'active.json')
    const previous = JSON.stringify({ session_id: 'before-compression' })
    writeFileSync(path, previous, { mode: 0o644 })
    const reader = openSync(path, 'r')

    try {
      writeActiveSessionFile('after-compression', path)

      expect(JSON.parse(readFileSync(path, 'utf8'))).toEqual({ session_id: 'after-compression' })
      // An existing reader must keep the complete old JSON, not the truncated/new file.
      expect(readFileSync(reader, 'utf8')).toBe(previous)

      if (process.platform !== 'win32') {
        expect(statSync(path).mode & 0o777).toBe(0o600)
      }

      expect(readdirSync(dir)).toEqual(['active.json'])
    } finally {
      closeSync(reader)
    }

    writeActiveSessionFile(null, path)
    writeActiveSessionFile('', path)
    expect(JSON.parse(readFileSync(path, 'utf8'))).toEqual({ session_id: 'after-compression' })

    // A real rename failure must neither escape nor leave a partial replacement behind.
    const blocked = join(dir, 'blocked')
    mkdirSync(blocked)
    writeFileSync(join(blocked, 'keep'), previous)
    expect(() => writeActiveSessionFile('cannot-replace-directory', blocked)).not.toThrow()
    expect(readFileSync(join(blocked, 'keep'), 'utf8')).toBe(previous)
    expect(readdirSync(dir).sort()).toEqual(['active.json', 'blocked'])
    expect(() => writeActiveSessionFile('missing-parent', join(dir, 'missing', 'active.json'))).not.toThrow()
  })
})

describe('live session activation in-flight state', () => {
  beforeEach(() => {
    resetUiState()
    resetTurnState()
    turnController.fullReset()
    patchUiState({ streaming: true })
  })

  it('keeps the in-flight user prompt in history and hydrates partial assistant text', () => {
    const inflight = { assistant: 'partial answer', streaming: true, user: 'write a long answer' }

    expect(liveSessionInflightMessages(inflight)).toEqual([{ role: 'user', text: 'write a long answer' }])

    hydrateLiveSessionInflight(inflight)

    expect(turnController.bufRef).toBe('partial answer')
    expect(getTurnState().streaming).toBe('partial answer')
  })

  it('preserves synthetic turn display metadata while rebuilding live history', () => {
    const inflight = {
      assistant: '',
      display_kind: 'process_complete',
      display_metadata: { display_text: 'Finished syncing the workspace' },
      streaming: true,
      user: 'process completed'
    }

    expect(liveSessionInflightMessages(inflight)).toEqual([
      {
        kind: 'event',
        role: 'system',
        text: 'Finished syncing the workspace'
      }
    ])
  })

  it('ignores empty in-flight payloads', () => {
    expect(liveSessionInflightMessages({ assistant: '', streaming: false, user: '   ' })).toEqual([])

    hydrateLiveSessionInflight({ assistant: '', streaming: false, user: '' })

    expect(turnController.bufRef).toBe('')
    expect(getTurnState().streaming).toBe('')
  })
})
