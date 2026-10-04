import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const execFile = vi.hoisted(() => vi.fn((_file, _args, _options, callback) => {
  callback(null, { stdout: 'feature\n', stderr: '' })
}))
vi.mock('node:child_process', () => ({ execFile }))

import { shouldPollGitBranch, useGitBranch } from '../hooks/useGitBranch.js'

const flush = async () => {
  await new Promise(resolve => setImmediate(resolve))
  await new Promise(resolve => setImmediate(resolve))
}

function Probe({ enabled, cwd }: { enabled: boolean; cwd: string }) {
  useGitBranch(cwd, enabled)
  return null
}

afterEach(() => {
  vi.restoreAllMocks()
  execFile.mockClear()
})

describe('useGitBranch lifecycle', () => {
  it('spawns nothing while disabled, starts on enable, and stops on disable at the same cwd', async () => {
    const stdout = Object.assign(new PassThrough(), { columns: 80, rows: 24, isTTY: false })
    const stdin = Object.assign(new PassThrough(), { isTTY: false })
    const stderr = Object.assign(new PassThrough(), { isTTY: false })
    const intervals = vi.spyOn(globalThis, 'setInterval')
    const clear = vi.spyOn(globalThis, 'clearInterval')
    const instance = renderSync(<Probe cwd="/hook-lifecycle" enabled={false} />, {
      patchConsole: false,
      stdout: stdout as unknown as NodeJS.WriteStream,
      stdin: stdin as unknown as NodeJS.ReadStream,
      stderr: stderr as unknown as NodeJS.WriteStream
    })
    try {
      await flush()
      expect(execFile).not.toHaveBeenCalled()
      expect(intervals).not.toHaveBeenCalled()
      instance.rerender(<Probe cwd="/hook-lifecycle" enabled />)
      await flush()
      expect(execFile).toHaveBeenCalledTimes(1)
      expect(execFile.mock.calls[0]?.slice(0, 3)).toEqual([
        'git', ['-C', '/hook-lifecycle', 'rev-parse', '--abbrev-ref', 'HEAD'], { timeout: 500 }
      ])
      const timer = intervals.mock.results.at(-1)?.value
      instance.rerender(<Probe cwd="/hook-lifecycle" enabled={false} />)
      await flush()
      expect(clear).toHaveBeenCalledWith(timer)
      expect(execFile).toHaveBeenCalledTimes(1)
      instance.rerender(<Probe cwd="/hook-lifecycle" enabled />)
      await flush()
      // Re-enabling installs the effect again while retaining its fresh cache.
      expect(intervals).toHaveBeenCalledTimes(2)
      expect(execFile).toHaveBeenCalledTimes(1)
    } finally {
      instance.unmount()
      instance.cleanup()
    }
  })

  it('keeps cwd polling enabled for a stale title without an active session', () => {
    const title = 'stale title'
    const sid = null
    expect(shouldPollGitBranch('top', sid ? title : '', null)).toBe(true)
    expect(shouldPollGitBranch('top', title, null)).toBe(false)
  })
})
