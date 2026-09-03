import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { describe, expect, it, vi } from 'vitest'

const pickerHarness = vi.hoisted(() => ({
  applySkinPreview: vi.fn(),
  handler: undefined as undefined | ((input: string, key: Record<string, boolean>) => void),
  persistedRevision: 0,
  restorePersistedSkin: vi.fn()
}))

vi.mock('@hermes/ink', async importOriginal => {
  const mod = await importOriginal<Record<string, unknown>>()

  return {
    ...mod,
    useInput: (handler: (input: string, key: Record<string, boolean>) => void) => {
      pickerHarness.handler = handler
    }
  }
})

vi.mock('../app/createGatewayEventHandler.js', () => ({
  applySkinPreview: pickerHarness.applySkinPreview,
  getPersistedSkinRevision: () => pickerHarness.persistedRevision,
  restorePersistedSkin: pickerHarness.restorePersistedSkin
}))

import { filterSkinOptions, SKIN_PREVIEW_DEBOUNCE_MS, skinAgentLabel, SkinPicker } from '../components/skinPicker.js'
import { DEFAULT_THEME } from '../theme.js'

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

function mountPicker(gw: unknown, onClose = vi.fn()) {
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()

  Object.assign(stdout, { columns: 100, isTTY: false, rows: 40 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  pickerHarness.handler = undefined

  const element = React.createElement(SkinPicker, { gw, onClose, t: DEFAULT_THEME } as never)

  const instance = renderSync(element, {
    patchConsole: false,
    stderr: stderr as unknown as NodeJS.WriteStream,
    stdin: stdin as unknown as NodeJS.ReadStream,
    stdout: stdout as unknown as NodeJS.WriteStream
  })

  return {
    cleanup: () => {
      instance.unmount()
      instance.cleanup()
    },
    onClose,
    rerender: () => instance.rerender(element)
  }
}

const skins = [
  { description: 'Ocean-god theme', name: 'poseidon', source: 'builtin' },
  { description: 'Volcanic theme', name: 'charizard', source: 'builtin' },
  { description: 'Custom blue theme', name: 'blueprint', source: 'user' }
]

describe('skin picker helpers', () => {
  it('keeps every skin when the filter is empty', () => {
    expect(filterSkinOptions(skins, '')).toEqual(skins)
  })

  it('filters by skin name and description', () => {
    expect(filterSkinOptions(skins, 'ocean').map(skin => skin.name)).toEqual(['poseidon'])
    expect(filterSkinOptions(skins, 'blue').map(skin => skin.name)).toEqual(['blueprint'])
  })

  it('shows optional agent branding without pretending every skin renames Hermes', () => {
    expect(skinAgentLabel({ agent_name: 'Charizard Agent' })).toBe('agent: Charizard Agent')
    expect(skinAgentLabel({})).toBe('agent: Hermes Agent')
  })
})

describe('skin preview scheduling', () => {
  it('debounces preview requests to avoid flooding the gateway', () => {
    expect(SKIN_PREVIEW_DEBOUNCE_MS).toBe(120)
  })

  it('resolves the active selection through the gateway instead of stale options', async () => {
    vi.useFakeTimers()

    const request = vi.fn((method: string) => {
      if (method === 'skin.options') {
        return Promise.resolve({
          active: 'default',
          active_skin: { name: 'default', version: 'old' },
          skins: [{ name: 'default' }]
        })
      }

      return Promise.resolve({ name: 'default', version: 'fresh' })
    })

    const mounted = mountPicker({ request })

    try {
      await vi.advanceTimersByTimeAsync(0)
      mounted.rerender()
      await vi.advanceTimersByTimeAsync(SKIN_PREVIEW_DEBOUNCE_MS)

      expect(request).toHaveBeenCalledWith('skin.preview', { name: 'default' })
    } finally {
      mounted.cleanup()
      vi.useRealTimers()
    }
  })

  it('ignores a preview response that arrives after selection is saved', async () => {
    vi.useFakeTimers()

    const pendingPreview = deferred<Record<string, unknown>>()

    const request = vi.fn((method: string) => {
      if (method === 'skin.options') {
        return Promise.resolve({
          active: 'default',
          active_skin: { name: 'default' },
          skins: [{ name: 'default' }, { name: 'charizard' }]
        })
      }

      if (method === 'skin.preview') {
        return pendingPreview.promise
      }

      return Promise.resolve({ value: 'charizard' })
    })

    const mounted = mountPicker({ request })

    try {
      await vi.advanceTimersByTimeAsync(0)
      mounted.rerender()
      pickerHarness.handler?.('', { downArrow: true })
      mounted.rerender()
      await vi.advanceTimersByTimeAsync(SKIN_PREVIEW_DEBOUNCE_MS)
      pickerHarness.applySkinPreview.mockClear()

      pickerHarness.handler?.('', { return: true })
      await vi.advanceTimersByTimeAsync(0)
      expect(mounted.onClose).toHaveBeenCalledTimes(1)

      pendingPreview.resolve({ name: 'charizard' })
      await vi.advanceTimersByTimeAsync(0)
      expect(pickerHarness.applySkinPreview).not.toHaveBeenCalled()
    } finally {
      mounted.cleanup()
      vi.useRealTimers()
    }
  })

  it('restores the latest persisted skin when cancelled', async () => {
    vi.useFakeTimers()

    const request = vi.fn(() =>
      Promise.resolve({
        active: 'default',
        active_skin: { name: 'default' },
        skins: [{ name: 'default' }, { name: 'charizard' }]
      })
    )

    const mounted = mountPicker({ request })

    try {
      await vi.advanceTimersByTimeAsync(0)
      mounted.rerender()
      pickerHarness.applySkinPreview.mockClear()
      pickerHarness.restorePersistedSkin.mockClear()

      pickerHarness.handler?.('', { escape: true })

      expect(pickerHarness.restorePersistedSkin).toHaveBeenCalledTimes(1)
      expect(pickerHarness.applySkinPreview).not.toHaveBeenCalled()
      expect(mounted.onClose).toHaveBeenCalledTimes(1)
    } finally {
      mounted.cleanup()
      vi.useRealTimers()
    }
  })

  it('handles Ctrl+C cancellation while idle', async () => {
    vi.useFakeTimers()

    const request = vi.fn(() =>
      Promise.resolve({
        active: 'default',
        active_skin: { name: 'default' },
        skins: [{ name: 'default' }, { name: 'charizard' }]
      })
    )

    const mounted = mountPicker({ request })

    try {
      await vi.advanceTimersByTimeAsync(0)
      mounted.rerender()
      pickerHarness.restorePersistedSkin.mockClear()

      pickerHarness.handler?.('c', { ctrl: true })

      expect(pickerHarness.restorePersistedSkin).toHaveBeenCalledTimes(1)
      expect(mounted.onClose).toHaveBeenCalledTimes(1)
    } finally {
      mounted.cleanup()
      vi.useRealTimers()
    }
  })

  it('ignores Ctrl+C while a skin save is in flight', async () => {
    vi.useFakeTimers()

    const pendingSave = deferred<{ value: string }>()

    const request = vi.fn((method: string) => {
      if (method === 'skin.options') {
        return Promise.resolve({
          active: 'default',
          active_skin: { name: 'default' },
          skins: [{ name: 'default' }]
        })
      }

      return pendingSave.promise
    })

    const mounted = mountPicker({ request })

    try {
      await vi.advanceTimersByTimeAsync(0)
      mounted.rerender()
      pickerHarness.handler?.('', { return: true })
      mounted.rerender()

      pickerHarness.handler?.('c', { ctrl: true })
      expect(mounted.onClose).not.toHaveBeenCalled()

      pendingSave.resolve({ value: 'default' })
      await vi.advanceTimersByTimeAsync(0)
      expect(mounted.onClose).toHaveBeenCalledTimes(1)
    } finally {
      mounted.cleanup()
      vi.useRealTimers()
    }
  })

  it('ignores a preview response older than the latest persisted skin', async () => {
    vi.useFakeTimers()
    pickerHarness.persistedRevision = 0

    const pendingPreview = deferred<Record<string, unknown>>()

    const request = vi.fn((method: string) => {
      if (method === 'skin.options') {
        return Promise.resolve({
          active: 'default',
          active_skin: { name: 'default' },
          skins: [{ name: 'default' }, { name: 'charizard' }]
        })
      }

      return pendingPreview.promise
    })

    const mounted = mountPicker({ request })

    try {
      await vi.advanceTimersByTimeAsync(0)
      mounted.rerender()
      pickerHarness.handler?.('', { downArrow: true })
      mounted.rerender()
      await vi.advanceTimersByTimeAsync(SKIN_PREVIEW_DEBOUNCE_MS)
      pickerHarness.applySkinPreview.mockClear()

      pickerHarness.persistedRevision = 1
      pendingPreview.resolve({ name: 'charizard' })
      await vi.advanceTimersByTimeAsync(0)

      expect(pickerHarness.applySkinPreview).not.toHaveBeenCalled()
    } finally {
      mounted.cleanup()
      vi.useRealTimers()
    }
  })
})
