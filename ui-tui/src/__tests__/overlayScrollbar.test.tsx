import { PassThrough } from 'node:stream'

import { renderSync, type ScrollBoxHandle } from '@hermes/ink'
import React from 'react'
import { expect, it, vi } from 'vitest'

import { OverlayScrollbar } from '../components/overlayScrollbar.js'
import { DEFAULT_THEME } from '../theme.js'

it('re-renders when ScrollBox reports content or viewport geometry changes', async () => {
  let total = 4
  let notify: (() => void) | null = null
  const getScrollHeight = vi.fn(() => total)
  const subscribe = vi.fn((listener: () => void) => {
    notify = listener
    return () => {
      notify = null
    }
  })

  const scroll = {
    getPendingDelta: () => 0,
    getScrollHeight,
    getScrollTop: () => 0,
    getViewportHeight: () => 4,
    scrollTo: vi.fn(),
    subscribe
  } as unknown as ScrollBoxHandle

  const stdout = Object.assign(new PassThrough(), { columns: 20, rows: 8, isTTY: false })
  const view = renderSync(
    <OverlayScrollbar scrollRef={{ current: scroll }} t={DEFAULT_THEME} tick={0} />,
    {
      stdout: stdout as unknown as NodeJS.WriteStream,
      stdin: Object.assign(new PassThrough(), { isTTY: false }) as unknown as NodeJS.ReadStream,
      stderr: new PassThrough() as unknown as NodeJS.WriteStream,
      patchConsole: false
    }
  )

  try {
    await vi.waitFor(() => expect(subscribe).toHaveBeenCalledTimes(1))
    const before = getScrollHeight.mock.calls.length

    total = 8
    notify?.()

    await vi.waitFor(() => expect(getScrollHeight.mock.calls.length).toBeGreaterThan(before))
  } finally {
    view.unmount()
    view.cleanup()
  }
})
