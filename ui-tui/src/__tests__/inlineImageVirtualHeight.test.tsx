import { PassThrough } from 'stream'

import { Box, renderSync, ScrollBox, type ScrollBoxHandle, Text } from '@hermes/ink'
import type { ImageRenderResult } from '@hermes/shared/gateway-events'
import React, { useLayoutEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { GatewayProvider } from '../app/gatewayContext.js'
import type { GatewayServices } from '../app/interfaces.js'
import { getUiState, resetUiState } from '../app/uiStore.js'
import { applyDisplay } from '../app/useConfigSync.js'
import { MessageLine } from '../components/messageLine.js'
import type { GatewayClient } from '../gatewayClient.js'
import { useVirtualHistory } from '../hooks/useVirtualHistory.js'
import { estimatedMsgHeight } from '../lib/virtualHeights.js'
import { DEFAULT_THEME } from '../theme.js'
import type { Msg } from '../types.js'

const unmounts: Array<() => void> = []

beforeEach(() => resetUiState())

afterEach(() => {
  unmounts.splice(0).forEach(fn => fn())
  vi.restoreAllMocks()
})

const settle = async () => {
  for (let i = 0; i < 12; i++) {
    await new Promise(resolve => setImmediate(resolve))
  }
}

const imageRow = (i: number): { key: string; msg: Msg } => ({
  key: String(i),
  msg: { images: [`/tmp/virtual-${i}.png`], kind: 'image', role: 'system', text: '' }
})

interface Exposed {
  vh: ReturnType<typeof useVirtualHistory>
}

function Harness({ expose, items }: { expose: { current: Exposed | null }; items: { key: string; msg: Msg }[] }) {
  const scrollRef = useRef<ScrollBoxHandle | null>(null)

  const vh = useVirtualHistory(scrollRef, items, 100, {
    coldStartCount: 8,
    estimateHeight: i => estimatedMsgHeight(items[i]!.msg, 100, { compact: false, details: true }),
    maxMounted: 8,
    overscan: 2
  })

  useLayoutEffect(() => {
    expose.current = { vh }
  })

  return (
    <ScrollBox flexDirection="column" height={10} ref={scrollRef} stickyScroll>
      <Box flexDirection="column">
        <Box height={vh.topSpacer} />
        {items.slice(vh.start, vh.end).map(item => (
          <Box flexDirection="column" key={item.key} ref={vh.measureRef(item.key)}>
            <MessageLine cols={100} msg={item.msg} t={DEFAULT_THEME} />
          </Box>
        ))}
        <Box height={vh.bottomSpacer} />
        <Text>tail</Text>
      </Box>
    </ScrollBox>
  )
}

const mount = (node: React.ReactNode, render: ImageRenderResult) => {
  const request = vi.fn(async () => render)
  const gateway = { gw: { request } as unknown as GatewayClient, rpc: vi.fn() } as unknown as GatewayServices
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()

  Object.assign(stdout, { columns: 100, isTTY: false, rows: 60 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', () => {})

  const instance = renderSync(<GatewayProvider value={gateway}>{node}</GatewayProvider>, {
    patchConsole: false,
    stderr: stderr as unknown as NodeJS.WriteStream,
    stdin: stdin as unknown as NodeJS.ReadStream,
    stdout: stdout as unknown as NodeJS.WriteStream
  })

  unmounts.push(() => {
    instance.unmount()
    instance.cleanup()
  })

  return request
}

describe('inline image rows in the virtual transcript', () => {
  it('an image row that draws nothing measures as 0 rows, not its full-height estimate', async () => {
    const items = Array.from({ length: 40 }, (_, i) => imageRow(i))
    const expose = { current: null as Exposed | null }

    mount(<Harness expose={expose} items={items} />, { mode: 'off' })
    await settle()

    const { vh } = expose.current!
    // The mounted (measured) rows are the ones the user scrolled past; they must not keep 16 rows each.
    let mountedHeight = 0

    for (let i = vh.start; i < vh.end; i++) {
      mountedHeight += vh.offsets[i + 1]! - vh.offsets[i]!
    }

    expect(vh.end - vh.start).toBeGreaterThan(0)
    expect(mountedHeight).toBe(0)
  })

  it('with display.inline_images off, image rows neither reserve rows nor ask the gateway', async () => {
    const tool: Msg = { images: ['/tmp/a.png'], kind: 'image', role: 'system', text: '' }
    const user: Msg = { images: ['/tmp/a.png'], role: 'user', text: 'what is this?' }
    const plainUser: Msg = { role: 'user', text: 'what is this?' }

    // YAML 1.1 turns a bare `off` into false; both spellings disable images.
    for (const inline_images of [false, 'off'] as const) {
      applyDisplay({ config: { display: { inline_images } } }, () => {})

      const opts = { compact: false, details: true, inlineImages: getUiState().inlineImages }

      expect(estimatedMsgHeight(tool, 100, opts)).toBe(0)
      expect(estimatedMsgHeight(user, 100, opts)).toBe(estimatedMsgHeight(plainUser, 100, opts))
    }

    const request = mount(<MessageLine cols={100} msg={tool} t={DEFAULT_THEME} />, { mode: 'off' })

    await settle()
    expect(request).not.toHaveBeenCalled()
  })
})
