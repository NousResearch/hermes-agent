import { PassThrough } from 'stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { GatewayProvider } from '../app/gatewayContext.js'
import type { GatewayServices } from '../app/interfaces.js'
import { MessageLine } from '../components/messageLine.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { ImageRenderResponse } from '../gatewayTypes.js'
import { DEFAULT_THEME } from '../theme.js'
import type { Msg } from '../types.js'

const PLACEHOLDER = '\u{10EEEE}'
const unmounts: Array<() => void> = []

afterEach(() => {
  unmounts.splice(0).forEach(fn => fn())
  vi.restoreAllMocks()
})

const settle = async () => {
  for (let i = 0; i < 4; i++) {
    await new Promise(resolve => setImmediate(resolve))
  }
}

const mount = (msgs: Msg[], render: ImageRenderResponse | ((params: { path: string }) => ImageRenderResponse)) => {
  const request = vi.fn(async (_method: string, params: { path: string }) =>
    typeof render === 'function' ? render(params) : render
  )

  const gateway = { gw: { request } as unknown as GatewayClient, rpc: vi.fn() } as unknown as GatewayServices
  const stdout = new PassThrough()
  const stdin = new PassThrough()
  const stderr = new PassThrough()
  let output = ''

  Object.assign(stdout, { columns: 100, isTTY: false, rows: 60 })
  Object.assign(stdin, { isTTY: false })
  Object.assign(stderr, { isTTY: false })
  stdout.on('data', chunk => {
    output += chunk.toString()
  })

  const instance = renderSync(
    <GatewayProvider value={gateway}>
      {msgs.map((msg, i) => (
        <MessageLine cols={100} key={i} msg={msg} t={DEFAULT_THEME} />
      ))}
    </GatewayProvider>,
    {
      patchConsole: false,
      stderr: stderr as NodeJS.WriteStream,
      stdin: stdin as NodeJS.ReadStream,
      stdout: stdout as NodeJS.WriteStream
    }
  )

  unmounts.push(() => {
    instance.unmount()
    instance.cleanup()
  })

  return { output: () => output, request }
}

describe('inline transcript images', () => {
  it('paints kitty placeholders and sends each image to the terminal once', async () => {
    const terminal = vi.spyOn(process.stdout, 'write').mockImplementation(() => true)
    const transmit = '\x1b_Ga=T,U=1,i=14698097,c=3,r=2,f=100,q=2,m=0;AAAA\x1b\\'

    const { output, request } = mount(
      [
        { images: ['/tmp/kitty-shot.png'], kind: 'image', role: 'system', text: '' },
        { images: ['/tmp/kitty-shot.png'], kind: 'image', role: 'system', text: '' }
      ],
      {
        color: '#e04671',
        cols: 3,
        mode: 'kitty',
        placeholder: [`${PLACEHOLDER}̅${PLACEHOLDER}${PLACEHOLDER}`, `${PLACEHOLDER}̍${PLACEHOLDER}${PLACEHOLDER}`],
        rows: 2,
        transmit
      }
    )

    await settle()

    expect(request).toHaveBeenCalledTimes(1)
    expect(request).toHaveBeenCalledWith('image.render', expect.objectContaining({ path: '/tmp/kitty-shot.png' }))
    expect(terminal.mock.calls.filter(([data]) => data === transmit)).toHaveLength(1)
    // Both rows of both copies: the shared image is painted by placeholder cells.
    expect(
      output()
        .split('\n')
        .filter(line => line.includes(PLACEHOLDER))
    ).toHaveLength(4)
  })

  it("draws half-blocks under a user turn's attachment", async () => {
    const red = [255, 0, 0, 255, 0, 0, 255, 255]

    const { output } = mount([{ images: ['/tmp/attached.png'], role: 'user', text: 'what is this?' }], {
      cells: [
        [red, red],
        [red, red]
      ],
      cols: 2,
      mode: 'unicode',
      rows: 2
    })

    await settle()

    expect(output()).toContain('what is this?')
    expect(output()).toContain('▀')
  })

  it('shows a new picture taken over the same path by a later tool call', async () => {
    // Piped output carries no color, so the two pictures differ in width.
    const px = [255, 0, 0, 255, 255, 0, 0, 255]
    const shot = '/tmp/reshot.png'
    let width = 1

    const picture = (): ImageRenderResponse => ({
      cells: [Array(width).fill(px)],
      cols: width,
      mode: 'unicode',
      rows: 1
    })

    const first = mount([{ imageRevision: 'call-1', images: [shot], kind: 'image', role: 'system', text: '' }], picture)

    await settle()
    expect(first.output()).not.toContain('▀▀▀')

    width = 3

    const second = mount(
      [{ imageRevision: 'call-2', images: [shot], kind: 'image', role: 'system', text: '' }],
      picture
    )

    await settle()
    expect(second.request).toHaveBeenCalledTimes(1)
    expect(second.output()).toContain('▀▀▀')
  })

  it('asks again after an off answer, so a later enable or a file that appears still shows', async () => {
    const path = '/tmp/later.png'
    const first = mount([{ images: [path], kind: 'image', role: 'system', text: '' }], { mode: 'off' })

    await settle()
    expect(first.request).toHaveBeenCalledTimes(1)

    const second = mount([{ images: [path], kind: 'image', role: 'system', text: '' }], {
      cells: [[[255, 0, 0, 255, 255, 0, 0, 255]]],
      cols: 1,
      mode: 'unicode',
      rows: 1
    })

    await settle()
    expect(second.request).toHaveBeenCalledTimes(1)
    expect(second.output()).toContain('▀')
  })

  it('draws nothing when inline images are off', async () => {
    const { output } = mount([{ images: ['/tmp/off.png'], kind: 'image', role: 'system', text: '' }], { mode: 'off' })

    await settle()

    expect(output()).not.toContain('▀')
    expect(output()).not.toContain(PLACEHOLDER)
  })
})
