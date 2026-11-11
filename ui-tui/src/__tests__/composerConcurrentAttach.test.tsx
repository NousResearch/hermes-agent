import { EventEmitter } from 'node:events'
import { PassThrough } from 'node:stream'

import { renderSync } from '@hermes/ink'
import React from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { patchUiState, resetUiState } from '../app/uiStore.js'
import { useComposerState } from '../app/useComposerState.js'
import { TextInput } from '../components/textInput.js'

class FakeInput extends EventEmitter {
  chunks: string[] = []
  isRaw = false
  isTTY = true
  readableLength = 0
  ref = vi.fn()
  setEncoding = vi.fn()
  unref = vi.fn()

  read() {
    const next = this.chunks.shift() ?? null
    this.readableLength = this.chunks.length

    return next
  }

  send(chunk: string) {
    this.chunks.push(chunk)
    this.readableLength = this.chunks.length
    this.emit('readable')
  }

  setRawMode = vi.fn((enabled: boolean) => {
    this.isRaw = enabled
  })
}

const settle = async () => {
  for (let i = 0; i < 4; i++) {
    await new Promise(resolve => setTimeout(resolve, 20))
  }
}

const unmounts: Array<() => void> = []

afterEach(() => {
  unmounts.splice(0).forEach(fn => fn())
  resetUiState()
})

// The real composer hook behind the real TextInput, wired the way the app
// wires them (onChange reconciles tokens, then sets the draft). Every
// image.attach / clipboard.paste reply is held at the transport until the
// test releases it.
function mount() {
  resetUiState()
  patchUiState({ sid: 'sess' })
  const held: Array<{ method: string; path: string; release: () => void }> = []
  let clip = 0

  const request = vi.fn((method: string, params: { path?: string }) => {
    if (method === 'image.attach' || method === 'clipboard.paste') {
      const path = params.path ?? `/clip-${++clip}.png`

      return new Promise(resolve =>
        held.push({ method, path, release: () => resolve({ attached: true, name: path, path }) })
      )
    }

    return Promise.resolve({})
  })

  const stdin = new FakeInput()
  const stdout = Object.assign(new PassThrough(), { columns: 100, isTTY: false, rows: 24 })
  let composer!: ReturnType<typeof useComposerState>

  function Harness() {
    composer = useComposerState({ gw: { request } as any, submitRef: { current: () => {} }, sys: vi.fn() })

    return (
      <TextInput
        columns={100}
        onChange={value => {
          composer.actions.syncTokens(value)
          composer.actions.setInput(value)
        }}
        onPaste={composer.actions.handleTextPaste}
        value={composer.state.input}
      />
    )
  }

  const instance = renderSync(<Harness />, {
    patchConsole: false,
    stderr: new PassThrough() as any,
    stdin: stdin as any,
    stdout: stdout as any
  })

  unmounts.push(() => {
    instance.unmount()
    instance.cleanup()
  })

  return {
    held,
    request,
    stdin,
    get composer() {
      return composer
    },
    paste: (text: string) => stdin.send(`\x1b[200~${text}\x1b[201~`)
  }
}

const imagePaths = (h: ReturnType<typeof mount>) =>
  h.composer.refs.tokensRef.current.filter(t => t.kind === 'image').map(t => t.path)

const detached = (h: ReturnType<typeof mount>) =>
  h.request.mock.calls.filter(([method]) => method === 'image.detach').map(([, params]) => (params as any).path)

it('two image pastes and attaches in flight together all land, whichever reply comes back first', async () => {
  for (const order of [
    [0, 1],
    [1, 0]
  ]) {
    // Two bracketed path pastes about 0 s apart, as a terminal delivers them.
    const h = mount()
    await settle()
    h.paste('/a.png')
    h.paste('/b.png')
    await settle()
    expect(h.held.map(x => x.path)).toEqual(['/a.png', '/b.png'])

    for (const i of order) {
      h.held[i]!.release()
      await settle()
    }

    expect(h.composer.state.input).toBe('[[ Image 1 ]] [[ Image 2 ]]')
    expect(imagePaths(h).sort()).toEqual(['/a.png', '/b.png'])
    expect(new Set(h.composer.refs.tokensRef.current.map(t => t.label)).size).toBe(2)
    expect(detached(h)).toEqual([])

    // `/image` and `/paste` racing the same way, plus a path paste on top.
    const s = mount()
    await settle()
    s.composer.actions.attachImagePath('/c.png')
    s.composer.actions.attachClipboardImage()
    s.paste('/d.png')
    await settle()
    expect(s.held).toHaveLength(3)

    for (const i of order[0] ? [2, 1, 0] : [0, 1, 2]) {
      s.held[i]!.release()
      await settle()
    }

    expect(s.composer.state.input.match(/\[\[ Image \d \]\]/g)).toHaveLength(3)
    expect(imagePaths(s).sort()).toEqual(['/c.png', '/clip-1.png', '/d.png'])
    expect(new Set(s.composer.refs.tokensRef.current.map(t => t.label)).size).toBe(3)
    expect(detached(s)).toEqual([])
  }
})

it('a user edit while an image paste is in flight still drops it and detaches the image', async () => {
  const h = mount()
  await settle()
  h.paste('/a.png')
  await settle()
  h.composer.actions.setInput('typed instead')
  await settle()
  h.held[0]!.release()
  await settle()

  expect(h.composer.refs.tokensRef.current).toEqual([])
  expect(h.composer.state.input).not.toContain('Image')
  expect(detached(h)).toEqual(['/a.png'])
})

it('a second paste in the middle of the draft lands right after the first, not one landing further on', async () => {
  const h = mount()
  await settle()
  h.composer.actions.setInput('abcdef')
  await settle()
  // Caret after `ab`.
  h.stdin.send('\x1b[D\x1b[D\x1b[D\x1b[D')
  await settle()
  h.paste('X')
  await settle()
  expect(h.composer.state.input).toBe('abXcdef')
  h.paste('Y')
  await settle()

  expect(h.composer.state.input).toBe('abXYcdef')
})

// Lets every promise continuation run without yielding a macrotask, so Ink
// cannot re-render (and hand the input line the new draft) in between.
const microtasks = async () => {
  for (let i = 0; i < 50; i++) {
    await Promise.resolve()
  }
}

it('a slash attach and a path paste replying before a re-render both keep their image', async () => {
  for (const order of [
    [0, 1],
    [1, 0]
  ]) {
    const h = mount()
    await settle()
    h.composer.actions.attachImagePath('/c.png')
    h.paste('/d.png')
    await settle()
    expect(h.held.map(x => x.path)).toEqual(['/c.png', '/d.png'])

    // The first reply lands in full before the second is released, with no
    // render between them: the input line still shows the draft before both.
    for (const i of order) {
      h.held[i]!.release()
      await microtasks()
    }

    await settle()

    expect(h.composer.state.input).toBe('[[ Image 1 ]] [[ Image 2 ]]')
    expect(imagePaths(h).sort()).toEqual(['/c.png', '/d.png'])
    expect(detached(h)).toEqual([])
  }
})

it('an edit image restore in flight and a user image paste both land, nothing detached', async () => {
  for (const order of [
    [0, 1, 2],
    [1, 0, 2]
  ]) {
    const h = mount()
    await settle()
    // Esc Esc → Enter on a two-image prompt restores its images one after another.
    void h.composer.actions.attachImagePaths(['/r1.png', '/r2.png'], { followTyping: true })
    h.paste('/p.png')
    await settle()
    expect(h.held.map(x => x.path)).toEqual(['/r1.png', '/p.png'])

    for (const i of order) {
      h.held[i]!.release()
      await settle()
    }

    expect(h.held.map(x => x.path)).toContain('/r2.png')
    expect(h.composer.state.input.match(/\[\[ Image \d \]\]/g)).toHaveLength(3)
    expect(imagePaths(h).sort()).toEqual(['/p.png', '/r1.png', '/r2.png'])
    expect(new Set(h.composer.refs.tokensRef.current.map(t => t.label)).size).toBe(3)
    expect(detached(h)).toEqual([])
  }
})
