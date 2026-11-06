import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { PassThrough } from 'node:stream'

import { renderSync, Text } from '@hermes/ink'
import React from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $overlayState, resetOverlayState } from '../app/overlayStore.js'
import { openRewindPicker } from '../app/slash/canonicalRewindPicker.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { useComposerState } from '../app/useComposerState.js'
import { stageImagePath } from '../lib/imageAttachments.js'

vi.mock('../lib/imageAttachments.js', () => ({ stageClipboardImage: vi.fn(), stageImagePath: vi.fn() }))

const flush = () => new Promise(resolve => setImmediate(resolve))
const staged = (name: string) => ({ mime: 'image/png', name: `${name}.png`, path: `/cache/${name}.png` })
const stored = 'question\n\n[Image attached at: /cache/a.png]\n[Image attached at: /cache/b.png]'

let home = ''
let unmount = () => {}

beforeEach(() => {
  home = mkdtempSync(join(tmpdir(), 'backtrack-image-'))
  vi.stubEnv('HERMES_HOME', home)
  resetUiState()
  resetOverlayState()
  patchUiState({ sid: 'A' })
})

afterEach(() => {
  unmount()
  vi.unstubAllEnvs()
  vi.mocked(stageImagePath).mockReset()
  resetUiState()
  resetOverlayState()
  rmSync(home, { force: true, recursive: true })
})

/** The real composer hook, `/rewind` Enter on a two-image prompt, staging held at the transport seam. */
async function editTwoImagePrompt() {
  const releases: Array<() => void> = []

  vi.mocked(stageImagePath).mockImplementation(
    path => new Promise(done => releases.push(() => done(staged(path.includes('a.png') ? 'a' : 'b')) as any))
  )

  let composer!: ReturnType<typeof useComposerState>

  const gw = {
    isCanonical: true,
    request: vi.fn(async () => ({ messages: [{ content: stored, role: 'user', row_id: 1 }] }))
  }

  function Harness() {
    composer = useComposerState({ gw, submitRef: { current: vi.fn() }, sys: vi.fn() } as any)

    return <Text>composer</Text>
  }

  const stdout = Object.assign(new PassThrough(), { columns: 80, isTTY: false, rows: 24 })
  const instance = renderSync(<Harness />, { patchConsole: false, stdin: new PassThrough() as any, stdout: stdout as any })

  unmount = () => instance.unmount()

  openRewindPicker({
    composer: { attachImagePath: (p: string) => composer.actions.attachImagePath(p), attachImagePaths: (p: string[]) => composer.actions.attachImagePaths(p), setInput: (v: any) => composer.actions.setInput(v) },
    gateway: { gw },
    guardedErr: () => {},
    sid: 'A',
    stale: () => false,
    transcript: { sys: vi.fn() },
    ui: { busy: false }
  } as any)
  await flush()

  const browse = $overlayState.get().rewind!

  browse.onEdit(browse.targets[0]!)
  await flush()

  return { composer: () => composer, releases }
}

const calls = () => vi.mocked(stageImagePath).mock.calls.map(([path, , destination]) => ({ path, sid: destination.sid }))

describe('restoring the images of an edited prompt', () => {
  it('attaches both images in order, the second only after the first landed', async () => {
    const { composer, releases } = await editTwoImagePrompt()

    expect(calls()).toEqual([{ path: '/cache/a.png', sid: 'A' }])
    releases[0]!()
    await flush()
    await flush()
    expect(calls()).toEqual([
      { path: '/cache/a.png', sid: 'A' },
      { path: '/cache/b.png', sid: 'A' }
    ])
    releases[1]!()
    await flush()
    await flush()
    expect(composer().refs.tokensRef.current.map(token => (token as { path?: string }).path)).toEqual([
      '/cache/a.png',
      '/cache/b.png'
    ])
  })

  it('a session switch mid-restore stops the chain: nothing is staged for or lands in the new session', async () => {
    const { composer, releases } = await editTwoImagePrompt()

    patchUiState({ sid: 'B' })
    composer().actions.clearIn()
    composer().actions.setInput('new draft')
    releases[0]!()
    await flush()
    await flush()

    expect(calls()).toEqual([{ path: '/cache/a.png', sid: 'A' }])
    expect(composer().refs.tokensRef.current).toEqual([])
  })

  it('replacing the draft mid-restore stops the chain in the same session too', async () => {
    const { composer, releases } = await editTwoImagePrompt()

    composer().actions.setInput('typed over it')
    releases[0]!()
    await flush()
    await flush()

    expect(calls()).toHaveLength(1)
    expect(composer().refs.tokensRef.current).toEqual([])
  })
})
