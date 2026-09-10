import { randomUUID } from 'node:crypto'
import { mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'

import { expect, it, vi } from 'vitest'

import { submitPrompt } from '../app/submissionCore.js'
import { captureDestination } from '../app/submissionDestination.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { stageImagePath } from '../lib/imageAttachments.js'
import { loadPendingInputs } from '../lib/pendingInputs.js'

const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2nL8AAAAASUVORK5CYII=', 'base64')

it('stages private immutable local bytes, uploads remote bytes, and never falls back to a client path', async () => {
  const home = mkdtempSync(join(tmpdir(), 'ink-images-'))
  vi.stubEnv('HERMES_HOME', home)
  vi.stubEnv('HERMES_TUI_GATEWAY_URL', '')
  resetUiState()
  patchUiState({ sid: 'owner' })
  const source = join(home, 'my shot.png')
  writeFileSync(source, png)
  const request = vi.fn(async () => ({ path: '/owner/cache/images/upload.png' }))
  const gw = { request, isCanonical: true } as any

  try {
    const image = await stageImagePath(`"${source}" caption`, gw, captureDestination())
    expect(dirname(image.path)).toBe(join(home, 'cache', 'images'))
    expect(image).toMatchObject({ mime: 'image/png', remainder: 'caption' })
    expect(statSync(image.path).mode & 0o777).toBe(0o600)
    writeFileSync(source, 'changed')
    expect(readFileSync(image.path)).toEqual(png)
    expect(request).not.toHaveBeenCalled()
    writeFileSync(source, png)
    const prompt = vi.fn(async (_method: string, _params: any) => ({ status: 'started' }))
    submitPrompt(source, { gw: { isCanonical: true, request: prompt } as any, appendMessage: vi.fn(), enqueue: vi.fn(), expand: s => s, setLastUserMsg: vi.fn(), sys: vi.fn() })
    await vi.waitFor(() => expect(prompt).toHaveBeenCalledWith('prompt.submit', expect.objectContaining({ attachments: [{ path: expect.any(String), mime: 'image/png' }] })))
    expect(readFileSync(prompt.mock.calls.find(([method]) => method === 'prompt.submit')![1].attachments[0].path)).toEqual(png)
    vi.stubEnv('HERMES_TUI_GATEWAY_URL', 'wss://owner.example/ws')
    const remote = await stageImagePath(source, gw, captureDestination())
    expect(remote.path).toBe('/owner/cache/images/upload.png')
    expect(request).toHaveBeenCalledWith('image.attach_bytes', expect.objectContaining({
      content_base64: png.toString('base64'), session_id: 'owner'
    }))
    request.mockRejectedValueOnce(new Error('upload refused'))
    await expect(stageImagePath(source, gw, captureDestination())).rejects.toThrow('upload refused')
    writeFileSync(source, 'not an image')
    await expect(stageImagePath(source, gw, captureDestination())).rejects.toThrow(/image/i)
  } finally { resetUiState(); vi.unstubAllEnvs(); rmSync(home, { recursive: true, force: true }) }
})

it('journals canonical attachment payload before sending and replays it unchanged after a lost acknowledgement', async () => {
  const home = mkdtempSync(join(tmpdir(), 'ink-image-retry-'))
  vi.stubEnv('HERMES_HOME', home)
  resetUiState()
  patchUiState({ sid: 'owner' })
  const destination = captureDestination()
  const attachments = [{ path: join(home, 'cache/images/shot.png'), mime: 'image/png' }]
  const item = { text: 'caption', display: 'caption [[ Image 1 ]]', attachments, submissionId: randomUUID(), destination, inFlight: true }
  const wires: any[] = []

  const request = vi.fn(async (method, params) => {
    expect(method).toBe('prompt.submit')
    const retained = loadPendingInputs(destination)[0]!
    expect(retained.attachments).toEqual(attachments)
    wires.push(params)
    throw new Error('lost acknowledgement')
  })

  const deps = { gw: { isCanonical: true, request } as any, appendMessage: vi.fn(), enqueue: vi.fn(), expand: (s: string) => s, setLastUserMsg: vi.fn(), sys: vi.fn() }

  try {
    submitPrompt(item.text, deps, true, item.display, { skipDetectDrop: true, queueItem: item })
    await new Promise(resolve => setImmediate(resolve))
    const recovered = loadPendingInputs(destination)[0]!
    expect(recovered.attachments).toEqual(attachments)
    submitPrompt(recovered.text, deps, true, recovered.display, { queueItem: recovered })
    await new Promise(resolve => setImmediate(resolve))
    expect(wires).toHaveLength(2)
    expect(wires[0]).toEqual(wires[1])
    expect(wires[0].attachments).toEqual(attachments)
  } finally { resetUiState(); vi.unstubAllEnvs(); rmSync(home, { recursive: true, force: true }) }
})
