import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'

const startup = vi.hoisted(() => ({ image: '' }))
vi.mock('../config/env.js', async importOriginal => ({ ...await importOriginal<Record<string, unknown>>(), get STARTUP_IMAGE() { return startup.image }, STARTUP_QUERY: 'literal caption' }))

it('startup image bytes travel with the literal prompt and attachment failure prevents a text-only run', async () => {
  const home = mkdtempSync(join(tmpdir(), 'ink-startup-image-'))
  vi.stubEnv('HERMES_HOME', home)
  vi.stubEnv('HERMES_TUI_GATEWAY_URL', '')
  startup.image = join(home, 'startup.png')
  const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2nL8AAAAASUVORK5CYII=', 'base64')
  writeFileSync(startup.image, png)

  try {
    for (const valid of [true, false]) {
      resetUiState()
      patchUiState({ sid: 'owner' })

      if (!valid) { rmSync(startup.image) }
      const submit = vi.fn()
      const sys = vi.fn()
      const request = vi.fn(async () => ({}))

      const handler = createGatewayEventHandler({
        gateway: { gw: { isCanonical: true, request }, rpc: request },
        composer: { setInput: vi.fn(), enqueue: vi.fn() },
        session: { STARTUP_RESUME_ID: 'owner', colsRef: { current: 80 }, resumeById: vi.fn(), newSession: vi.fn(), resetSession: vi.fn(), setCatalog: vi.fn() },
        submission: { submitLiteralRef: { current: submit }, submitRef: { current: vi.fn() } },
        system: { sys, bellOnComplete: false, bellOnPrompt: false },
        transcript: { appendMessage: vi.fn(), panel: vi.fn(), setHistoryItems: vi.fn() },
        voice: { setProcessing: vi.fn(), setRecording: vi.fn(), setVoiceEnabled: vi.fn() }
      } as any)

      handler({ type: 'gateway.ready', payload: {} } as any)

      if (valid) {
        await vi.waitFor(() => expect(submit).toHaveBeenCalled())
        expect(submit).toHaveBeenCalledWith('literal caption', [{ path: expect.any(String), mime: 'image/png' }])
        expect(readFileSync(submit.mock.calls[0]![1][0].path)).toEqual(png)
      } else {
        await vi.waitFor(() => expect(sys).toHaveBeenCalledWith(expect.stringContaining('startup image attach failed')))
        expect(submit).not.toHaveBeenCalled()
      }

      expect(request.mock.calls.some(([method]) => method === 'image.attach')).toBe(false)
    }
  } finally { resetUiState(); vi.unstubAllEnvs(); rmSync(home, { recursive: true, force: true }) }
})
