import { once } from 'node:events'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, expect, it, vi } from 'vitest'

import { GatewayClient } from '../gatewayClient.js'

// Real loopback WebSockets; only the gateway application is a protocol fixture.
const { WebSocketServer } = createRequire(import.meta.url)('ws')
const cleanup: Array<() => unknown | Promise<unknown>> = []
afterEach(async () => {
  for (const fn of cleanup.reverse()) {await fn()}
  cleanup.length = 0
  vi.unstubAllEnvs()
})

async function harness(mode: 'ws' | 'stdio') {
  const home = await mkdtemp(join(tmpdir(), 'hermes-draft-wire-'))
  cleanup.push(() => rm(home, { recursive: true, force: true }))
  vi.stubEnv('HERMES_HOME', home)
  const server = new WebSocketServer({ host: '127.0.0.1', port: 0 })
  await once(server, 'listening')
  cleanup.push(() => new Promise<void>(resolve => { for (const ws of server.clients) {ws.terminate();} server.close(resolve) }))
  const base = `ws://127.0.0.1:${server.address().port}`
  const sidecars: Array<{ ws: any; url: URL; frames: any[] }> = []
  const gatewayFrames: any[] = []
  server.on('connection', (ws: any, req: any) => {
    const url = new URL(req.url, base)

    if (url.pathname === '/api/pub') {
      const sidecar = { ws, url, frames: [] as any[] }
      sidecars.push(sidecar)
      ws.on('message', (raw: Buffer) => sidecar.frames.push(JSON.parse(raw.toString())))
      ws.send(JSON.stringify({ type: 'draft.refresh', pty_instance: 'pty-1', connection_generation: sidecars.length }))
    } else {
      ws.send(JSON.stringify({ method: 'event', params: { type: 'gateway.ready' } }))
      ws.on('message', (raw: Buffer) => {
        const frame = JSON.parse(raw.toString())
        gatewayFrames.push(frame)
        ws.send(JSON.stringify({ id: frame.id, result: { ok: true } }))
      })
    }
  })
  vi.stubEnv('HERMES_TUI_SIDECAR_URL', `${base}/api/pub?token=server-token&channel=channel&controller=private-credential`)
  vi.stubEnv('HERMES_TUI_GATEWAY_URL', mode === 'ws' ? `${base}/api/ws` : '')

  if (mode === 'stdio') {
    await mkdir(join(home, 'tui_gateway'))
    await writeFile(join(home, 'tui_gateway', 'entry.py'), `import json, os, sys
print(json.dumps({'method': 'event', 'params': {'type': 'gateway.ready'}}), flush=True)
for line in sys.stdin:
    frame = json.loads(line)
    print(json.dumps({'id': frame['id'], 'result': {'ok': True, 'sidecar': os.environ.get('HERMES_TUI_SIDECAR_URL')}}), flush=True)
`)
    vi.stubEnv('HERMES_PYTHON_SRC_ROOT', home)
    vi.stubEnv('HERMES_CWD', home)
    vi.stubEnv('HERMES_PYTHON', 'python3')
  }

  const client = new GatewayClient()
  cleanup.push(() => client.kill('test cleanup'))

  return { client, sidecars, gatewayFrames }
}

it.each(['ws', 'stdio'] as const)('bridges typed drafts over the Node sidecar in %s mode', async mode => {
  const { client, sidecars, gatewayFrames } = await harness(mode)
  const requests: any[] = []
  const refresh: unknown[] = []
  client.on('draft.request', request => requests.push(request))
  client.on('draft.refresh', () => {
    refresh.push(true)
    client.publishDraftState({ session_id: 'native-session', draft_id: 'live-draft', available: true })
  })
  client.publishDraftState({ session_id: 'native-session', draft_id: 'live-draft', available: true })
  client.start()
  await vi.waitFor(() => expect(sidecars[0]?.frames.some(frame => frame.type === 'draft.state')).toBe(true), { timeout: 5000 })
  const sidecar = sidecars[0]!
  expect(sidecar.url.searchParams.get('controller')).toBe('private-credential')
  const state = sidecar.frames.find(frame => frame.type === 'draft.state')
  expect(sidecar.frames.filter(frame => frame.type === 'draft.state')).toHaveLength(1)
  expect(state).toMatchObject({ session_id: 'native-session', draft_id: 'live-draft', available: true,
    pty_instance: 'pty-1', connection_generation: 1 })
  expect(refresh.length).toBeGreaterThan(0)

  const request = { type: 'draft.attach', request_id: 'one', path: '/staged/report.txt', expected: {
    pty_instance: 'pty-1', connection_generation: 1, session_id: 'native-session', draft_id: 'live-draft'
  } }

  sidecar.ws.send(JSON.stringify({ jsonrpc: '2.0', id: 'attack', method: 'shell.exec', params: {} }))
  sidecar.ws.send(JSON.stringify(request))
  await vi.waitFor(() => expect(requests).toEqual([request]))
  const result = { type: 'draft.result' as const, request_id: 'one', identity: request.expected, status: 'attached' as const }
  client.publishDraftResult(result)
  await vi.waitFor(() => expect(sidecar.frames).toContainEqual(result))
  const rpc = await client.request<{ ok: boolean; sidecar?: string }>('file.attach', { session_id: 'native-session', path: request.path })
  expect(rpc.ok).toBe(true)

  if (mode === 'stdio') {expect(rpc.sidecar).toBeNull()}
  expect(gatewayFrames.every(frame => frame.method === 'file.attach')).toBe(true)
  expect(client.getLogTail()).not.toContain('private-credential')
})

it('retires pending authority on browser refresh, sidecar loss and gateway replacement', async () => {
  const { client, sidecars } = await harness('ws')
  let disconnected = 0
  const requests: any[] = []
  client.on('draft.disconnected', () => disconnected++)
  client.on('draft.request', req => requests.push(req))
  client.publishDraftState({ session_id: 'sid', draft_id: 'draft', available: true })
  client.start()
  await vi.waitFor(() => expect(sidecars[0]?.frames.some(f => f.type === 'draft.state')).toBe(true))
  const first = sidecars[0]!
  const identity = { pty_instance: 'pty-1', connection_generation: 1, session_id: 'sid', draft_id: 'draft' }
  const request = { type: 'draft.attach', request_id: 'pending', path: '/staged', expected: identity }
  first.ws.send(JSON.stringify(request))
  await vi.waitFor(() => expect(requests).toHaveLength(1))
  first.ws.send(JSON.stringify({ type: 'draft.refresh', pty_instance: 'pty-1', connection_generation: 2 }))
  await vi.waitFor(() => expect(disconnected).toBeGreaterThan(0))
  client.publishDraftResult({ type: 'draft.result', request_id: 'pending', identity, status: 'attached' })
  first.ws.send(JSON.stringify({ type: 'draft.refresh', pty_instance: 'pty-1', connection_generation: 1 }))
  first.ws.send(JSON.stringify({ ...request, request_id: 'old-generation' }))
  first.ws.send(JSON.stringify({ type: 'draft.disconnected' }))
  await vi.waitFor(() => expect(disconnected).toBe(2))
  first.ws.close()
  await vi.waitFor(() => expect(sidecars).toHaveLength(2), { timeout: 5000 })
  const second = sidecars[1]!
  await vi.waitFor(() => expect(second.frames.some(f => f.type === 'draft.state')).toBe(true))
  expect(Number(second.url.searchParams.get('controller_generation')))
    .toBeGreaterThan(Number(first.url.searchParams.get('controller_generation')))
  expect(requests).toHaveLength(1)
  expect(first.frames.some(f => f.type === 'draft.result')).toBe(false)
  client.start()
  await vi.waitFor(() => expect(sidecars).toHaveLength(3), { timeout: 5000 })
  client.publishDraftState({ session_id: 'replacement', draft_id: 'new', available: true })
  await vi.waitFor(() => expect(sidecars[2]!.frames.some(f => f.session_id === 'replacement')).toBe(true))
  expect(sidecars[2]!.frames.some(f => f.session_id === 'sid')).toBe(false)
})
