// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api, setManagementProfile } from "./api";
import type { McpNetwork } from "./api-types";

const fetchMock = vi.fn<typeof fetch>(
  async () =>
    new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    })
)
beforeEach(() => {
  fetchMock.mockClear()
  vi.stubGlobal('fetch', fetchMock)
  window.__HERMES_SESSION_TOKEN__ = 'session-token'
  setManagementProfile('work & play')
})
afterEach(() => {
  setManagementProfile('')
  delete window.__HERMES_SESSION_TOKEN__
  vi.unstubAllGlobals()
})
function request(index = 0) {
  const [url, init] = fetchMock.mock.calls[index]
  return { url, init, body: init?.body ? JSON.parse(String(init.body)) : undefined }
}
describe('MCP public api facade', () => {
  it.each([undefined, 'auto', 'local', 'windows'] as const)(
    'preserves catalog network=%s without adding a legacy default',
    async network => {
      await api.installMcpCatalogEntry('unity', { TEST_KEY: 'test-only' }, false, network)
      expect(request().url).toBe('/api/mcp/catalog/install?profile=work%20%26%20play')
      expect(request().body).toEqual({
        name: 'unity',
        env: { TEST_KEY: 'test-only' },
        enable: false,
        ...(network ? { network } : {})
      })
      expect(request().init?.method).toBe('POST')
    }
  )
  it.each(['auto', 'local', 'windows'] as const)(
    'preserves HTTP and SSE creation network=%s',
    async (network: McpNetwork) => {
      for (const transport of ['http', 'sse'] as const) {
        const body = {
          name: 'unity',
          url: 'http://localhost:8080/mcp',
          network,
          transport,
          auth: 'header' as const,
          bearer_token: 'fixture-token'
        }
        await api.addMcpServer(body)
        expect(request(fetchMock.mock.calls.length - 1).body).toEqual(body)
      }
      for (const [url, init] of fetchMock.mock.calls) {
        expect(url).toBe('/api/mcp/servers?profile=work%20%26%20play')
        expect(init?.credentials).toBe('include')
        expect(new Headers(init?.headers).get('X-Hermes-Session-Token')).toBe('session-token')
      }
    }
  )
  it('keeps all MCP routes and escaping on the current management profile', async () => {
    await api.getMcpServers()
    await api.getMcpCatalog()
    await api.authMcpServer('a/b')
    await api.getMcpOAuthFlow('flow/id')
    await api.removeMcpServer('a/b')
    await api.testMcpServer('a/b')
    await api.setMcpServerEnabled('a/b', false)
    const paths = [
      '/api/mcp/servers',
      '/api/mcp/catalog',
      '/api/mcp/servers/a%2Fb/auth',
      '/api/mcp/oauth/flows/flow%2Fid',
      '/api/mcp/servers/a%2Fb',
      '/api/mcp/servers/a%2Fb/test',
      '/api/mcp/servers/a%2Fb/enabled'
    ]
    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual(paths.map(path => `${path}?profile=work%20%26%20play`))
    expect(fetchMock.mock.calls.map(([, init]) => init?.method ?? 'GET')).toEqual([
      'GET',
      'GET',
      'POST',
      'GET',
      'DELETE',
      'POST',
      'PUT'
    ])
    expect(request(6).body).toEqual({ enabled: false })
    setManagementProfile('next')
    await api.getMcpServers()
    expect(request(7).url).toBe('/api/mcp/servers?profile=next')
  })
  it('keeps gated cookie auth and stdio payloads without inventing network', async () => {
    delete window.__HERMES_SESSION_TOKEN__
    window.__HERMES_AUTH_REQUIRED__ = true
    try {
      const body = { name: 'time', command: 'uvx', args: ['mcp-server-time'] }
      await api.addMcpServer(body)
      expect(request().body).toEqual(body)
      expect(request().init?.credentials).toBe('include')
      expect(new Headers(request().init?.headers).has('X-Hermes-Session-Token')).toBe(false)
    } finally {
      delete window.__HERMES_AUTH_REQUIRED__
    }
  })
})
