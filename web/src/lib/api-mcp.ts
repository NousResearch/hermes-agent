/** MCP administration uses the facade's existing authenticated, profile-scoped transport. */
export function createMcpApi(fetchJSON: <T>(url: string, init?: RequestInit) => Promise<T>) {
  return {
    // ── Admin: MCP servers ──────────────────────────────────────────────
    getMcpServers: () => fetchJSON<{ servers: McpServer[] }>('/api/mcp/servers'),
    addMcpServer: (body: McpServerCreate) =>
      fetchJSON<McpServer>('/api/mcp/servers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      }),
    authMcpServer: (name: string) =>
      fetchJSON<McpOAuthFlow>(`/api/mcp/servers/${encodeURIComponent(name)}/auth`, { method: 'POST' }),
    getMcpOAuthFlow: (flowId: string) => fetchJSON<McpOAuthFlow>(`/api/mcp/oauth/flows/${encodeURIComponent(flowId)}`),
    removeMcpServer: (name: string) =>
      fetchJSON<{ ok: boolean }>(`/api/mcp/servers/${encodeURIComponent(name)}`, {
        method: 'DELETE'
      }),
    testMcpServer: (name: string) =>
      fetchJSON<McpTestResult>(`/api/mcp/servers/${encodeURIComponent(name)}/test`, { method: 'POST' }),
    setMcpServerEnabled: (name: string, enabled: boolean) =>
      fetchJSON<{ ok: boolean; name: string; enabled: boolean }>(
        `/api/mcp/servers/${encodeURIComponent(name)}/enabled`,
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ enabled })
        }
      ),
    getMcpCatalog: () =>
      fetchJSON<{ entries: McpCatalogEntry[]; diagnostics: McpCatalogDiagnostic[] }>('/api/mcp/catalog'),
    installMcpCatalogEntry: (name: string, env: Record<string, string> = {}, enable = true, network?: McpNetwork) =>
      fetchJSON<{ ok: boolean; name: string; background: boolean; action?: string }>('/api/mcp/catalog/install', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, env, enable, ...(network ? { network } : {}) })
      })
  }
}

export interface McpServer {
  name: string
  transport: 'http' | 'sse' | 'stdio' | 'unknown'
  network?: McpNetwork
  url: string | null
  command: string | null
  args: string[]
  env: Record<string, string>
  auth: 'header' | 'oauth' | null
  enabled: boolean
  tools: string[] | null
}

export interface McpCatalogEntry {
  name: string
  description: string
  source: string
  transport: 'http' | 'stdio'
  auth_type: 'api_key' | 'oauth' | 'none'
  required_env: Array<{ name: string; prompt: string; required: boolean }>
  // Transport details — what actually connects (http) or runs (stdio).
  command: string | null
  args: string[]
  url: string | null
  // Git bootstrap (only set for entries that clone + build locally).
  install_url: string | null
  install_ref: string | null
  bootstrap: string[]
  // Default tool pre-selection (null = all tools pre-checked) + guidance text.
  default_enabled: string[] | null
  post_install: string
  needs_install: boolean
  installed: boolean
  enabled: boolean
}

export interface McpCatalogDiagnostic {
  name: string
  kind: string
  message: string
}

export type McpHttpAuth = 'none' | 'header' | 'oauth'
export type McpNetwork = 'auto' | 'local' | 'windows'

export interface McpServerCreate {
  name: string
  network?: McpNetwork
  transport?: 'http' | 'sse'
  url?: string
  command?: string
  args?: string[]
  env?: Record<string, string>
  auth?: McpHttpAuth
  bearer_token?: string
}

export interface McpTestResult {
  ok: boolean
  error?: string
  tools: Array<{ name: string; description: string }>
}

export interface McpOAuthFlow {
  flow_id: string
  server_name: string
  status: 'starting' | 'authorization_required' | 'approved' | 'error'
  authorization_url: string | null
  error: string | null
  tools?: Array<{ name: string; description: string }>
}
