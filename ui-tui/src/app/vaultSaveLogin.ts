import type { GatewayRpc } from './interfaces.js'

export const submitVaultSaveLogin = (
  rpc: GatewayRpc,
  requestId: string,
  identifier?: string,
  password?: string
) =>
  rpc('vault.save_login.respond', {
    login: identifier === undefined || password === undefined ? '' : JSON.stringify({ identifier: identifier.trim(), password }),
    request_id: requestId
  })
