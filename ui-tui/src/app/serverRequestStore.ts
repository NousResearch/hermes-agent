import type { ServerRequest } from '@hermes/shared/json-rpc-channel'

// Live server→client requests (clarify, approval, sudo, …) keyed by request
// id. Overlay state keeps only the id; answering resolves the stored request
// so a re-delivered (`open_requests`) request with the same id reuses the
// same card. Module-level, like the overlay store: the gateway client and
// the Ink handlers share one instance per process.
const open = new Map<string, ServerRequest>()

export function rememberServerRequest(request: ServerRequest): void {
  open.set(request.id, request)
}

export function forgetServerRequest(id: string): void {
  open.delete(id)
}

/** Answer request `id` and forget it. False when nothing is open under that id (expired / already answered). */
export function respondToServerRequest(id: string, result: Record<string, unknown>): boolean {
  const request = open.get(id)

  if (!request) {
    return false
  }

  open.delete(id)
  request.respond(result)

  return true
}

export function hasOpenServerRequest(id: string): boolean {
  return open.has(id)
}

export function resetServerRequestsForTests(): void {
  open.clear()
}
