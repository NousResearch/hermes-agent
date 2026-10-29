import type { GatewayClient } from '../gatewayClient.js'

import { asRpcResult } from './rpc.js'

export interface CompanionMetaResult {
  enabled?: boolean
  scale?: number
  slug?: string
  spritesheetRevision?: string
}

interface CompanionUpdate<TCells> {
  cells: TCells | null
  meta: CompanionMetaResult
}

type CompanionGateway = Pick<GatewayClient, 'request'>

/**
 * Suppress overlapping cosmetic polls so a slow gateway can never accumulate
 * a queue of companion requests. Returning false tells callers that an existing
 * probe is still in flight.
 */
export function createCompanionSingleFlight() {
  let active = false

  return async (operation: () => Promise<void>): Promise<boolean> => {
    if (active) {
      return false
    }

    active = true

    try {
      await operation()

      return true
    } finally {
      active = false
    }
  }
}

/**
 * Probe cheap companion metadata on the gateway reader thread, then request the
 * expensive frame payload only when the active selection/state is not cached.
 * This deliberately bypasses the transcript-logging RPC wrapper: companion display
 * is cosmetic, so an unavailable gateway must not print an error.
 */
export async function requestCompanionUpdate<TCells>(
  gateway: CompanionGateway,
  state: string,
  graphics: boolean,
  needsCells: (meta: CompanionMetaResult) => boolean
): Promise<CompanionUpdate<TCells> | null> {
  try {
    const meta = asRpcResult(await gateway.request('companion.info.meta')) as CompanionMetaResult | null

    if (!meta) {
      return null
    }

    if (!meta.enabled || !needsCells(meta)) {
      return { cells: null, meta }
    }

    const cells = asRpcResult(await gateway.request('companion.cells', { graphics, state })) as TCells | null

    return { cells, meta }
  } catch {
    return null
  }
}
