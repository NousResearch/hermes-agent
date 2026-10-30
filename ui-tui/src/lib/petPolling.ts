import type { GatewayClient } from '../gatewayClient.js'

import { asRpcResult } from './rpc.js'

export interface PetMetaResult {
  enabled?: boolean
  scale?: number
  slug?: string
  spritesheetRevision?: string
}

interface PetUpdate<TCells> {
  cells: TCells | null
  meta: PetMetaResult
}

type PetGateway = Pick<GatewayClient, 'request'>

/**
 * Suppress overlapping cosmetic polls so a slow gateway can never accumulate
 * a queue of pet requests. Returning false tells callers that an existing
 * probe is still in flight.
 */
export function createPetSingleFlight() {
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
 * Probe cheap pet metadata on the gateway reader thread, then request the
 * expensive frame payload only when the active selection/state is not cached.
 * This deliberately bypasses the transcript-logging RPC wrapper: pet display
 * is cosmetic, so an unavailable gateway must not print an error.
 */
export async function requestPetUpdate<TCells>(
  gateway: PetGateway,
  state: string,
  graphics: boolean,
  needsCells: (meta: PetMetaResult) => boolean
): Promise<PetUpdate<TCells> | null> {
  try {
    const meta = asRpcResult(await gateway.request('pet.info.meta')) as PetMetaResult | null

    if (!meta) {
      return null
    }

    if (!meta.enabled || !needsCells(meta)) {
      return { cells: null, meta }
    }

    const cells = asRpcResult(await gateway.request('pet.cells', { graphics, state })) as TCells | null

    return { cells, meta }
  } catch {
    return null
  }
}
