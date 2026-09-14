import type { PetCellsResult, PetInfoMetaResult } from '@hermes/shared/gateway-events'

import type { GatewayClient } from '../gatewayClient.js'

interface PetUpdate {
  cells: PetCellsResult | null
  meta: PetInfoMetaResult
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
export async function requestPetUpdate(
  gateway: PetGateway,
  state: string,
  graphics: boolean,
  needsCells: (meta: PetInfoMetaResult) => boolean
): Promise<PetUpdate | null> {
  try {
    const meta = await gateway.request('pet.info.meta', {})

    if (!meta.enabled || !needsCells(meta)) {
      return { cells: null, meta }
    }

    return { cells: await gateway.request('pet.cells', { graphics, state }), meta }
  } catch {
    return null
  }
}
