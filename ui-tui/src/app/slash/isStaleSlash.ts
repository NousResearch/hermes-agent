import type { SlashHandlerContext } from '../interfaces.js'
import { getUiState } from '../uiStore.js'

export function isStaleSlash(
  ctx: Pick<SlashHandlerContext, 'slashFlightRef'>,
  flight: number,
  sid: null | string
): boolean {
  return flight !== ctx.slashFlightRef.current || getUiState().sid !== sid
}
