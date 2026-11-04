import type { MutableRefObject } from 'react'

import type { SlashHandlerContext, UiState } from '../interfaces.js'

export interface SlashRunCtx extends SlashHandlerContext {
  flight: number
  guarded: <T>(fn: (r: T) => void) => (r: null | T) => void
  guardedErr: (e: unknown) => void
  /** A recognized command Ink runs itself calls this once it ran (`command.observe`, so the gateway
   *  fires `pre_command`); never on a refusal or a forward. Absent off a typed command. */
  ran?: () => void
  sid: null | string
  slashFlightRef: MutableRefObject<number>
  stale: () => boolean
  ui: UiState
}

export interface SlashCommand {
  aliases?: string[]
  help?: string
  name: string
  run: (arg: string, ctx: SlashRunCtx, cmd: string) => void
  usage?: string
}
