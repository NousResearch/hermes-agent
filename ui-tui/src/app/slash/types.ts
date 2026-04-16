import type { MutableRefObject } from 'react'

import type { SlashHandlerContext, UiState } from '../interfaces.js'

import type { SlashShared } from './shared.js'

export interface SlashRunCtx extends SlashHandlerContext {
  flight: number
  guarded: <T>(fn: (r: T) => void) => (r: null | T) => void
  guardedErr: (e: unknown) => void
  shared: SlashShared
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
