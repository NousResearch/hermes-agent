import type { WakeStartResponse, WakeStatusResponse, WakeStopResponse } from '../../../gatewayTypes.js'
import { setWakeUserDisabled } from '../../wakeState.js'
import type { SlashCommand, SlashRunCtx } from '../types.js'

const WAKE_SUBCOMMANDS = ['on', 'off', 'status'] as const

type WakeSub = (typeof WAKE_SUBCOMMANDS)[number]

const isWakeSub = (value: string): value is WakeSub => (WAKE_SUBCOMMANDS as readonly string[]).includes(value)

// Friendly text for the gateway's wake.start refusal codes. Unknown codes
// fall through to the raw reason so new server-side codes stay visible.
const START_REASON_TEXT: Record<string, string> = {
  disabled_for_surface: 'disabled for this surface (config wake_word.enabled / wake_word.surface)',
  not_owner: 'another surface owns the listener',
  owned: 'another surface owns the listener',
  unavailable: 'unavailable'
}

const startFailureLine = (r: WakeStartResponse): string => {
  const reason = r.reason ?? 'unknown'
  const base = START_REASON_TEXT[reason] ?? reason
  const owner = r.owner_surface ? ` (owned by ${r.owner_surface})` : ''
  const hint = r.hint?.trim() ? ` — ${r.hint.trim()}` : ''

  return `wake: not started — ${base}${owner}${hint}`
}

const statusLine = (r: WakeStatusResponse): string => {
  const phrase = r.phrase ? ` for “${r.phrase}”` : ''
  const provider = r.provider ? ` · ${r.provider}` : ''

  if (r.listening) {
    return `wake: listening${phrase}${provider}`
  }

  if (r.owner_surface && !r.owned_by_caller) {
    return `wake: off here · listener owned by ${r.owner_surface}${phrase}${provider}`
  }

  if (r.available === false) {
    const hint = r.hint?.trim() ? ` — ${r.hint.trim()}` : ''

    return `wake: unavailable${hint}`
  }

  return `wake: off${phrase}${provider} · /wake on to arm`
}

const runOn = (ctx: SlashRunCtx): void => {
  setWakeUserDisabled(false)

  ctx.gateway
    .rpc<WakeStartResponse>('wake.start', { surface: 'tui' })
    .then(
      ctx.guarded<WakeStartResponse>(r => {
        if (!r.started) {
          return ctx.transcript.sys(startFailureLine(r))
        }

        const phrase = r.phrase ? ` for “${r.phrase}”` : ''
        const provider = r.provider ? ` · ${r.provider}` : ''

        ctx.transcript.sys(`wake: listening${phrase}${provider}`)
      })
    )
    .catch(ctx.guardedErr)
}

const runOff = (ctx: SlashRunCtx): void => {
  // Remember the explicit opt-out so gateway reconnects don't re-arm the
  // listener behind the user's back (see wakeState.ts).
  setWakeUserDisabled(true)

  ctx.gateway
    .rpc<WakeStopResponse>('wake.stop', {})
    .then(
      ctx.guarded<WakeStopResponse>(r => {
        if (r.stopped) {
          return ctx.transcript.sys('wake: listener off (won’t re-arm this session)')
        }

        const reason = r.reason === 'not_owner' ? 'this surface doesn’t own the listener' : (r.reason ?? 'not running')

        ctx.transcript.sys(`wake: nothing to stop — ${reason}`)
      })
    )
    .catch(ctx.guardedErr)
}

const runStatus = (ctx: SlashRunCtx): void => {
  ctx.gateway
    .rpc<WakeStatusResponse>('wake.status', {})
    .then(ctx.guarded<WakeStatusResponse>(r => ctx.transcript.sys(statusLine(r))))
    .catch(ctx.guardedErr)
}

const WAKE_RUNNERS: Record<WakeSub, (ctx: SlashRunCtx) => void> = {
  off: runOff,
  on: runOn,
  status: runStatus
}

export const wakeCommands: SlashCommand[] = [
  {
    help: "toggle the 'Hey Hermes' wake word listener [on|off|status]",
    name: 'wake',
    usage: '/wake [on|off|status]',
    run: (arg, ctx) => {
      const sub = arg.trim().toLowerCase()

      if (sub && !isWakeSub(sub)) {
        return ctx.transcript.sys('usage: /wake [on|off|status]')
      }

      WAKE_RUNNERS[sub && isWakeSub(sub) ? sub : 'status'](ctx)
    }
  }
]
