import type { WakeStartResponse, WakeStatusResponse, WakeStopResponse } from '../../../gatewayTypes.js'
import { type Locale, translate, type TranslationKey } from '../../../i18n/index.js'
import { setWakeUserDisabled } from '../../wakeState.js'
import type { SlashCommand, SlashRunCtx } from '../types.js'

const WAKE_SUBCOMMANDS = ['on', 'off', 'status'] as const

type WakeSub = (typeof WAKE_SUBCOMMANDS)[number]

const isWakeSub = (value: string): value is WakeSub => (WAKE_SUBCOMMANDS as readonly string[]).includes(value)

// Friendly text for the gateway's wake.start refusal codes. Unknown codes
// fall through to the raw reason so new server-side codes stay visible.
const START_REASON_KEYS: Record<string, TranslationKey> = {
  disabled: 'wake.reason.disabled',
  disabled_for_surface: 'wake.reason.disabledForSurface',
  not_owner: 'wake.reason.otherOwner',
  owned: 'wake.reason.otherOwner',
  unavailable: 'wake.reason.unavailable'
}

const startFailureLine = (r: WakeStartResponse, locale: Locale): string => {
  const reason = r.reason ?? 'unknown'
  const reasonKey = START_REASON_KEYS[reason]
  const base = reasonKey ? translate(locale, reasonKey) : reason
  const owner = r.owner_surface ? translate(locale, 'wake.ownerSuffix', { surface: r.owner_surface }) : ''
  const hint = r.hint?.trim() ? translate(locale, 'wake.hintSuffix', { hint: r.hint.trim() }) : ''

  return translate(locale, 'wake.notStarted', { reason: base, owner, hint })
}

const statusLine = (r: WakeStatusResponse, locale: Locale): string => {
  const phrase = r.phrase ? translate(locale, 'wake.phraseSuffix', { phrase: r.phrase }) : ''
  const provider = r.provider ? translate(locale, 'wake.providerSuffix', { provider: r.provider }) : ''

  if (r.listening) {
    if (r.audio_silent) {
      const hint = r.hint?.trim() ? translate(locale, 'wake.hintSuffix', { hint: r.hint.trim() }) : ''

      return translate(locale, 'wake.listeningSilent', { phrase, provider, hint })
    }

    return translate(locale, 'wake.listening', { phrase, provider, saved: '' })
  }

  if (r.owner_surface && !r.owned_by_caller) {
    return translate(locale, 'wake.offHere', { surface: r.owner_surface, phrase, provider })
  }

  if (r.available === false) {
    const hint = r.hint?.trim() ? translate(locale, 'wake.hintSuffix', { hint: r.hint.trim() }) : ''

    return translate(locale, 'wake.unavailable', { hint })
  }

  return translate(locale, 'wake.off', { phrase, provider })
}

const runOn = (ctx: SlashRunCtx): void => {
  setWakeUserDisabled(false)

  // persist: true — an explicit /wake on writes wake_word.enabled to config
  // so the choice survives restarts (the backend only persists on gesture
  // paths; reconnect auto-arm never does).
  ctx.gateway
    .rpc<WakeStartResponse>('wake.start', { persist: true, surface: 'tui' })
    .then(
      ctx.guarded<WakeStartResponse>(r => {
        if (!r.started) {
          return ctx.transcript.sys(startFailureLine(r, ctx.ui.locale))
        }

        const phrase = r.phrase ? translate(ctx.ui.locale, 'wake.phraseSuffix', { phrase: r.phrase }) : ''
        const provider = r.provider ? translate(ctx.ui.locale, 'wake.providerSuffix', { provider: r.provider }) : ''
        const saved = r.enabled_persisted ? translate(ctx.ui.locale, 'wake.enabledSavedSuffix') : ''

        ctx.transcript.sys(translate(ctx.ui.locale, 'wake.listening', { phrase, provider, saved }))
      })
    )
    .catch(ctx.guardedErr)
}

const runOff = (ctx: SlashRunCtx): void => {
  // Remember the explicit opt-out so gateway reconnects don't re-arm the
  // listener behind the user's back (see wakeState.ts).
  setWakeUserDisabled(true)

  ctx.gateway
    .rpc<WakeStopResponse>('wake.stop', { persist: true })
    .then(
      ctx.guarded<WakeStopResponse>(r => {
        const saved = r.disabled_persisted ? translate(ctx.ui.locale, 'wake.disabledSavedSuffix') : ''

        if (r.stopped) {
          return ctx.transcript.sys(translate(ctx.ui.locale, 'wake.listenerOff', { saved }))
        }

        const reason =
          r.reason === 'not_owner'
            ? translate(ctx.ui.locale, 'wake.reason.thisSurfaceNotOwner')
            : (r.reason ?? translate(ctx.ui.locale, 'wake.reason.notRunning'))

        ctx.transcript.sys(translate(ctx.ui.locale, 'wake.nothingToStop', { reason, saved }))
      })
    )
    .catch(ctx.guardedErr)
}

const runStatus = (ctx: SlashRunCtx): void => {
  ctx.gateway
    .rpc<WakeStatusResponse>('wake.status', {})
    .then(ctx.guarded<WakeStatusResponse>(r => ctx.transcript.sys(statusLine(r, ctx.ui.locale))))
    .catch(ctx.guardedErr)
}

const WAKE_RUNNERS: Record<WakeSub, (ctx: SlashRunCtx) => void> = {
  off: runOff,
  on: runOn,
  status: runStatus
}

export const wakeCommands: SlashCommand[] = [
  {
    name: 'wake',
    usage: '/wake [on|off|status]',
    run: (arg, ctx) => {
      const sub = arg.trim().toLowerCase()

      if (sub && !isWakeSub(sub)) {
        return ctx.transcript.sys(translate(ctx.ui.locale, 'wake.usage'))
      }

      WAKE_RUNNERS[sub && isWakeSub(sub) ? sub : 'status'](ctx)
    }
  }
]
