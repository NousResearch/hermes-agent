import type { WakeStartResponse, WakeStatusResponse, WakeStopResponse } from '../../../gatewayTypes.js'
import { t } from '../../../i18n/runtime.js'
import type { TranslationKey } from '../../../i18n/types.js'
import { setWakeUserDisabled } from '../../wakeState.js'
import type { SlashCommand, SlashRunCtx } from '../types.js'

const WAKE_SUBCOMMANDS = ['on', 'off', 'status'] as const

type WakeSub = (typeof WAKE_SUBCOMMANDS)[number]

const isWakeSub = (value: string): value is WakeSub => (WAKE_SUBCOMMANDS as readonly string[]).includes(value)

// Friendly text for the gateway's wake.start refusal codes (catalog keys,
// resolved at reply time so a locale swap is observed). Unknown codes fall
// through to the raw reason so new server-side codes stay visible.
const START_REASON_KEY: Record<string, TranslationKey> = {
  disabled: 'slashCmd.wake.reason.disabled',
  disabled_for_surface: 'slashCmd.wake.reason.disabledForSurface',
  not_owner: 'slashCmd.wake.reason.notOwner',
  owned: 'slashCmd.wake.reason.owned',
  unavailable: 'slashCmd.wake.reason.unavailable'
}

const withHint = (text: string, hint: string | undefined): string =>
  hint?.trim() ? t('slashCmd.wake.withHint', text, hint.trim()) : text

const startFailureLine = (r: WakeStartResponse): string => {
  const key = r.reason ? START_REASON_KEY[r.reason] : undefined
  const base = key ? t(key) : (r.reason ?? t('slashCmd.wake.reason.unknown'))
  const owner = r.owner_surface ? ` (${t('slashCmd.wake.ownedBy', r.owner_surface)})` : ''

  return withHint(t('slashCmd.wake.notStarted', `${base}${owner}`), r.hint)
}

/** ` for “phrase” · provider` detail suffix shared by the on/status replies. */
const detailSuffix = (r: { phrase?: string; provider?: string }): string => {
  const phrase = r.phrase ? ` ${t('slashCmd.wake.forPhrase', r.phrase)}` : ''
  const provider = r.provider ? ` · ${r.provider}` : ''

  return `${phrase}${provider}`
}

const statusLine = (r: WakeStatusResponse): string => {
  const details = detailSuffix(r)

  if (r.listening) {
    const listening = t('slashCmd.wake.listening', details)

    if (r.audio_silent) {
      return withHint(`${listening} · ${t('slashCmd.wake.micSilent')}`, r.hint)
    }

    return listening
  }

  if (r.owner_surface && !r.owned_by_caller) {
    return t('slashCmd.wake.offOwnedBy', r.owner_surface, details)
  }

  if (r.available === false) {
    return withHint(t('slashCmd.wake.unavailable'), r.hint)
  }

  return t('slashCmd.wake.off', details)
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
          return ctx.transcript.sys(startFailureLine(r))
        }

        const saved = r.enabled_persisted ? ` · ${t('slashCmd.wake.enabledInConfig')}` : ''

        ctx.transcript.sys(t('slashCmd.wake.listening', `${detailSuffix(r)}${saved}`))
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
        const saved = r.disabled_persisted ? ` · ${t('slashCmd.wake.disabledInConfig')}` : ''

        if (r.stopped) {
          return ctx.transcript.sys(t('slashCmd.wake.listenerOff', saved))
        }

        const reason =
          r.reason === 'not_owner' ? t('slashCmd.wake.notOwnerStop') : (r.reason ?? t('slashCmd.wake.notRunning'))

        ctx.transcript.sys(t('slashCmd.wake.nothingToStop', reason, saved))
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
        return ctx.transcript.sys(t('slashCmd.wake.usage'))
      }

      WAKE_RUNNERS[sub && isWakeSub(sub) ? sub : 'status'](ctx)
    }
  }
]
