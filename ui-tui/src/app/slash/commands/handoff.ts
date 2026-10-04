import { t } from '../../../i18n/runtime.js'
import { rpcErrorMessage } from '../../../lib/rpc.js'
import { getUiState, patchUiState } from '../../uiStore.js'
import type { SlashCommand, SlashRunCtx } from '../types.js'

interface HandoffRequest {
  queued: boolean
  platform: string
  home_name: string
}

interface HandoffState {
  state: string
  error?: string
}

interface HandoffFail {
  failed: boolean
  state: string
}

// Same two-phase bound as the CLI (`_handoff_wait`) and desktop: an unclaimed row is
// cancelled after this long (a claim racing the cancel wins), while a claimed one
// is replaying the transcript at the destination and is only watched, never failed.
const PENDING_TIMEOUT_MS = 60_000
const RUNNING_TIMEOUT_MS = 180_000
const POLL_INTERVAL_MS = 1000

async function handoff(platform: string, ctx: SlashRunCtx): Promise<void> {
  const params = { session_id: ctx.sid }
  const current = () => getUiState().handoffSessionId === ctx.sid && getUiState().sid === null
  // Detach before the first await: prompts and /new must not write to or close
  // the session while the messaging gateway is taking ownership.
  patchUiState({ handoffSessionId: ctx.sid, sid: null, status: t('slashCmd.handoff.statusPending') })
  let queued = false

  const completed = () => {
    ctx.transcript.sys(t('slashCmd.handoff.completed'))
    patchUiState({ sid: null, status: t('slashCmd.handoff.statusCompleted') })
  }

  try {
    const result = await ctx.gateway.gw.request<HandoffRequest>('handoff.request', { ...params, platform })

    if (result?.queued !== true) {
      throw new Error(t('slashCmd.handoff.invalidAck'))
    }

    queued = true

    if (current()) {
      ctx.transcript.sys(t('slashCmd.handoff.pending', result.platform, result.home_name))
    }

    let state = 'pending'
    let deadline = Date.now() + PENDING_TIMEOUT_MS

    for (;;) {
      if (!current()) {
        return
      }

      let record: HandoffState | null = null

      try {
        record = await ctx.gateway.gw.request<HandoffState>('handoff.state', params)
      } catch {
        // A transient poll failure proves nothing either way; the deadline still bounds it.
      }

      if (!current()) {
        return
      }

      if (record?.state === 'completed') {
        return completed()
      }

      if (record?.state === 'running' && state !== 'running') {
        state = 'running'
        deadline = Date.now() + RUNNING_TIMEOUT_MS
      } else if (record && record.state !== 'pending' && record.state !== 'running') {
        if (record.state === 'failed') {
          const error = record.error || t('slashCmd.handoff.unknownError')
          ctx.transcript.sys(t('slashCmd.handoff.failed', error))
          patchUiState({ status: t('slashCmd.handoff.statusFailed', error) })
        } else {
          ctx.transcript.sys(t('slashCmd.handoff.unknown'))
          patchUiState({ status: t('slashCmd.handoff.statusUnknown') })
        }

        return
      }

      if (Date.now() >= deadline) {
        if (state !== 'pending') {
          throw new Error(t('slashCmd.handoff.pollTimedOut'))
        }

        // Cancel only an unclaimed row (server-side CAS). Its success proves nothing was
        // delivered, so the source is restored; a claim that won the race keeps polling.
        const cancel = await ctx.gateway.gw.request<HandoffFail>('handoff.fail', {
          ...params,
          error: 'timed out waiting for gateway'
        })

        if (!current()) {
          return
        }

        if (cancel?.failed === true) {
          patchUiState({ sid: ctx.sid, status: 'ready' })
          ctx.transcript.sys(t('slashCmd.handoff.timedOutRestored'))

          return
        }

        if (cancel?.state === 'completed') {
          return completed()
        }

        if (cancel?.state !== 'running') {
          throw new Error(t('slashCmd.handoff.pollTimedOut'))
        }

        state = 'running'
        deadline = Date.now() + RUNNING_TIMEOUT_MS
      }

      await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL_MS))
    }
  } catch (error) {
    if (current()) {
      // Only these preflight errors prove the request never reached the DB.
      // A lost acknowledgement (or failed delivery after rebinding) does not.
      const code = (error as { code?: number } | null)?.code

      if (!queued && code !== undefined && [4009, 4023, 4024, 4025, 4026, 5021].includes(code)) {
        patchUiState({ sid: ctx.sid, status: 'ready' })
        ctx.transcript.sys(t('slashCmd.handoff.rejected', rpcErrorMessage(error)))
      } else {
        patchUiState({ status: t('slashCmd.handoff.statusUnknown') })
        ctx.transcript.sys(t('slashCmd.handoff.unknownWithError', rpcErrorMessage(error)))
      }
    }
  } finally {
    if (getUiState().handoffSessionId === ctx.sid) {
      patchUiState({ handoffSessionId: null })
    }
  }
}

export const handoffCommands: SlashCommand[] = [
  {
    name: 'handoff',
    help: 'hand this session off to a messaging platform',
    usage: '<platform>',
    run: (arg, ctx) => {
      if (!arg.trim() || /\s/.test(arg.trim())) {
        return ctx.transcript.sys(t('slashCmd.handoff.usage'))
      }

      if (!ctx.sid) {
        return ctx.transcript.sys(t('slashCmd.handoff.noSession'))
      }

      if (ctx.ui.busy || ctx.ui.compacting || ctx.ui.bgTasks.size || ctx.composer.queueRef.current.length) {
        return ctx.transcript.sys(t('slashCmd.handoff.waitForWork'))
      }

      void handoff(arg.trim(), ctx)
    }
  }
]
