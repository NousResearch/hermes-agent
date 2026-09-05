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

async function handoff(platform: string, ctx: SlashRunCtx): Promise<void> {
  const params = { session_id: ctx.sid }
  const current = () => getUiState().handoffSessionId === ctx.sid && getUiState().sid === null
  // Detach before the first await: prompts and /new must not write to or close
  // the session while the messaging gateway is taking ownership.
  patchUiState({ handoffSessionId: ctx.sid, sid: null, status: t('slashCmd.handoff.statusPending') })
  let queued = false
  const deadline = Date.now() + 180_000

  try {
    const result = await ctx.gateway.gw.request<HandoffRequest>('handoff.request', { ...params, platform })

    if (result?.queued !== true) {
      throw new Error(t('slashCmd.handoff.invalidAck'))
    }

    queued = true

    if (current()) {
      ctx.transcript.sys(t('slashCmd.handoff.pending', result.platform, result.home_name))
    }

    while (Date.now() < deadline) {
      if (!current()) {
        return
      }

      const result = await ctx.gateway.gw.request<HandoffState>('handoff.state', params)

      if (!current()) {
        return
      }

      if (result.state === 'completed') {
        ctx.transcript.sys(t('slashCmd.handoff.completed'))
        patchUiState({ sid: null, status: t('slashCmd.handoff.statusCompleted') })

        return
      }

      if (result.state !== 'pending' && result.state !== 'running') {
        if (result.state === 'failed') {
          const error = result.error || t('slashCmd.handoff.unknownError')
          ctx.transcript.sys(t('slashCmd.handoff.failed', error))
          patchUiState({ status: t('slashCmd.handoff.statusFailed', error) })
        } else {
          ctx.transcript.sys(t('slashCmd.handoff.unknown'))
          patchUiState({ status: t('slashCmd.handoff.statusUnknown') })
        }

        return
      }

      await new Promise(resolve => setTimeout(resolve, 1000))
    }

    throw new Error(t('slashCmd.handoff.pollTimedOut'))
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
