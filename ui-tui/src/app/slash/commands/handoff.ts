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
  patchUiState({ handoffSessionId: ctx.sid, sid: null, status: 'handoff pending…' })
  let queued = false
  const deadline = Date.now() + 180_000

  try {
    const result = await ctx.gateway.gw.request<HandoffRequest>('handoff.request', { ...params, platform })

    if (result?.queued !== true) {
      throw new Error('invalid handoff acknowledgement')
    }

    queued = true

    if (current()) {
      ctx.transcript.sys(`handoff pending: ${result.platform} · ${result.home_name}`)
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
        if (current()) {
          ctx.transcript.sys('handoff completed — continue on the destination platform; /new starts a new session here')
          patchUiState({ sid: null, status: 'handoff completed' })
        }

        return
      }

      if (result.state !== 'pending' && result.state !== 'running') {
        if (current()) {
          const reason = result.state === 'failed' ? `failed: ${result.error || 'unknown error'}` : 'outcome unknown'
          ctx.transcript.sys(`handoff ${reason} — check the destination before resuming; /new starts a fresh session`)
          patchUiState({ status: `handoff ${reason}` })
        }

        return
      }

      await new Promise(resolve => setTimeout(resolve, 1000))
    }

    throw new Error('handoff status polling timed out')
  } catch (error) {
    if (current()) {
      // Only these preflight errors prove the request never reached the DB.
      // A lost acknowledgement (or failed delivery after rebinding) does not.
      const code = (error as { code?: number } | null)?.code

      if (!queued && code !== undefined && [4009, 4023, 4024, 4025, 4026, 5021].includes(code)) {
        patchUiState({ sid: ctx.sid, status: 'ready' })
        ctx.transcript.sys(`handoff: ${rpcErrorMessage(error)}`)
      } else {
        patchUiState({ status: 'handoff outcome unknown' })
        ctx.transcript.sys(
          `handoff outcome unknown: ${rpcErrorMessage(error)} — check the destination before resuming; /new starts a fresh session`
        )
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
        return ctx.transcript.sys('usage: /handoff <platform>')
      }

      if (!ctx.sid) {
        return ctx.transcript.sys('no active session — nothing to hand off')
      }

      if (ctx.ui.busy || ctx.ui.compacting || ctx.ui.bgTasks.size || ctx.composer.queueRef.current.length) {
        return ctx.transcript.sys('wait for the current turn and queued/background work before handoff')
      }

      void handoff(arg.trim(), ctx)
    }
  }
]
