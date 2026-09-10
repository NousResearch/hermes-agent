import { randomUUID } from 'node:crypto'

import { introMsg, toTranscriptMessages } from '../../domain/messages.js'
import { TUI_SESSION_MODEL_FLAG } from '../../domain/slash.js'
import { asRpcResult } from '../../lib/rpc.js'
import { getUiState, patchUiState } from '../uiStore.js'

import type { SlashRunCtx } from './types.js'

interface MutationGateway {
  request: (method: string, params?: Record<string, unknown>) => Promise<unknown>
}

// Keep the original CAS tuple after a lost reply. Reissuing the same command
// must query that receipt, not silently authorize another write at a new revision.
const pending = new WeakMap<MutationGateway, Map<string, Record<string, unknown>>>()

function modelPayload(arg: string) {
  const parts = arg.trim().split(/\s+/)
  const model = parts.shift()!
  const payload: Record<string, string> = { model }

  if (model.startsWith('--')) {
    throw new Error('usage: /model <model> [--provider <provider>] [--session]')
  }

  while (parts.length) {
    const flag = parts.shift()!

    if (flag === '--session' || flag === TUI_SESSION_MODEL_FLAG) {
      continue
    }

    if (flag === '--provider' && parts[0] && !parts[0].startsWith('--')) {
      payload.provider = parts.shift()!

      continue
    }

    throw new Error(`unsupported canonical model option: ${flag}`)
  }

  return payload
}

export async function mutateCanonicalSession(
  gw: MutationGateway,
  sid: string,
  operation: 'model' | 'branch' | 'compress',
  arg: string,
  stale: () => boolean = () => false
) {
  const payload =
    operation === 'model' ? modelPayload(arg) : arg ? { [operation === 'branch' ? 'title' : 'focus']: arg } : {}

  let requests = pending.get(gw)

  if (!requests) {
    requests = new Map()
    pending.set(gw, requests)
  }

  const key = JSON.stringify([sid, operation, payload])
  let params = requests.get(key)

  if (!params) {
    const snapshot = asRpcResult(await gw.request('session.resume', { session_id: sid }))

    if (stale()) {
      return
    }

    if (!Number.isSafeInteger(snapshot?.revision) || !Number.isSafeInteger(snapshot?.execution_generation)) {
      throw new Error('session execution identity unavailable; reconnect before editing')
    }

    params = {
      session_id: sid,
      request_id: randomUUID(),
      expected_revision: snapshot!.revision,
      expected_generation: snapshot!.execution_generation,
      operation,
      payload
    }
    requests.set(key, params)
  }

  let result

  try {
    result = asRpcResult(await gw.request('session.mutate', params))

    if (!result) {
      throw new Error('invalid response: session.mutate')
    }

    requests.delete(key)
  } catch (error) {
    // A structured authority refusal is definitive; transport errors are not.
    if ((error as { data?: { reason?: string } })?.data?.reason) {
      requests.delete(key)
    }

    throw error
  }
  return { result, expectedGeneration: params.expected_generation as number }
}

export async function runCanonicalSessionControl(
  operation: 'model' | 'branch' | 'compress',
  arg: string,
  ctx: SlashRunCtx
) {
  const gw = ctx.gateway.gw

  try {
    if (!ctx.sid) {
      throw new Error('no active session')
    }

    const mutation = await mutateCanonicalSession(gw, ctx.sid, operation, arg, ctx.stale)

    if (!mutation) {
      return
    }

    const { result, expectedGeneration } = mutation
    if (ctx.stale()) {
      return
    }

    const current = getUiState().info

    if (
      operation !== 'branch' &&
      (current?.execution_epoch !== ctx.ui.info?.execution_epoch ||
        (current?.execution_generation ?? 0) > (result.execution_generation ?? expectedGeneration))
    ) {
      return
    }

    if (operation === 'branch') {
      if (!result.branched_session_id) {
        throw new Error('invalid response: branch')
      }

      ctx.session.resumeById(result.branched_session_id)
      ctx.transcript.sys(`branched → ${arg || result.branched_session_id}`)
    } else if (operation === 'model') {
      if (!result.model) {
        throw new Error('invalid response: model switch')
      }

      patchUiState(state => ({
        ...state,
        info: {
          ...state.info,
          model: result.model,
          execution_generation: Math.max(state.info?.execution_generation ?? 0, result.execution_generation ?? 0),
          skills: state.info?.skills ?? {},
          tools: state.info?.tools ?? {}
        }
      }))
      ctx.transcript.sys(`model → ${result.model}`)
    } else {
      const before = getUiState()
      const snapshot = asRpcResult(await gw.request('session.resume', { session_id: ctx.sid }))

      if (ctx.stale() || getUiState().info !== before.info || getUiState().busy !== before.busy) {
        return
      }

      if (!snapshot || !Array.isArray(snapshot.messages)) {
        throw new Error('invalid response: compressed transcript')
      }

      const info = { ...before.info, ...snapshot.info }

      if (
        info.execution_epoch !== before.info?.execution_epoch ||
        (info.execution_generation ?? -1) < (result.execution_generation ?? 0)
      ) {
        throw new Error('compressed transcript is stale; reopen the session')
      }

      ctx.transcript.setHistoryItems([introMsg(info), ...toTranscriptMessages(snapshot.messages)])
      patchUiState({ info })
      ctx.transcript.sys('✓ transcript compressed')
    }
  } catch (error) {
    ctx.guardedErr(error)
  }
}
