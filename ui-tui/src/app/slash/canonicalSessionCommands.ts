// Ink slash commands whose legacy RPCs (`session.title`, `session.undo`, `session.usage`, the
// `/tools` slash worker) the canonical gateway does not serve, mapped onto what it does serve:
// `session.mutate` rename / rewind (`gateway/session_mutations.py`), the committed admission
// result (`prompt.receipt include_result`) and the frozen launch policy (`session.info`).

import type { SlashExecResponse } from '../../gatewayTypes.js'
import { t } from '../../i18n/runtime.js'
import { asRpcResult } from '../../lib/rpc.js'
import type { PanelSection } from '../../types.js'
import { getUiState, patchUiState } from '../uiStore.js'

import { mutateCanonical, type MutationSnapshot } from './canonicalSessionControls.js'
import type { SlashRunCtx } from './types.js'

const COMPACTION_MARKERS = ['[CONTEXT COMPACTION', '[CONTEXT SUMMARY]:', '[END OF PRIOR CONTEXT']

/** A human-authored user turn (`agent/context_compressor.py::split_user_originated_turn`): no
 * display-only kind other than a typed /steer, and not a compaction handoff carrier. */
const isUserTurn = (row: Record<string, unknown>) =>
  row.role === 'user' &&
  (!row.display_kind || row.display_kind === 'steer') &&
  Number.isSafeInteger(row.row_id) &&
  !(typeof row.content === 'string' && COMPACTION_MARKERS.some(marker => row.content!.toString().includes(marker)))

/** The rewind payload for the newest user turn in the snapshot the CAS tuple came from. */
export function lastTurnRewind(snapshot: MutationSnapshot): null | { target_message_id: number } {
  const target = (snapshot.messages ?? []).findLast(isUserTurn)

  return target ? { target_message_id: target.row_id as number } : null
}

// sid → the last admission whose completion this view saw; `/usage` reads its committed result.
const lastAdmission = new Map<string, string>()

export function noteCanonicalCompletion(sid: null | string | undefined, admissionId: unknown) {
  if (sid && typeof admissionId === 'string' && admissionId) {
    lastAdmission.set(sid, admissionId)
  }
}

export function canonicalTitle(arg: string, ctx: SlashRunCtx) {
  const sid = ctx.sid!
  const title = arg.trim()

  if (!title) {
    // The live listing is the authority's projection of the stored title.
    return void ctx.gateway.gw
      .request<{ sessions?: Array<{ id?: string; title?: string }> }>('session.list', { limit: 200 })
      .then(r => {
        const current = (r?.sessions?.find(row => row.id === sid)?.title ?? '').trim()

        if (!ctx.stale()) {
          ctx.transcript.sys(current ? t('slashCmd.core.title.current', current) : t('slashCmd.core.title.none'))
        }
      })
      .catch(ctx.guardedErr)
  }

  void renameCanonicalSession(ctx.gateway.gw, sid, title, ctx.stale)
    .then(next => {
      if (next !== undefined && !ctx.stale()) {
        patchUiState({ sessionTitle: next })
        ctx.transcript.sys(t('slashCmd.core.title.set', next, ''))
      }
    })
    .catch(ctx.guardedErr)
}

/** `session.mutate operation=rename`; resolves to the stored (sanitized) title. */
export async function renameCanonicalSession(
  gw: SlashRunCtx['gateway']['gw'],
  sid: string,
  title: string,
  stale?: () => boolean
): Promise<string | undefined> {
  const mutation = await mutateCanonical(gw, sid, 'rename', { title }, stale)

  return mutation?.result ? String(mutation.result.title ?? title).trim() : undefined
}

/** `/undo` and `/retry`: rewind to before the newest user turn; `/retry` then resubmits its text. */
export function canonicalRewind(intent: 'retry' | 'undo', ctx: SlashRunCtx) {
  const sid = ctx.sid!

  void mutateCanonical(ctx.gateway.gw, sid, 'rewind', lastTurnRewind, ctx.stale)
    .then(mutation => {
      if (!mutation || ctx.stale()) {
        return
      }

      const result = mutation.result
      const removed = Number(result?.rewound_count ?? 0)

      if (!result || removed <= 0) {
        return ctx.transcript.sys(t(intent === 'retry' ? 'slashCmd.core.retry.nothing' : 'slashCmd.core.undo.nothing'))
      }

      ctx.transcript.setHistoryItems(prev => ctx.transcript.trimLastExchange(prev))

      if (intent === 'undo') {
        return ctx.transcript.sys(
          t(removed === 1 ? 'slashCmd.core.undo.undidOne' : 'slashCmd.core.undo.undidOther', String(removed))
        )
      }

      // Main's /retry: the rewound user turn's own text, not whatever this view typed last.
      const content = (result.target_message as undefined | { content?: unknown })?.content
      const text = typeof content === 'string' && content.trim() ? content : ctx.local.getLastUserMsg()

      return text ? ctx.transcript.send(text) : ctx.transcript.sys(t('slashCmd.core.retry.nothing'))
    })
    .catch(ctx.guardedErr)
}

interface CommittedResult {
  api_calls?: number
  estimated_cost_usd?: number
  input_tokens?: number
  model?: string
  output_tokens?: number
  total_tokens?: number
}

/** `/usage`: session totals are the owner's own `/usage` report for this session (a reviewed read,
 * as the classic CLI shows it); every settled turn also commits its own usage with the admission,
 * so the last one this view saw follows. */
export function canonicalUsage(ctx: SlashRunCtx) {
  const sid = ctx.sid!
  const admissionId = lastAdmission.get(sid)
  const unavailable = () => !ctx.stale() && ctx.transcript.sys(t('canonical.controls.usageTotalsUnavailable'))

  const totals = ctx.gateway.gw
    .request<SlashExecResponse>('slash.exec', { command: 'usage', session_id: sid })
    .then(r => {
      if (!r?.output) {
        return unavailable()
      }

      if (!ctx.stale()) {
        ctx.transcript.page(r.output, t('slashCmd.session.usage.usageTitle'))
      }
    }, unavailable)

  if (!admissionId) {
    return void totals.then(() => !ctx.stale() && ctx.transcript.sys(t('canonical.controls.usageNoTurn')))
  }

  void totals
    .then(() =>
      ctx.gateway.gw.request('prompt.receipt', { admission_id: admissionId, include_result: true, session_id: sid })
    )
    .then(raw => {
      if (ctx.stale()) {
        return
      }

      const result = asRpcResult<{ result?: CommittedResult }>(raw)?.result

      if (!result) {
        return ctx.transcript.sys(t('canonical.controls.usageNoTurn'))
      }

      const f = (value: number | undefined) => (value ?? 0).toLocaleString()

      const rows: [string, string][] = [
        [t('slashCmd.session.usage.rowModel'), result.model ?? getUiState().info?.model ?? ''],
        [t('slashCmd.session.usage.rowInputTokens'), f(result.input_tokens)],
        [t('slashCmd.session.usage.rowOutputTokens'), f(result.output_tokens)],
        [t('slashCmd.session.usage.rowTotalTokens'), f(result.total_tokens)],
        [t('slashCmd.session.usage.rowApiCalls'), f(result.api_calls)],
        ...(typeof result.estimated_cost_usd === 'number'
          ? ([[t('canonical.controls.rowCost'), `$${result.estimated_cost_usd.toFixed(4)}`]] as [string, string][])
          : [])
      ]

      const sections: PanelSection[] = [{ rows }]

      ctx.transcript.panel(t('canonical.controls.usageLastTurnTitle'), sections)
    })
    .catch(ctx.guardedErr)
}

/** `/tools`: a canonical session's toolsets are frozen at launch (`session.info.launch_request`);
 * listing and toggling them live has no canonical verb yet. */
export function canonicalTools(ctx: SlashRunCtx) {
  const sid = ctx.sid!

  void ctx.gateway.gw
    .request<{ launch_request?: { toolsets?: unknown } }>('session.info', { session_id: sid })
    .then(info => {
      if (ctx.stale()) {
        return
      }

      const toolsets = info?.launch_request?.toolsets

      if (Array.isArray(toolsets) && toolsets.length) {
        ctx.transcript.sys(t('canonical.controls.launchToolsets', toolsets.join(', ')))
      }

      ctx.transcript.sys(t('canonical.controls.notAvailable', 'tools'))
    })
    .catch(ctx.guardedErr)
}
