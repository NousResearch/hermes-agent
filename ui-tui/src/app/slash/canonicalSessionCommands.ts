// Ink slash commands whose legacy RPCs (`session.title`, `session.undo`, `session.usage`, the
// `/tools` slash worker) the canonical gateway does not serve, mapped onto what it does serve:
// `session.mutate` rename / rewind (`gateway/session_mutations.py`), the committed admission
// result (`prompt.receipt include_result`) and the frozen launch policy (`session.info`).

import { CANONICAL_REASONING_EFFORTS } from '../../domain/slash.js'
import { usageFromTick } from '../../domain/usage.js'
import { t } from '../../i18n/runtime.js'
import { asRpcResult } from '../../lib/rpc.js'
import { patchUiState } from '../uiStore.js'

import { mutateCanonical, type MutationSnapshot } from './canonicalSessionControls.js'
import type { SlashRunCtx } from './types.js'
import { usageTotalsSections } from './usagePanel.js'

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
  const target = (snapshot.stored_messages ?? snapshot.messages ?? []).findLast(isUserTurn)

  return target ? { target_message_id: target.row_id as number } : null
}

const TEXT_PARTS = new Set(['input_text', 'output_text', 'text'])

/** Lossless /retry text (agent/context_compressor_retry.py::retryable_user_text): '' when the turn has
 * no text, null when it carries media, unknown or annotated parts (no replay protocol: fail closed). */
function retryText(content: unknown): null | string {
  let text = ''

  for (const part of typeof content === 'string' ? [content] : Array.isArray(content) ? content : [null]) {
    if (typeof part === 'string') {
      text += part
    } else if (
      part?.type &&
      TEXT_PARTS.has(part.type) &&
      typeof part.text === 'string' &&
      Object.keys(part).length === 2
    ) {
      text += part.text
    } else {
      return null
    }
  }

  return text.trim() ? text : ''
}

/** /retry's rewind payload: validated on the CAS snapshot, so a refusal archives nothing. */
export function retryTurnRewind(snapshot: MutationSnapshot): null | { target_message_id: number } {
  const target = (snapshot.stored_messages ?? snapshot.messages ?? []).findLast(isUserTurn)
  const text = target ? retryText(target.content) : ''

  if (text === null) {
    throw new Error(t('canonical.controls.retryUnsupported'))
  }

  return target && text ? { target_message_id: target.row_id as number } : null
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

/** Session-scope `/reasoning <level>`: the owner's revision-fenced `reasoning` mutation of the frozen
 * policy. False when the shared gateway has no verb for it (`--global`, a display word). */
export function canonicalReasoning(payload: { scope?: string; value: string }, ctx: SlashRunCtx): boolean {
  const effort = payload.value.trim().toLowerCase()

  if (payload.scope === 'global' || !CANONICAL_REASONING_EFFORTS.has(effort)) {
    return false
  }

  void mutateCanonical(ctx.gateway.gw, ctx.sid!, 'reasoning', { effort }, ctx.stale)
    .then(
      mutation =>
        mutation?.result && !ctx.stale() && ctx.transcript.sys(t('slashCmd.session.reasoning.current', effort))
    )
    .catch(ctx.guardedErr)

  return true
}

/** `/undo` and `/retry`: rewind to before the newest user turn; `/retry` then resubmits its text. */
export function canonicalRewind(intent: 'retry' | 'undo', ctx: SlashRunCtx) {
  const sid = ctx.sid!

  void mutateCanonical(ctx.gateway.gw, sid, 'rewind', intent === 'retry' ? retryTurnRewind : lastTurnRewind, ctx.stale)
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
      const text = retryText((result.target_message as undefined | { content?: unknown })?.content)

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

/** `/usage`: main's session totals panel. The owner's `session.stats` is the same session-cumulative
 * dict its `session.usage` ticks carry (`gateway/session_usage.py`), mapped onto main's `Usage` shape
 * and rendered exactly as main's sidecar `/usage` did (Usage panel, or "no API calls yet", then the
 * plan CTA). */
export function canonicalUsage(ctx: SlashRunCtx) {
  const sid = ctx.sid!

  void ctx.gateway.gw
    .request('session.stats', { session_id: sid })
    .then(raw => {
      if (ctx.stale()) {
        return
      }

      const usage = usageFromTick(asRpcResult<Record<string, unknown>>(raw))

      if (!usage?.calls) {
        ctx.transcript.sys(t('slashCmd.session.usage.noCalls'))
        ctx.transcript.sys(t('slashCmd.session.usage.cta'))

        return
      }

      patchUiState(state => ({ ...state, usage: { ...state.usage, ...usage } }))
      ctx.transcript.panel(t('slashCmd.session.usage.usageTitle'), usageTotalsSections(usage))
      ctx.transcript.sys(t('slashCmd.session.usage.cta'))
    })
    .catch(ctx.guardedErr)
}

interface ToolsShowResult {
  sections?: { name: string; tools?: { description?: string; name: string }[] }[]
  total?: number
}

/** Main's `/tools` text (hermes_cli/cli_info_mixin.py show_tools): kawaii header box, one
 *  `[toolset]` block per section with the first description sentence, then the total. */
export function renderToolsListing(result: ToolsShowResult): string {
  const sections = result.sections ?? []

  if (!sections.some(section => section.tools?.length)) {
    return '(;_;) No tools available'
  }

  const title = '(^_^)/ Available Tools'
  const pad = 78 - title.length
  const rule = `+${'-'.repeat(78)}+`
  const lines = [rule, `|${' '.repeat(Math.floor(pad / 2))}${title}${' '.repeat(pad - Math.floor(pad / 2))}|`, rule, '']

  for (const section of sections) {
    lines.push(`  [${section.name}]`)

    for (const tool of section.tools ?? []) {
      lines.push(`    * ${tool.name.padEnd(20)} - ${tool.description ?? ''}`)
    }

    lines.push('')
  }

  lines.push(`  Total: ${result.total ?? 0} tools  ヽ(^o^)ノ`)

  return lines.join('\n')
}

/** `/tools`: the session's catalog from the owner's `tools.show` (its toolsets, MCP and plugin
 *  tools), rendered as main's listing in the pager. */
export function canonicalTools(ctx: SlashRunCtx) {
  const sid = ctx.sid!

  void ctx.gateway.gw
    .request<ToolsShowResult>('tools.show', { session_id: sid })
    .then(raw => {
      if (ctx.stale()) {
        return
      }

      ctx.transcript.page(
        renderToolsListing(asRpcResult<ToolsShowResult>(raw) ?? {}),
        t('slashCmd.ops.slashWorker.toolsTitle')
      )
    })
    .catch(ctx.guardedErr)
}
