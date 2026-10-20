import { randomUUID } from 'node:crypto'

import { introMsg, toTranscriptMessages } from '../../domain/messages.js'
import { CANONICAL_REASONING_EFFORTS, TUI_SESSION_MODEL_FLAG } from '../../domain/slash.js'
import { t } from '../../i18n/runtime.js'
import { asRpcResult } from '../../lib/rpc.js'
import { patchOverlayState } from '../overlayStore.js'
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
    throw new Error(t('canonical.controls.modelUsage'))
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

    // A picker-issued level (#138856) rides in the same selection: a busy session defers the pick
    // and commits model and level together. An unknown level is refused like any other flag.
    if (flag === '--reasoning' && CANONICAL_REASONING_EFFORTS.has(parts[0]?.toLowerCase() ?? '')) {
      payload.reasoning = parts.shift()!.toLowerCase()

      continue
    }

    throw new Error(t('canonical.controls.unsupportedModelOption', flag))
  }

  return payload
}

export type CanonicalOperation = 'model' | 'branch' | 'compress' | 'delete' | 'rename' | 'rewind' | 'reasoning'

/** The snapshot a mutation's CAS tuple was read from; `payload` may derive its target from it. */
export interface MutationSnapshot {
  messages?: Array<Record<string, unknown>>
  /** The stored rows (content + row_id) behind a display-projected `messages`. */
  stored_messages?: Array<Record<string, unknown>>
  revision: number
  execution_generation: number
  /** The viewer attachment the revision read made (`session.resume` attaches). */
  subscription_id?: string
  /** The id the owner resolved the read to and subscribed under; may differ from the one sent. */
  session_id?: string
}

/** The CAS tuple read (`session.resume`, which attaches this client as a viewer). */
async function readCasTuple(gw: MutationGateway, sid: string, onRead?: (snapshot: MutationSnapshot) => void) {
  const snapshot = asRpcResult<MutationSnapshot>(await gw.request('session.resume', { session_id: sid }))

  if (snapshot) {
    onRead?.(snapshot)
  }

  return snapshot
}

/** One revision-fenced `session.mutate` (`gateway/session_mutations.py`). The CAS tuple comes from a
 * fresh `session.resume`; `payload` may be a function of that same snapshot (a rewind names its
 * target row from it) and return null for "nothing to do". Metadata edits (`rename`) carry no
 * generation fence, matching Desktop's `retainedMutation(..., withGeneration=false)`. */
/** Only a session's very next control may retry an ambiguous one: any other verb retires the rest. */
export function retireCanonicalControls(gw: MutationGateway, sid: string, keep?: string) {
  const requests = pending.get(gw)

  for (const key of [...(requests?.keys() ?? [])]) {
    if (key !== keep && JSON.parse(key)[0] === sid) {
      requests!.delete(key)
    }
  }
}

export async function mutateCanonical(
  gw: MutationGateway,
  sid: string,
  operation: CanonicalOperation,
  payload: Record<string, unknown> | ((snapshot: MutationSnapshot) => null | Record<string, unknown>),
  stale: () => boolean = () => false,
  onRead?: (snapshot: MutationSnapshot) => void
) {
  let requests = pending.get(gw)

  if (!requests) {
    requests = new Map()
    pending.set(gw, requests)
  }

  // A derived payload is keyed by operation alone: a lost-reply retry must re-present the
  // original target, never re-derive one from a transcript the first attempt already rewound.
  const key = JSON.stringify([sid, operation, typeof payload === 'function' ? null : payload])
  retireCanonicalControls(gw, sid, key)
  let params = requests.get(key)

  if (!params) {
    const snapshot = await readCasTuple(gw, sid, onRead)

    if (stale()) {
      return
    }

    if (!Number.isSafeInteger(snapshot?.revision) || !Number.isSafeInteger(snapshot?.execution_generation)) {
      throw new Error(t('canonical.controls.identityUnavailable'))
    }

    const body = typeof payload === 'function' ? payload(snapshot!) : payload

    if (!body) {
      return { result: null, expectedGeneration: snapshot!.execution_generation }
    }

    params = {
      session_id: sid,
      request_id: randomUUID(),
      expected_revision: snapshot!.revision,
      ...(operation === 'rename' ? {} : { expected_generation: snapshot!.execution_generation }),
      operation,
      payload: body
    }
    requests.set(key, params)
  }

  let result

  try {
    result = asRpcResult(await gw.request('session.mutate', params))

    if (!result) {
      throw new Error(t('session.common.invalidResponse', 'session.mutate'))
    }

    requests.delete(key)
  } catch (error) {
    // A structured authority refusal is definitive; transport errors are not.
    if ((error as { data?: { reason?: string } })?.data?.reason) {
      requests.delete(key)
    }

    throw error
  }

  return { result, expectedGeneration: (params.expected_generation ?? result.execution_generation) as number }
}

export async function mutateCanonicalSession(
  gw: MutationGateway,
  sid: string,
  operation: 'model' | 'branch' | 'compress',
  arg: string,
  stale: () => boolean = () => false,
  confirm?: string
) {
  // `confirm`: the owner's one-time token for a guarded model target the user accepted. It keys a
  // separate retained request, so an ambiguous reply to the confirmed send retries that exact one.
  const payload = {
    ...(operation === 'model' ? modelPayload(arg) : arg ? { [operation === 'branch' ? 'title' : 'focus']: arg } : {}),
    ...(confirm ? { confirm } : {})
  }

  const mutation = await mutateCanonical(gw, sid, operation, payload, stale)

  return mutation?.result ? { result: mutation.result, expectedGeneration: mutation.expectedGeneration } : undefined
}

type CanonicalControlResult = NonNullable<Awaited<ReturnType<typeof mutateCanonicalSession>>>['result']

interface ModelConfirmation {
  confirm: string
  confirm_message?: string
  status: 'confirmation_required'
  target_model?: string
}

const needsModelConfirmation = (result: CanonicalControlResult): result is ModelConfirmation =>
  result.status === 'confirmation_required' && typeof result.confirm === 'string'

/** The owner refused a guarded model target (cost / data policy / large context) and wrote
 *  nothing. Ask with the same dialog the legacy `config.set` path used; only "switch anyway"
 *  re-sends, once, with the owner's token. A confirmed send refused again (a turn or another
 *  switch landed first) is reported, never re-asked in a loop. */
function askModelConfirmation(refusal: ModelConfirmation, arg: string, ctx: SlashRunCtx, confirmed: boolean) {
  if (confirmed) {
    throw new Error(`${refusal.confirm_message ?? ''}\n\n${t('slashCmd.session.model.confirmStale')}`.trim())
  }

  openModelConfirmation(
    refusal.confirm_message,
    refusal.target_model ?? arg.trim().split(/\s+/)[0] ?? '',
    () => void runCanonicalSessionControl('model', arg, ctx, refusal.confirm)
  )
}

/** The guarded-model dialog: an idle pick's refusal and a deferred pick's boundary ask share it. */
export function openModelConfirmation(detail: string | undefined, target: string, onConfirm: () => void) {
  patchOverlayState({
    confirm: {
      cancelLabel: t('slashCmd.session.model.cancel'),
      confirmLabel: t('slashCmd.session.model.switchAnyway'),
      danger: true,
      detail: detail || t('slashCmd.session.model.expensiveDetail'),
      onConfirm,
      title: t('slashCmd.session.model.confirmTitle', target)
    }
  })
}

function isSupersededControl(
  operation: 'model' | 'branch' | 'compress',
  result: CanonicalControlResult,
  expectedGeneration: number,
  ctx: SlashRunCtx
) {
  const current = getUiState().info

  return (
    operation !== 'branch' &&
    (current?.execution_epoch !== ctx.ui.info?.execution_epoch ||
      (current?.execution_generation ?? 0) > (result.execution_generation ?? expectedGeneration))
  )
}

async function applyBranchResult(result: CanonicalControlResult, arg: string, ctx: SlashRunCtx) {
  if (!result.branched_session_id) {
    throw new Error(t('session.common.invalidResponse', 'branch'))
  }

  // The child resume repaints the transcript; main's "branched →" line is printed after it, or the
  // reset wipes it.
  const child = result.branched_session_id as string

  await ctx.session.resumeById(child)

  // Main named the child by its title ("Slash lane #2" for an untitled branch of "Slash lane").
  const info = await ctx.gateway.gw
    .request<{ title?: null | string }>('session.info', { session_id: child })
    .catch(() => null)

  // The resume moved this view to the child, so the slash's own staleness check no longer applies:
  // print only while the child is still the view's session.
  if (getUiState().sid === child) {
    ctx.transcript.sys(t('slashCmd.session.branch.branched', asRpcResult(info)?.title?.trim() || arg || child))
  }
}

function applyModelResult(result: CanonicalControlResult, ctx: SlashRunCtx) {
  if (!result.model) {
    throw new Error(t('session.main.invalidModelSwitchResponse'))
  }

  if (result.status === 'deferred') {
    // A turn is running: the owner keeps the pick for the next turn, and session.info repaints the
    // model when it lands (or a refusal / confirmation follows on session.updated).
    ctx.transcript.sys(t('slashCmd.session.model.switchedDeferred', result.model))

    return
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
  ctx.transcript.sys(t('session.main.modelSwitched', result.model))

  if (typeof result.effort === 'string') {
    // The level committed in the same selection (a deferred result carries none yet).
    ctx.transcript.sys(t('slashCmd.session.reasoning.current', result.effort))
  }
}

type CompressSummary = { headline?: string; noop?: boolean; note?: string; token_line?: string } | null

/** The authority's report (headline, token line, note), as the native /compress prints it. */
function printCompressSummary(summary: CompressSummary, ctx: SlashRunCtx) {
  ctx.transcript.sys(
    summary?.headline ? `${summary.noop ? '' : '✓ '}${summary.headline}` : t('canonical.controls.compressed')
  )

  for (const line of [summary?.token_line, summary?.note]) {
    if (line) {
      ctx.transcript.sys(`  ${line}`)
    }
  }
}

async function applyCompressResult(gw: MutationGateway, sid: string, result: CanonicalControlResult, ctx: SlashRunCtx) {
  const before = getUiState()
  const snapshot = asRpcResult(await gw.request('session.resume', { session_id: sid }))

  if (ctx.stale() || getUiState().info !== before.info || getUiState().busy !== before.busy) {
    return
  }

  if (!snapshot || !Array.isArray(snapshot.messages)) {
    throw new Error(t('session.common.invalidResponse', 'compressed transcript'))
  }

  const info = { ...before.info, ...snapshot.info }

  if (
    info.execution_epoch !== before.info?.execution_epoch ||
    (info.execution_generation ?? -1) < (result.execution_generation ?? 0)
  ) {
    throw new Error(t('canonical.controls.compressedStale'))
  }

  ctx.transcript.setHistoryItems([introMsg(info), ...toTranscriptMessages(snapshot.messages)])
  patchUiState({ info })
  printCompressSummary(result.summary as CompressSummary, ctx)
}

export async function runCanonicalSessionControl(
  operation: 'model' | 'branch' | 'compress',
  arg: string,
  ctx: SlashRunCtx,
  confirm?: string
) {
  const gw = ctx.gateway.gw

  try {
    if (!ctx.sid) {
      throw new Error(t('slashCmd.core.status.noActiveSession'))
    }

    const mutation = await mutateCanonicalSession(gw, ctx.sid, operation, arg, ctx.stale, confirm)

    if (!mutation) {
      return
    }

    const { result, expectedGeneration } = mutation

    if (ctx.stale()) {
      return
    }

    if (isSupersededControl(operation, result, expectedGeneration, ctx)) {
      return
    }

    if (operation === 'model' && needsModelConfirmation(result)) {
      askModelConfirmation(result, arg, ctx, Boolean(confirm))
    } else if (operation === 'branch') {
      await applyBranchResult(result, arg, ctx)
    } else if (operation === 'model') {
      applyModelResult(result, ctx)
    } else if (result.status === 'preview') {
      // `--preview` is a read-only report; nothing to re-hydrate.
      ctx.transcript.sys((result.lines as string[]).join('\n'))
    } else if (result.status === 'noop') {
      // Nothing removable: main's "No changes from compression" report; the transcript is untouched.
      printCompressSummary(result.summary as CompressSummary, ctx)
    } else {
      await applyCompressResult(gw, ctx.sid, result, ctx)
    }
  } catch (error) {
    ctx.guardedErr(error)
  }
}
