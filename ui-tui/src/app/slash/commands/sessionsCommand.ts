import { patchUiState, getUiState, $uiState } from '../../uiStore.js'
import { t } from '../../../i18n/runtime.js'
import { asRpcResult, rpcErrorMessage } from '../../../lib/rpc.js'
import { patchOverlayState } from '../../overlayStore.js'
import { mutateCanonical, type MutationSnapshot } from '../canonicalSessionControls.js'
import type { SlashRunCtx } from '../types.js'

type Row = {
  id: string
  resolved_id?: string
  stored_session_id?: string
  title?: string
  source?: string
  model?: string
  message_count?: number
  started_at?: number
  running?: boolean
}
const scopes = ['live', 'stored'] as const
const invalidTarget = (id: string | undefined) =>
  !id || id.startsWith('--') || /[\0\r\n]/.test(id) || ['all', '*'].includes(id) || id.includes(',')

async function readFeed(ctx: SlashRunCtx, scope: (typeof scopes)[number]) {
  const result = asRpcResult<{ sessions: Row[] }>(await ctx.gateway.gw.request('session.list', { scope, limit: 200 }))
  if (!Array.isArray(result?.sessions)) throw new Error('Invalid session.list response')
  return result.sessions
}

async function list(ctx: SlashRunCtx) {
  const feeds = await Promise.allSettled(scopes.map(scope => readFeed(ctx, scope)))
  if (ctx.stale()) return
  ctx.transcript.sys(t('slashCmd.session.sessions.inventory'))
  const rows = new Map<string, { row: Row; live: boolean; ids: Set<string> }>()
  feeds.forEach((feed, index) => {
    if (feed.status === 'rejected') {
      ctx.transcript.sys(t('slashCmd.session.sessions.feedError', scopes[index]!, rpcErrorMessage(feed.reason)))
      return
    }
    for (const row of feed.value) {
      const key = row.resolved_id || row.stored_session_id || row.id
      const existing = rows.get(key)
      if (existing) existing.ids.add(row.id)
      else rows.set(key, { row, live: index === 0, ids: new Set([row.id]) })
    }
  })
  for (const [owner, { row, live, ids }] of rows) {
    if (!ids.has(owner)) ids.add(owner)
    ctx.transcript.sys(
      t(
        'slashCmd.session.sessions.row',
        [...ids].join(' / '),
        row.title || '—',
        row.source || '—',
        row.model || '—',
        String(row.message_count ?? '—'),
        row.started_at ? new Date(row.started_at * 1000).toISOString() : '—',
        live ? 'live' : '',
        row.running ? 'running' : ''
      )
    )
  }
  if (!rows.size && feeds.some(feed => feed.status === 'fulfilled'))
    ctx.transcript.sys(t('slashCmd.session.sessions.empty'))
  if (feeds.some(feed => feed.status === 'fulfilled')) ctx.ran?.()
}

/** Per gateway/view UI guards only. All CAS/idempotency state stays in mutateCanonical on the
 * original gateway: a component helper's private proxy would split rename/delete retry retirement. */
const views = new WeakMap<object, WeakMap<object, ReturnType<typeof createActions>>>()
function createActions() {
  const state = {
    live: new Set<string>(),
    acquired: new Set<string>(),
    pending: false,
    resolvedDeletes: new Map<string, string>()
  }
  const keepView = (id: string) => {
    const current = getUiState()
    return id === current.sid || id === current.info?.stored_session_id || state.live.has(id) || state.acquired.has(id)
  }
  return { state, keepView }
}
function actionsFor(ctx: SlashRunCtx) {
  let gatewayViews = views.get(ctx.gateway.gw)
  if (!gatewayViews) views.set(ctx.gateway.gw, (gatewayViews = new WeakMap()))
  let view = gatewayViews.get(ctx.slashFlightRef)
  if (!view) gatewayViews.set(ctx.slashFlightRef, (view = createActions()))
  return view
}
function trackAcquiredViews(view: ReturnType<typeof createActions>) {
  // A real switch acquires a hold that can stay live after another switch. Observe UI identities,
  // not the action's resume (which deliberately never changes UI identity).
  view.state.acquired.clear()
  const remember = () => {
    const current = getUiState()
    for (const id of [current.sid, current.info?.stored_session_id]) if (id) view.state.acquired.add(id)
  }
  remember()
  return $uiState.listen(remember)
}
async function rename(id: string, title: string, ctx: SlashRunCtx) {
  const view = actionsFor(ctx)
  if (view.state.pending) return
  view.state.pending = true
  const stopTracking = trackAcquiredViews(view)
  let snapshot: MutationSnapshot | undefined
  let alreadyHeld = false
  let refresh = false
  try {
    // Read the owner roster, not a stale picker snapshot, before deciding which holds to preserve.
    const live = await readFeed(ctx, 'live')
    view.state.live = new Set(
      live.flatMap(row => [row.id, row.resolved_id, row.stored_session_id].filter((id): id is string => Boolean(id)))
    )
    if (ctx.stale()) return
    alreadyHeld = view.keepView(id)
    const mutation = await mutateCanonical(ctx.gateway.gw, id, 'rename', { title }, ctx.stale, read => {
      snapshot = read
    })
    if (ctx.stale() || !mutation?.result) return
    const storedTitle = String(mutation.result.title ?? title).trim()
    const current = getUiState()
    const resolved = snapshot?.session_id || id
    if (resolved === current.sid || id === current.sid || id === current.info?.stored_session_id)
      patchUiState({ sessionTitle: storedTitle })
    ctx.transcript.sys(t('slashCmd.session.sessions.renamed', id, storedTitle))
    ctx.ran?.()
    refresh = true
  } catch (error) {
    if (!ctx.stale()) ctx.transcript.sys(t('slashCmd.session.sessions.mutationError', rpcErrorMessage(error)))
  } finally {
    try {
      const resolved = snapshot?.session_id || id
      if (snapshot?.subscription_id && !alreadyHeld && !view.keepView(resolved)) {
        await ctx.gateway.gw.request('session.detach', {
          session_id: resolved,
          subscription_id: snapshot.subscription_id
        })
      }
    } catch (error) {
      if (!ctx.stale()) ctx.guardedErr(error)
    } finally {
      stopTracking()
      view.state.pending = false
    }
  }
  if (refresh && !ctx.stale()) await list(ctx).catch(ctx.guardedErr)
}

async function deleteSession(id: string, ctx: SlashRunCtx) {
  const view = actionsFor(ctx)
  if (view.state.pending || ctx.stale()) return
  view.state.pending = true
  const stopTracking = trackAcquiredViews(view)
  let snapshot: MutationSnapshot | undefined
  let retired = false
  try {
    const live = await readFeed(ctx, 'live')
    view.state.live = new Set(
      live.flatMap(row => [row.id, row.resolved_id, row.stored_session_id].filter((id): id is string => Boolean(id)))
    )
    if (ctx.stale()) return
    if (view.keepView(id) || view.keepView(view.state.resolvedDeletes.get(id) || id))
      throw new Error(t('slashCmd.session.sessions.held'))
    // The existing canonical engine retains exact UUID/CAS on a lost reply. This adapter only
    // observes the attachment and rejects a resolved current/live alias before that engine writes.
    const mutation = await mutateCanonical(ctx.gateway.gw, id, 'delete', {}, ctx.stale, read => {
      snapshot = read
      const resolved = read.session_id || id
      view.state.resolvedDeletes.set(id, resolved)
      if (view.keepView(resolved)) throw new Error(t('slashCmd.session.sessions.held'))
    })
    if (!mutation) return
    if (!Array.isArray(mutation.result?.deleted_ids) || !mutation.result.deleted_ids.length)
      throw new Error('Invalid session.mutate delete response')
    retired = true
    view.state.resolvedDeletes.delete(id)
    if (ctx.stale()) return
    ctx.transcript.sys(t('slashCmd.session.sessions.deleted', id))
    ctx.ran?.()
    await list(ctx)
  } catch (error) {
    if (!ctx.stale()) ctx.transcript.sys(t('slashCmd.session.sessions.mutationError', rpcErrorMessage(error)))
  } finally {
    try {
      const resolved = snapshot?.session_id || id
      if (!retired && snapshot?.subscription_id && !view.keepView(resolved)) {
        await ctx.gateway.gw.request('session.detach', {
          session_id: resolved,
          subscription_id: snapshot.subscription_id
        })
      }
    } catch (error) {
      if (!ctx.stale()) ctx.guardedErr(error)
    } finally {
      stopTracking()
      view.state.pending = false
    }
  }
}

/** Only the actual /sessions spelling reserves management words. */
export function runSessionsManagement(arg: string, ctx: SlashRunCtx, cmd: string): boolean {
  if (cmd.match(/^\/([^\s]+)/)?.[1]?.toLowerCase() !== 'sessions') return false
  const [verb, ...rest] = arg.trim().split(/\s+/)
  if (verb === 'browse') {
    if (rest.length) ctx.transcript.sys(t('slashCmd.session.sessions.usage', 'browse'))
    else {
      patchOverlayState({ sessions: true })
      ctx.ran?.()
    }
    return true
  }
  if (verb === 'prune') {
    const seen = new Set<string>()
    let olderThan = ''
    let valid = true
    for (let i = 0; i < rest.length; i++) {
      const flag = rest[i]!
      if (seen.has(flag) || !['--older-than', '--dry-run', '--yes'].includes(flag)) {
        valid = false
        break
      }
      seen.add(flag)
      if (flag === '--older-than') {
        olderThan = rest[++i] ?? ''
        const match = olderThan
          .toLowerCase()
          .match(
            /^(\d+(?:\.\d+)?)(s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days|w|wk|wks|week|weeks)?$/
          )
        const unit = match?.[2]?.[0] ?? 'd'
        const seconds = Number(match?.[1]) * ({ s: 1, m: 60, h: 3600, d: 86400, w: 604800 }[unit] ?? 0)
        if (!match || !Number.isFinite(seconds)) {
          valid = false
          break
        }
      }
    }
    ctx.transcript.sys(
      valid
        ? t('slashCmd.session.sessions.pruneRefusal', olderThan ? ` --older-than ${olderThan}` : '')
        : t('slashCmd.session.sessions.usage', 'prune [--older-than <duration>] [--dry-run] [--yes]')
    )
    return true
  }
  if (verb === 'delete') {
    const id = rest[0]
    if (!id || invalidTarget(id) || rest.length > 2 || (rest.length === 2 && rest[1] !== '--yes')) {
      ctx.transcript.sys(t('slashCmd.session.sessions.usage', 'delete <exactID> [--yes]'))
    } else if (!ctx.gateway.gw.isCanonical) {
      ctx.transcript.sys(t('slashCmd.session.sessions.canonicalRequired'))
    } else if (id === getUiState().sid || id === getUiState().info?.stored_session_id) {
      ctx.transcript.sys(t('slashCmd.session.sessions.held'))
    } else if (rest[1] === '--yes') {
      void deleteSession(id, ctx)
    } else {
      let confirmed = false
      patchOverlayState({
        confirm: {
          title: t('slashCmd.session.sessions.confirmTitle', id),
          detail: t('slashCmd.session.sessions.confirmDetail', id),
          danger: true,
          cancelLabel: t('slashCmd.session.sessions.cancel'),
          confirmLabel: t('slashCmd.session.sessions.delete'),
          onConfirm: () => {
            if (confirmed || ctx.stale()) return
            confirmed = true
            void deleteSession(id, ctx)
          }
        }
      })
    }
    return true
  }
  if (verb === 'rename') {
    const match = arg.trim().match(/^rename\s+(\S+)\s+([\s\S]+)$/)
    let title = match?.[2]?.trim() ?? ''
    try {
      if (title.startsWith('"')) title = JSON.parse(title)
      if (
        !match ||
        invalidTarget(match[1]) ||
        /[\r\n\0]/.test(cmd) ||
        typeof title !== 'string' ||
        !title.trim() ||
        /[\r\n\0]/.test(title)
      )
        throw new Error('title')
    } catch {
      ctx.transcript.sys(t('slashCmd.session.sessions.usage', 'rename <exactID> <title remainder or JSON string>'))
      return true
    }
    if (!ctx.gateway.gw.isCanonical) ctx.transcript.sys(t('slashCmd.session.sessions.canonicalRequired'))
    else void rename(match[1]!, title, ctx)
    return true
  }
  if (verb !== 'list' && verb !== 'ls') return false
  if (rest.length) ctx.transcript.sys(t('slashCmd.session.sessions.usage', 'list'))
  else void list(ctx).catch(ctx.guardedErr)
  return true
}
