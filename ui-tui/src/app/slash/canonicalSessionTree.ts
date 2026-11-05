// `/tree`: the session family around this session, drawn as an overlay that switches sessions
// through the same `session.resume` path `/resume` and `/sessions <id>` use. The family itself is
// built by the owner (`session.family` → hermes_state_family), so the Ink rows and the classic
// CLI's picker rows come from one walk and one set of kinds.

import { t } from '../../i18n/runtime.js'
import { asRpcResult } from '../../lib/rpc.js'
import type { SessionTreeNode, SessionTreeView } from '../interfaces.js'
import { patchOverlayState } from '../overlayStore.js'

import type { SlashRunCtx } from './types.js'

/** Nothing folded, continuations and subagent runs in their default (folded-away) state. */
export const DEFAULT_TREE_VIEW: SessionTreeView = {
  folded: [],
  show_compressions: false,
  show_subagents: false
}

interface SessionFamilyResult {
  nodes: SessionTreeNode[]
  root_id: string
  truncated: boolean
}

/** `/tree [--all]`: open the overlay, or report the same text the classic CLI prints. */
export function openSessionTree(ctx: SlashRunCtx, view: SessionTreeView = DEFAULT_TREE_VIEW) {
  const sid = ctx.sid

  if (!sid) {
    return ctx.transcript.sys(t('prompt.tree.empty'))
  }

  if (ctx.ui.busy) {
    return ctx.transcript.sys(t('slashCmd.core.busy.replying', 'tree'))
  }

  return load(ctx, sid, view)
}

function load(ctx: SlashRunCtx, sid: string, view: SessionTreeView) {
  const stale = () => ctx.stale() || ctx.sid !== sid

  return ctx.gateway.gw
    .request('session.family', { session_id: sid, ...view })
    .then(raw => {
      if (stale()) {
        return
      }

      const family = asRpcResult<SessionFamilyResult>(raw)

      if (!family?.nodes?.length) {
        return ctx.transcript.sys(t('prompt.tree.empty'))
      }

      patchOverlayState({
        tree: {
          nodes: family.nodes,
          onSelect: id => switchTo(ctx, id),
          onView: next => void load(ctx, sid, next),
          rootId: family.root_id,
          truncated: family.truncated,
          view
        }
      })
    })
    .catch(ctx.guardedErr)
}

/** Enter: resume that session. A refused switch (busy turn) leaves the overlay up and says why. */
function switchTo(ctx: SlashRunCtx, id: string) {
  if (id === ctx.sid) {
    return
  }

  if (ctx.session.guardBusySessionSwitch(t('slashCmd.session.sessions.busyGuardAction'))) {
    return
  }

  patchOverlayState({ tree: null })
  void ctx.session.resumeById(id)
}
