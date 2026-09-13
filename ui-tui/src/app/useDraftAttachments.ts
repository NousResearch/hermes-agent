import { randomUUID } from 'node:crypto'

import type { DraftAttachRequest, DraftResult } from '@hermes/shared'
import { type MutableRefObject, useCallback, useEffect, useRef } from 'react'

import type { NativeInputHandle } from '../components/textInput.js'
import { imageToken, nextImageIndex } from '../domain/attachments.js'
import type { GatewayClient } from '../gatewayClient.js'

import type { ComposerToken, StateSetter } from './interfaces.js'
import { $isBlocked, $overlayState } from './overlayStore.js'
import { $uiState, getUiState } from './uiStore.js'

// Wire limit enforced by tui_gateway.prompt_attachments._validate_draft_image_paths.
const MAX_DRAFT_IMAGES = 32

/** The browser owns bytes, never input. Only this live native handle can commit. */
export function useDraftAttachments(gw: GatewayClient, tokens: MutableRefObject<ComposerToken[]>, setTokens: StateSetter<ComposerToken[]>) {
  const input = useRef<NativeInputHandle | null>(null)
  const draftId = useRef(randomUUID())
  const requests = useRef(new Map<string, { request: DraftAttachRequest; result?: DraftResult }>())

  const publish = useCallback(() => gw.publishDraftState({
    session_id: getUiState().sid ?? '', draft_id: draftId.current,
    available: Boolean(getUiState().sid && input.current && !$isBlocked.get())
  }), [gw])

  const invalidate = useCallback(() => {
    draftId.current = randomUUID()
    requests.current.clear()
    publish()
  }, [publish])

  const setNativeInput = useCallback((handle: NativeInputHandle | null) => {
    input.current = handle

    if (!handle) {invalidate()}
    else {publish()}
  }, [invalidate, publish])

  useEffect(() => {
    const destination = () => JSON.stringify([getUiState().sid, getUiState().info?.profile_name ?? ''])
    let scope = destination()
    let blocked = $isBlocked.get()

    const stopUi = $uiState.listen(() => {
      if (scope !== destination()) {
        scope = destination()
        setTokens(prev => prev.filter(token => token.kind !== 'image' || token.source !== 'draft'))
        invalidate()
      }
    })

    const stopOverlay = $overlayState.listen(() => {
      const next = $isBlocked.get()

      if (next !== blocked) { blocked = next; invalidate() }
    })

    const isCurrent = (request: DraftAttachRequest) =>
      request.expected.session_id === getUiState().sid && request.expected.draft_id === draftId.current

    const attach = async (request: DraftAttachRequest) => {
      const previous = requests.current.get(request.request_id)

      if (previous) {
        // A retry is the same transaction, never permission to retarget an ID.
        const a = previous.request.expected
        const b = request.expected

        if (previous.request.path !== request.path || a.pty_instance !== b.pty_instance ||
          a.connection_generation !== b.connection_generation || a.session_id !== b.session_id || a.draft_id !== b.draft_id) {
          gw.publishDraftResult({ type: 'draft.result', request_id: request.request_id, identity: b,
            status: 'failed', error: 'Attachment request changed' })

          return
        }

        if (previous.result) {gw.publishDraftResult(previous.result)}

        return
      }

      const record: { request: DraftAttachRequest; result?: DraftResult } = { request }
      requests.current.set(request.request_id, record)

      const finish = (status: DraftResult['status'], detail: Partial<DraftResult> = {}) => {
        const result: DraftResult = { type: 'draft.result', request_id: request.request_id, identity: request.expected, status, ...detail }
        record.result = result

        if (status === 'failed' && requests.current.get(request.request_id) === record) {requests.current.delete(request.request_id)}
        gw.publishDraftResult(result)
      }

      if (!isCurrent(request)) {
        return finish('stale')
      }

      if (!input.current || $isBlocked.get()) {return finish('unavailable')}

      try {
        const staged = await gw.request<{ path: string; ref_text: string; image?: { name: string; mime_type: string } }>('file.attach', {
          session_id: request.expected.session_id, path: request.path
        })

        if (!isCurrent(request)) {return finish('stale')}

        if (!staged?.path || !staged.ref_text) {return finish('failed', { error: 'Could not attach file' })}

        // Check after staging, without an await before insertion: concurrent
        // completions must see earlier admissions, and failure leaves the draft intact.
        if (staged.image && tokens.current.filter(token => token.kind === 'image' && token.source === 'draft').length >= MAX_DRAFT_IMAGES) {
          return finish('failed', { error: `A draft can contain at most ${MAX_DRAFT_IMAGES} uploaded images` })
        }

        const index = nextImageIndex(tokens.current, input.current?.snapshot().value)
        const label = staged.image ? imageToken(index) : staged.ref_text

        // Generic references are complete, not path-completion prefixes.
        if (!input.current?.insert(staged.image ? label : `${label} `)) {return finish('unavailable')}

        if (staged.image) {setTokens(prev => [...prev, { index, kind: 'image', label, path: staged.path, source: 'draft' }])}
        finish('attached', { path: staged.path, label })
      } catch {
        if (!isCurrent(request)) {return finish('stale')}
        finish('failed', { error: 'Could not attach file' })
      }
    }

    gw.on('draft.request', attach)
    gw.on('draft.refresh', publish)
    gw.on('draft.disconnected', invalidate)
    publish()

    return () => {
      stopUi(); stopOverlay(); invalidate()
      gw.off('draft.request', attach); gw.off('draft.refresh', publish); gw.off('draft.disconnected', invalidate)
    }
  }, [gw, invalidate, publish, setTokens, tokens])

  return { input, invalidate, setNativeInput }
}
