import { type MutableRefObject, useCallback, useEffect, useRef } from 'react'

import { TYPING_IDLE_MS } from '../config/timing.js'
import { attachedImageNotice } from '../domain/messages.js'
import { completionToApplyOnSubmit, looksLikeSlashCommand } from '../domain/slash.js'
import type { GatewayClient } from '../gatewayClient.js'
import type {
  InputDetectDropResponse,
  PromptSubmitResponse,
  SessionSteerResponse,
  ShellExecResponse
} from '../gatewayTypes.js'
import { asRpcResult } from '../lib/rpc.js'
import { hasInterpolation, INTERPOLATION_RE } from '../protocol/interpolation.js'
import { PASTE_SNIPPET_RE } from '../protocol/paste.js'
import type { Msg } from '../types.js'

import type { ComposerActions, ComposerRefs, ComposerState, PasteSnippet } from './interfaces.js'
import { turnController } from './turnController.js'
import { getUiState, patchUiState } from './uiStore.js'

const DOUBLE_ENTER_MS = 450
const SESSION_BUSY_RE = /session busy|waiting for model response/i

const isSessionBusyError = (e: unknown) => e instanceof Error && SESSION_BUSY_RE.test(e.message)

export const submittedValueCanUseComposerCompletion = ({
  completionCount,
  composerInput,
  value
}: {
  completionCount: number
  composerInput: string
  value: string
}): boolean => completionCount > 0 && value === composerInput

const expandSnips = (snips: PasteSnippet[]) => {
  const byLabel = new Map<string, string[]>()

  for (const { label, text } of snips) {
    const hit = byLabel.get(label)
    hit ? hit.push(text) : byLabel.set(label, [text])
  }

  return (value: string) => value.replace(PASTE_SNIPPET_RE, tok => byLabel.get(tok)?.shift() ?? tok)
}

const spliceMatches = (text: string, matches: RegExpMatchArray[], results: string[]) =>
  matches.reduceRight((acc, m, i) => acc.slice(0, m.index!) + results[i] + acc.slice(m.index! + m[0].length), text)

export function useSubmission(opts: UseSubmissionOptions) {
  const {
    appendMessage,
    composerActions,
    composerRefs,
    composerState,
    gw,
    maybeGoodVibes,
    setLastUserMsg,
    slashRef,
    submitRef,
    sys
  } = opts

  const lastEmptyAt = useRef(0)
  const typingIdleTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (typingIdleTimer.current) {
      clearTimeout(typingIdleTimer.current)
      typingIdleTimer.current = null
    }

    if (!composerState.input && !composerState.inputBuf.length) {
      turnController.relaxStreaming()

      return
    }

    if (getUiState().busy) {
      turnController.boostStreamingForTyping()
    }

    typingIdleTimer.current = setTimeout(() => {
      typingIdleTimer.current = null
      turnController.relaxStreaming()
    }, TYPING_IDLE_MS)

    return () => {
      if (typingIdleTimer.current) {
        clearTimeout(typingIdleTimer.current)
        typingIdleTimer.current = null
      }
    }
  }, [composerState.input, composerState.inputBuf])

  const send = useCallback(
    (text: string, showUserMessage = true) => {
      const expand = expandSnips(composerState.pasteSnips)

      const startSubmit = (displayText: string, submitText: string, showUserMessage = true) => {
        const sid = getUiState().sid

        if (!sid) {
          return sys('session not ready yet')
        }

        turnController.clearStatusTimer()
        maybeGoodVibes(submitText)
        setLastUserMsg(text)

        if (showUserMessage) {
          appendMessage({ role: 'user', text: displayText })
        }

        patchUiState({ busy: true, status: 'running…' })
        turnController.bufRef = ''
        turnController.interrupted = false

        gw.request<PromptSubmitResponse>('prompt.submit', { session_id: sid, text: submitText }).catch((e: Error) => {
          if (isSessionBusyError(e)) {
            composerActions.enqueue(submitText)
            patchUiState({ busy: true, status: 'queued for next turn' })

            return sys(`queued: "${submitText.slice(0, 50)}${submitText.length > 50 ? '…' : ''}"`)
          }

          sys(`error: ${e.message}`)
          patchUiState({ busy: false, status: 'ready' })
        })
      }

      const sid = getUiState().sid

      if (!sid) {
        return sys('session not ready yet')
      }

      // Always ask the backend whether this looks like a file drop.
      // The backend's _detect_file_drop handles paths with spaces, quotes,
      // Windows drive letters, and escaped characters correctly.
      gw.request<InputDetectDropResponse>('input.detect_drop', { session_id: sid, text })
        .then(r => {
          if (!r?.matched) {
            return startSubmit(text, expand(text), showUserMessage)
          }

          if (r.is_image) {
            turnController.pushActivity(attachedImageNotice(r))
          } else {
            turnController.pushActivity(`detected file: ${r.name}`)
          }

          startSubmit(r.text || text, expand(r.text || text), showUserMessage)
        })
        .catch(() => startSubmit(text, expand(text), showUserMessage))
    },
    [appendMessage, composerActions, composerState.pasteSnips, gw, maybeGoodVibes, setLastUserMsg, sys]
  )

  const shellExec = useCallback(
    (cmd: string) => {
      appendMessage({ role: 'user', text: `!${cmd}` })
      patchUiState({ busy: true, status: 'running…' })

      gw.request<ShellExecResponse>('shell.exec', { command: cmd })
        .then(raw => {
          const r = asRpcResult<ShellExecResponse>(raw)

          if (!r) {
            return sys('error: invalid response: shell.exec')
          }

          const out = [r.stdout, r.stderr].filter(Boolean).join('\n').trim()

          if (out) {
            sys(out)
          }

          if (r.code !== 0 || !out) {
            sys(`exit ${r.code}`)
          }
        })
        .catch((e: Error) => sys(`error: ${e.message}`))
        .finally(() => patchUiState({ busy: false, status: 'ready' }))
    },
    [appendMessage, gw, sys]
  )

  const interpolate = useCallback(
    (text: string, then: (result: string) => void) => {
      patchUiState({ status: 'interpolating…' })
      const matches = [...text.matchAll(new RegExp(INTERPOLATION_RE.source, 'g'))]

      Promise.all(
        matches.map(m =>
          gw
            .request<ShellExecResponse>('shell.exec', { command: m[1]! })
            .then(raw => {
              const r = asRpcResult<ShellExecResponse>(raw)

              return [r?.stdout, r?.stderr].filter(Boolean).join('\n').trim()
            })
            .catch(() => '(error)')
        )
      ).then(results => then(spliceMatches(text, matches, results)))
    },
    [gw]
  )

  const sendQueued = useCallback(
    (text: string) => {
      if (text.startsWith('!')) {
        return shellExec(text.slice(1).trim())
      }

      if (hasInterpolation(text)) {
        patchUiState({ busy: true })

        return interpolate(text, send)
      }

      send(text)
    },
    [interpolate, send, shellExec]
  )

  // Honors `display.busy_input_mode` from config.yaml (CLI parity):
  //   - 'queue'     (legacy): append to queueRef; drains on busy → false
  //   - 'steer'     : inject into the current turn via session.steer; falls
  //                   back to queue when steer is rejected (no agent / no
  //                   tool window).
  //   - 'interrupt' (default): queue the text + interrupt with `keepBusy`; the
  //                   busy→false settle edge drains it once (desktop parity).
  //                   No optimistic send → no duplicate bubble / race note.
  //
  // `opts.fallbackToFront` re-inserts at the queue head (queue-edit picks keep
  // their position); the mainline submit path appends.
  const handleBusyInput = useCallback(
    (full: string, opts: { fallbackToFront?: boolean } = {}) => {
      const live = getUiState()
      const mode = live.busyInputMode

      const enqueueText = () => {
        if (opts.fallbackToFront) {
          composerRefs.queueRef.current.unshift(full)
          composerActions.syncQueue()
        } else {
          composerActions.enqueue(full)
        }
      }

      const fallback = (note: string) => {
        enqueueText()
        sys(note)
      }

      if (mode === 'queue') {
        return composerActions.enqueue(full)
      }

      if (mode === 'steer' && live.sid) {
        gw.request<SessionSteerResponse>('session.steer', { session_id: live.sid, text: full })
          .then(raw => {
            const r = asRpcResult<SessionSteerResponse>(raw)

            if (r?.status !== 'queued') {
              fallback('steer rejected — message queued for next turn')
            }
          })
          .catch(() => fallback('steer failed — message queued for next turn'))

        return
      }

      // 'interrupt': queue + interrupt(keepBusy); the settle edge drains it once.
      enqueueText()

      if (live.sid) {
        turnController.interruptTurn({ appendMessage, gw, sid: live.sid, sys }, { keepBusy: true })
      }
    },
    [appendMessage, composerActions, composerRefs, gw, sys]
  )

  const dispatchSubmission = useCallback(
    (full: string) => {
      if (!full.trim()) {
        return
      }

      if (looksLikeSlashCommand(full)) {
        appendMessage({ kind: 'slash', role: 'system', text: full })
        composerActions.pushHistory(full)
        slashRef.current(full)
        composerActions.clearIn()

        return
      }

      if (full.startsWith('!')) {
        composerActions.clearIn()

        return shellExec(full.slice(1).trim())
      }

      const live = getUiState()

      if (!live.sid) {
        composerActions.pushHistory(full)
        composerActions.enqueue(full)
        composerActions.clearIn()

        return
      }

      const editIdx = composerRefs.queueEditRef.current
      composerActions.clearIn()

      if (editIdx !== null) {
        composerActions.replaceQueue(editIdx, full)
        const picked = composerRefs.queueRef.current.splice(editIdx, 1)[0]
        composerActions.syncQueue()
        composerActions.setQueueEdit(null)

        if (!picked || !live.sid) {
          return
        }

        if (getUiState().busy) {
          // 'interrupt' / 'steer' should reach the live turn instead of
          // silently going back to the queue.  handleBusyInput resolves
          // mode-specific behavior (interrupt-and-send, steer, or queue).
          if (getUiState().busyInputMode === 'queue') {
            composerRefs.queueRef.current.unshift(picked)

            return composerActions.syncQueue()
          }

          return handleBusyInput(picked, { fallbackToFront: true })
        }

        return sendQueued(picked)
      }

      composerActions.pushHistory(full)

      if (getUiState().busy) {
        return handleBusyInput(full)
      }

      if (hasInterpolation(full)) {
        patchUiState({ busy: true })

        return interpolate(full, send)
      }

      send(full)
    },
    [appendMessage, composerActions, composerRefs, handleBusyInput, interpolate, send, sendQueued, shellExec, slashRef]
  )

  const submit = useCallback(
    (value: string) => {
      if (
        submittedValueCanUseComposerCompletion({
          completionCount: composerState.completions.length,
          composerInput: composerState.input,
          value
        })
      ) {
        const row = composerState.completions[composerState.compIdx]
        const next = completionToApplyOnSubmit(value, row?.text, composerState.compReplace)

        if (next !== null) {
          return composerActions.setInput(next)
        }
      }

      if (!value.trim() && !composerState.inputBuf.length) {
        const live = getUiState()
        const now = Date.now()
        const doubleTap = now - lastEmptyAt.current < DOUBLE_ENTER_MS
        lastEmptyAt.current = now

        if (doubleTap && live.busy && live.sid) {
          // Force-send: keep busy when a message is queued so the settle edge
          // drains it once (no race). Empty queue = plain Stop → 'ready'.
          const hasQueued = composerRefs.queueRef.current.length > 0

          return turnController.interruptTurn({ appendMessage, gw, sid: live.sid, sys }, { keepBusy: hasQueued })
        }

        if (doubleTap && live.sid && composerRefs.queueRef.current.length) {
          const next = composerActions.dequeue()

          composerActions.syncQueue()

          if (next) {
            composerActions.setQueueEdit(null)
            dispatchSubmission(next)
          }
        }

        return
      }

      lastEmptyAt.current = 0

      if (value.endsWith('\\')) {
        composerActions.setInputBuf(prev => [...prev, value.slice(0, -1)])

        return composerActions.setInput('')
      }

      dispatchSubmission([...composerState.inputBuf, value].join('\n'))
    },
    [appendMessage, composerActions, composerRefs, composerState, dispatchSubmission, gw, sys]
  )

  submitRef.current = submit

  return { dispatchSubmission, send, sendQueued, submit }
}

export interface UseSubmissionOptions {
  appendMessage: (msg: Msg) => void
  composerActions: ComposerActions
  composerRefs: ComposerRefs
  composerState: ComposerState
  gw: GatewayClient
  maybeGoodVibes: (text: string) => void
  setLastUserMsg: (value: string) => void
  slashRef: MutableRefObject<(cmd: string) => boolean>
  submitRef: MutableRefObject<(value: string) => void>
  sys: (text: string) => void
}
