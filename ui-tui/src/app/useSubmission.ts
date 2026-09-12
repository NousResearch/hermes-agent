import { type MutableRefObject, useCallback, useEffect, useRef } from 'react'

import { TYPING_IDLE_MS } from '../config/timing.js'
import { expandTokens, imageAttachments } from '../domain/attachments.js'
import { completionToApplyOnSubmit, looksLikeSlashCommand, parseSlashCommand } from '../domain/slash.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { ShellExecResponse } from '../gatewayTypes.js'
import { queueItem, type QueueItem } from '../hooks/useQueue.js'
import { savePendingInput } from '../lib/pendingInputs.js'
import { asRpcResult } from '../lib/rpc.js'
import { hasInterpolation, INTERPOLATION_RE } from '../protocol/interpolation.js'
import type { Msg } from '../types.js'

import type { ComposerActions, ComposerRefs, ComposerState, ComposerToken, SlashHandler } from './interfaces.js'
import { submitPrompt } from './submissionCore.js'
import { captureDestination, isCurrentDestination, type SubmissionDestination } from './submissionDestination.js'
import { turnController } from './turnController.js'
import { getUiState, patchUiState } from './uiStore.js'

const DOUBLE_ENTER_MS = 450

const spliceMatches = (text: string, matches: RegExpMatchArray[], results: string[]) =>
  matches.reduceRight((acc, m, i) => acc.slice(0, m.index!) + results[i] + acc.slice(m.index! + m[0].length), text)

export const expandPasteTokens = (tokens: ComposerToken[]) =>
  expandTokens(tokens.filter(token => token.kind === 'paste'))

const slashArgument = (command: string) => /^\/\S+\s+([\s\S]+)$/.exec(command)?.[1] ?? ''

export const queueItemFromSlash = (displayCommand: string, expandedCommand: string): QueueItem | undefined => {
  const display = slashArgument(displayCommand)

  if (!display.trim()) {
    return undefined
  }

  return queueItem(slashArgument(expandedCommand), display)
}

export const prepareSubmission = (display: string, tokens: ComposerToken[]) => ({
  display,
  text: expandTokens(tokens)(display),
  attachments: imageAttachments(display, tokens)
})

/**
 * Split a slash submission into the two things it has to be at once.
 *
 * A slash command's argument is ordinary user text, so a collapsed paste in it
 * must resolve BEFORE the command runs — otherwise `/pr-triage [[ … [412 lines]
 * … ]]` hands the skill the label and the agent faithfully reports that the
 * paste is truncated. The transcript still shows the compact form, because a
 * 412-line paste inlined into the scrollback is exactly what collapsing it was
 * for.
 *
 * Image tokens stay as labels: the gateway already holds those files in
 * `attached_images` and splices them in at submit.
 */
export const prepareSlashSubmission = (display: string, tokens: ComposerToken[]) => ({
  command: expandPasteTokens(tokens)(display),
  display
})

export const shouldInterpolateSubmission = (display: string) => hasInterpolation(display)

export function useSubmission(opts: UseSubmissionOptions) {
  const { appendMessage, composerActions, composerRefs, composerState, gw, setLastUserMsg, slashRef, submitRef, sys } =
    opts

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
    (
      text: string,
      showUserMessage = true,
      displayText?: string,
      expandOverride?: (value: string) => string,
      submitOpts: { attachments?: Array<{ path: string; mime: string }>; skipDetectDrop?: boolean; destination?: SubmissionDestination; queueItem?: QueueItem; behindTurn?: boolean } = {}
    ) => {
      // Read tokens off the ref, not render state: a paste immediately followed
      // by Enter submits before React has re-rendered with the new token.
      const expand = expandOverride ?? expandTokens(composerRefs.tokensRef.current)

      const destination = submitOpts.destination ?? captureDestination()

      const item =
        submitOpts.queueItem ?? (destination.sid ? composerActions.stage?.(text, displayText, destination) : undefined)

      submitPrompt(
        text,
        {
          appendMessage,
          enqueue: composerActions.enqueue,
          expand,
          gw,
          setLastUserMsg,
          sys
        },
        showUserMessage,
        displayText,
        { ...submitOpts, destination, queueItem: item }
      )
    },
    [appendMessage, composerActions, composerRefs, gw, setLastUserMsg, sys]
  )

  const shellExec = useCallback(
    (cmd: string, destination = captureDestination(), item?: QueueItem) => {
      const focused = () => isCurrentDestination(destination)
      appendMessage({ role: 'user', text: `!${cmd}` })
      patchUiState({ busy: true, status: 'running…' })

      gw.request<ShellExecResponse>('shell.exec', { command: cmd, session_id: destination.sid })
        .then(raw => {
          const r = asRpcResult<ShellExecResponse>(raw)
          item?.settle?.(Boolean(r))

          if (!focused()) {
            return
          }

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
        .catch((e: Error) => {
          item?.settle?.(false)

          if (focused()) {
            sys(`error: ${e.message}`)
          }
        })
        .finally(() => {
          if (focused()) {
            patchUiState({ busy: false, status: 'ready' })
          }
        })
    },
    [appendMessage, gw, sys]
  )

  const interpolate = useCallback(
    (text: string, then: (result: string) => void, destination = captureDestination()) => {
      patchUiState({ status: 'interpolating…' })
      const matches = [...text.matchAll(new RegExp(INTERPOLATION_RE.source, 'g'))]

      Promise.all(
        matches.map(m =>
          gw
            .request<ShellExecResponse>('shell.exec', { command: m[1]!, session_id: destination.sid })
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
    (input: string | QueueItem) => {
      const item = typeof input === 'string' ? undefined : input
      const text = item?.preparedText ?? (typeof input === 'string' ? input : input.text)
      const destination = item?.destination ?? captureDestination()

      if (item?.preparedText !== undefined) {
        return send(text, true, item.display, value => value, { destination, queueItem: item, skipDetectDrop: true })
      }

      if (text.startsWith('!')) {
        return shellExec(text.slice(1).trim(), destination, item)
      }

      if (hasInterpolation(text)) {
        patchUiState({ busy: true })

        return interpolate(
          text,
          result => send(result, true, undefined, value => value, { destination, queueItem: item }),
          destination
        )
      }

      send(text, true, undefined, value => value, { destination, queueItem: item })
    },
    [interpolate, send, shellExec]
  )

  // Honors `display.busy_input_mode` from config.yaml (CLI parity):
  //   - 'queue'     (legacy): append to queueRef; drains on busy → false
  //   - 'steer'     : inject into the current turn via session.steer; falls
  //                   back to queue when steer is rejected (no agent / no
  //                   tool window).
  //   - 'interrupt' (default): submit immediately; the backend redirects the
  //                   active model request (or safely steers after a tool),
  //                   with legacy interrupt + queue as its compatibility path.
  //
  // `opts.fallbackToFront` re-inserts at the queue head (queue-edit picks keep
  // their position); the mainline submit path appends.
  const handleBusyInput = useCallback(
    (item: QueueItem, opts: { fallbackToFront?: boolean } = {}) => {
      const live = getUiState()
      const destination = captureDestination()
      const mode = live.busyInputMode

      const enqueueText = () => {
        if (opts.fallbackToFront) {
          composerActions.prependQueue(item, destination)
        } else {
          composerActions.enqueue(item.text, item.display, destination)
        }
      }

      if (gw.isCanonical && mode !== 'queue') {
        const staged = item.settle ? item : composerActions.stage?.(item.text, item.display, destination)

        if (!staged) { return }
        staged.controlMethod ??= mode === 'steer' ? 'session.steer' : 'session.redirect'
        staged.executionGeneration ??= live.info?.execution_generation
        staged.attachments ??= item.attachments

        return send(item.text, true, item.display, value => value, {
          destination, behindTurn: true, queueItem: staged, attachments: item.attachments })
      }

      if (mode === 'queue') {
        // Canonical authority: admit now so the input is crash-durable and in
        // every viewer's pending list; the server FIFO orders it behind the
        // running turn. Legacy gateways keep the renderer-side queue.
        if (gw.isCanonical) {
          return send(item.text, true, item.display, value => value, {
            destination, behindTurn: true, queueItem: item.settle ? item : undefined, attachments: item.attachments })
        }

        return enqueueText()
      }

      if (item.settle) {
        item.queued = false
      }

      // The gateway owns the atomic redirect decision because it knows whether
      // the agent is in model generation, tool execution, or an older runtime.
      // Reuse the normal submit pipeline so the correction gets its user bubble
      // and file-drop interpolation exactly once.
      send(item.text, true, item.display, value => value, { destination, queueItem: item.settle ? item : undefined, attachments: item.attachments })
    },
    [composerActions, gw, send]
  )

  const dispatchSubmission = useCallback(
    (input: string | QueueItem) => {
      if (typeof input !== 'string') {
        return sendQueued(input)
      }

      const full = input

      if (!full.trim()) {
        return
      }

      // History stores resolved content, not `[[…]]` labels: tokens are cleared
      // on submit, so recall must be self-contained. Image tokens resolve to
      // nothing — a detached image can't be re-attached by recalling the text.
      // Idempotent on token-free text, so re-submitting a recalled entry is
      // stable.
      const destination = captureDestination()
      const submissionTokens = [...composerRefs.tokensRef.current]
      const submission = prepareSubmission(full, submissionTokens)
      const toHistory = submission.text

      if (looksLikeSlashCommand(full)) {
        const slash = prepareSlashSubmission(full, submissionTokens)

        appendMessage({ kind: 'slash', role: 'system', text: slash.display })
        composerActions.pushHistory(toHistory)

        const parsed = parseSlashCommand(full)

        const queued =
          parsed.name === 'queue' || parsed.name === 'q' ? queueItemFromSlash(slash.display, slash.command) : undefined

        // Attachment commands capture the cleared composer's revision.
        composerActions.clearIn()

        if (queued) {
          const retained = composerActions.enqueue(queued.text, queued.display, destination)

          if (retained) { retained.attachments = submission.attachments; savePendingInput(retained) }
          sys(`queued: "${queued.display.slice(0, 50)}${queued.display.length > 50 ? '…' : ''}"`)
        } else {
          // Image tokens are labels in the command; the descriptors and their
          // expander ride along so a skill/alias send still carries the image.
          slashRef.current(slash.command, { attachments: submission.attachments, expand: expandTokens(submissionTokens) })
        }

        return
      }

      if (full.startsWith('!')) {
        composerActions.clearIn()

        return shellExec(full.slice(1).trim())
      }

      const live = getUiState()

      if (!live.sid) { return sys('session not ready — draft kept; reconnect or choose a session') }

      if (live.gatewayConnected === false) {
        composerActions.pushHistory(toHistory)
        const retained = composerActions.enqueue(submission.text, submission.display, destination)

        if (retained) { retained.attachments = submission.attachments; savePendingInput(retained) }
        composerActions.clearIn()

        return
      }

      const editIdx = composerRefs.queueEditRef.current
      composerActions.clearIn()

      if (editIdx !== null) {
        const picked = composerActions.takeQueue(editIdx, full)
        composerActions.setQueueEdit(null)

        if (!picked || !live.sid) {
          return
        }

        if (getUiState().busy) {
          // 'interrupt' / 'steer' should reach the live turn instead of
          // silently going back to the queue.  handleBusyInput resolves
          // mode-specific behavior (interrupt-and-send, steer, or queue).
          if (getUiState().busyInputMode === 'queue' && !gw.isCanonical) {
            return composerActions.prependQueue(picked)
          }

          return handleBusyInput(picked, { fallbackToFront: true })
        }

        return sendQueued(picked)
      }

      composerActions.pushHistory(toHistory)

      if (getUiState().busy) {
        return handleBusyInput({ ...queueItem(submission.text, submission.display), attachments: submission.attachments })
      }

      if (shouldInterpolateSubmission(full)) {
        patchUiState({ busy: true })

        const item = composerActions.stage?.(submission.text, submission.display, destination)

        return interpolate(
          full,
          text =>
            send(prepareSubmission(text, submissionTokens).text, true, text, value => value, {
              destination,
              queueItem: item, attachments: submission.attachments
            }),
          destination
        )
      }

      send(submission.text, true, submission.display, value => value, { attachments: submission.attachments })
    },
    [
      appendMessage,
      composerActions,
      composerRefs,
      handleBusyInput,
      interpolate,
      send,
      sendQueued,
      shellExec,
      slashRef,
      sys
    ]
  )

  const submit = useCallback(
    (value: string) => {
      if (composerState.completions.length) {
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
          const next = composerActions.dequeue(true)

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

  // Literal submission: route text straight to the prompt pipeline, skipping
  // slash-command routing, `!` shell dispatch, [[token]] expansion, and
  // $(...) interpolation. Startup `-q` queries use this — they're arbitrary
  // launcher/script text, and one-shot mode already treats them literally.
  const submitLiteral = useCallback(
    (value: string, attachments?: Array<{ path: string; mime: string }>) => {
      if (!value.trim()) {
        return
      }

      send(value, true, value, v => v, { skipDetectDrop: true, attachments })
    },
    [send]
  )

  return { dispatchSubmission, send, sendQueued, submit, submitLiteral }
}

export interface UseSubmissionOptions {
  appendMessage: (msg: Msg) => void
  composerActions: ComposerActions
  composerRefs: ComposerRefs
  composerState: ComposerState
  gw: GatewayClient
  setLastUserMsg: (value: string) => void
  slashRef: MutableRefObject<SlashHandler>
  submitRef: MutableRefObject<(value: string) => void>
  sys: (text: string) => void
}
