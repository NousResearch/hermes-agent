import { type ScrollBoxHandle, useApp, useHasSelection, useSelection, useStdout } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { STARTUP_RESUME_ID } from '../config/env.js'
import { MAX_HISTORY, WHEEL_SCROLL_STEP } from '../config/limits.js'
import { imageTokenMeta } from '../domain/messages.js'
import { shortCwd } from '../domain/paths.js'
import { type GatewayClient } from '../gatewayClient.js'
import type {
  ClarifyRespondResponse,
  ClipboardPasteResponse,
  GatewayEvent,
  TerminalResizeResponse
} from '../gatewayTypes.js'
import { useVirtualHistory } from '../hooks/useVirtualHistory.js'
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'
import { buildToolTrailLine, sameToolTrailGroup, toolTrailLabel } from '../lib/text.js'
import type { Msg, PanelSection, SlashCatalog } from '../types.js'

import { createGatewayEventHandler } from './createGatewayEventHandler.js'
import { createSlashHandler } from './createSlashHandler.js'
import { type GatewayRpc, type TranscriptRow } from './interfaces.js'
import { $overlayState, patchOverlayState } from './overlayStore.js'
import { turnController } from './turnController.js'
import { $turnState, patchTurnState } from './turnStore.js'
import { $uiState, getUiState, patchUiState } from './uiStore.js'
import { useComposerState } from './useComposerState.js'
import { useConfigSync } from './useConfigSync.js'
import { useInputHandlers } from './useInputHandlers.js'
import { useLongRunToolCharms } from './useLongRunToolCharms.js'
import { useSessionLifecycle } from './useSessionLifecycle.js'
import { useSubmission } from './useSubmission.js'

const GOOD_VIBES_RE = /\b(good bot|thanks|thank you|thx|ty|ily|love you)\b/i
const BRACKET_PASTE_ON = '\x1b[?2004h'
const BRACKET_PASTE_OFF = '\x1b[?2004l'

const capHistory = (items: Msg[]): Msg[] => {
  if (items.length <= MAX_HISTORY) {
    return items
  }

  return items[0]?.kind === 'intro' ? [items[0]!, ...items.slice(-(MAX_HISTORY - 1))] : items.slice(-MAX_HISTORY)
}

const statusColorOf = (status: string, t: { dim: string; error: string; ok: string; warn: string }) => {
  if (status === 'ready') {
    return t.ok
  }

  if (status.startsWith('error')) {
    return t.error
  }

  if (status === 'interrupted') {
    return t.warn
  }

  return t.dim
}

interface SelectionSnap {
  anchor?: { row: number }
  focus?: { row: number }
  isDragging?: boolean
}

export function useMainApp(gw: GatewayClient) {
  const { exit } = useApp()
  const { stdout } = useStdout()
  const [cols, setCols] = useState(stdout?.columns ?? 80)

  useEffect(() => {
    if (!stdout) {
      return
    }

    const sync = () => setCols(stdout.columns ?? 80)

    stdout.on('resize', sync)

    if (stdout.isTTY) {
      stdout.write(BRACKET_PASTE_ON)
    }

    return () => {
      stdout.off('resize', sync)

      if (stdout.isTTY) {
        stdout.write(BRACKET_PASTE_OFF)
      }
    }
  }, [stdout])

  const [historyItems, setHistoryItems] = useState<Msg[]>(() => [{ kind: 'intro', role: 'system', text: '' }])
  const [lastUserMsg, setLastUserMsg] = useState('')
  const [stickyPrompt, setStickyPrompt] = useState('')
  const [catalog, setCatalog] = useState<null | SlashCatalog>(null)
  const [voiceEnabled, setVoiceEnabled] = useState(false)
  const [voiceRecording, setVoiceRecording] = useState(false)
  const [voiceProcessing, setVoiceProcessing] = useState(false)
  const [sessionStartedAt, setSessionStartedAt] = useState(() => Date.now())
  const [goodVibesTick, setGoodVibesTick] = useState(0)
  const [bellOnComplete, setBellOnComplete] = useState(false)

  const ui = useStore($uiState)
  const overlay = useStore($overlayState)
  const turn = useStore($turnState)

  const slashFlightRef = useRef(0)
  const slashRef = useRef<(cmd: string) => boolean>(() => false)
  const colsRef = useRef(cols)
  const scrollRef = useRef<null | ScrollBoxHandle>(null)
  const onEventRef = useRef<(ev: GatewayEvent) => void>(() => {})
  const clipboardPasteRef = useRef<(quiet?: boolean) => Promise<void> | void>(() => {})
  const submitRef = useRef<(value: string) => void>(() => {})
  const historyItemsRef = useRef(historyItems)
  const lastUserMsgRef = useRef(lastUserMsg)
  const msgIdsRef = useRef(new WeakMap<Msg, string>())
  const nextMsgIdRef = useRef(0)

  colsRef.current = cols
  historyItemsRef.current = historyItems
  lastUserMsgRef.current = lastUserMsg

  const hasSelection = useHasSelection()
  const selection = useSelection()

  useEffect(() => {
    selection.setSelectionBgColor(ui.theme.color.selectionBg)
  }, [selection, ui.theme.color.selectionBg])

  const composer = useComposerState({
    gw,
    onClipboardPaste: quiet => clipboardPasteRef.current(quiet),
    submitRef
  })

  const { actions: composerActions, refs: composerRefs, state: composerState } = composer
  const empty = !historyItems.some(msg => msg.kind !== 'intro')

  const messageId = useCallback((msg: Msg) => {
    const hit = msgIdsRef.current.get(msg)

    if (hit) {
      return hit
    }

    const next = `m${++nextMsgIdRef.current}`

    msgIdsRef.current.set(msg, next)

    return next
  }, [])

  const virtualRows = useMemo<TranscriptRow[]>(
    () => historyItems.map((msg, index) => ({ index, key: messageId(msg), msg })),
    [historyItems, messageId]
  )

  const virtualHistory = useVirtualHistory(scrollRef, virtualRows)

  const scrollWithSelection = useCallback(
    (delta: number) => {
      const s = scrollRef.current

      if (!s) {
        return
      }

      const sel = selection.getState() as null | SelectionSnap
      const top = s.getViewportTop()
      const bottom = top + s.getViewportHeight() - 1

      if (
        !sel?.anchor ||
        !sel.focus ||
        sel.anchor.row < top ||
        sel.anchor.row > bottom ||
        (!sel.isDragging && (sel.focus.row < top || sel.focus.row > bottom))
      ) {
        return s.scrollBy(delta)
      }

      const max = Math.max(0, s.getScrollHeight() - s.getViewportHeight())
      const cur = s.getScrollTop() + s.getPendingDelta()
      const actual = Math.max(0, Math.min(max, cur + delta)) - cur

      if (actual === 0) {
        return
      }

      const shift = sel!.isDragging ? selection.shiftAnchor : selection.shiftSelection

      if (actual > 0) {
        selection.captureScrolledRows(top, top + actual - 1, 'above')
      } else {
        selection.captureScrolledRows(bottom + actual + 1, bottom, 'below')
      }

      shift(-actual, top, bottom)
      s.scrollBy(delta)
    },
    [selection]
  )

  const appendMessage = useCallback((msg: Msg) => setHistoryItems(prev => capHistory([...prev, msg])), [])

  const sys = useCallback((text: string) => appendMessage({ role: 'system', text }), [appendMessage])

  const page = useCallback(
    (text: string, title?: string) => patchOverlayState({ pager: { lines: text.split('\n'), offset: 0, title } }),
    []
  )

  const panel = useCallback(
    (title: string, sections: PanelSection[]) =>
      appendMessage({ kind: 'panel', panelData: { sections, title }, role: 'system', text: '' }),
    [appendMessage]
  )

  const maybeWarn = useCallback(
    (value: unknown) => {
      const warning = (value as { warning?: unknown } | null)?.warning

      if (typeof warning === 'string' && warning) {
        sys(`warning: ${warning}`)
      }
    },
    [sys]
  )

  const maybeGoodVibes = useCallback((text: string) => {
    if (GOOD_VIBES_RE.test(text)) {
      setGoodVibesTick(v => v + 1)
    }
  }, [])

  const rpc: GatewayRpc = useCallback(
    async <T extends Record<string, any> = Record<string, any>>(
      method: string,
      params: Record<string, unknown> = {}
    ) => {
      try {
        const result = asRpcResult<T>(await gw.request<T>(method, params))

        if (result) {
          return result
        }

        sys(`error: invalid response: ${method}`)
      } catch (e) {
        sys(`error: ${rpcErrorMessage(e)}`)
      }

      return null
    },
    [gw, sys]
  )

  const gateway = useMemo(() => ({ gw, rpc }), [gw, rpc])

  const die = useCallback(() => {
    gw.kill()
    exit()
  }, [exit, gw])

  const session = useSessionLifecycle({
    colsRef,
    composerActions,
    gw,
    panel,
    rpc,
    scrollRef,
    setHistoryItems,
    setLastUserMsg,
    setSessionStartedAt,
    setStickyPrompt,
    setVoiceProcessing,
    setVoiceRecording,
    sys
  })

  useConfigSync({ gw, setBellOnComplete, setVoiceEnabled, sid: ui.sid })

  useEffect(() => {
    if (!ui.sid || !stdout) {
      return
    }

    const onResize = () =>
      rpc<TerminalResizeResponse>('terminal.resize', { cols: stdout.columns ?? 80, session_id: ui.sid })

    stdout.on('resize', onResize)

    return () => {
      stdout.off('resize', onResize)
    }
  }, [rpc, stdout, ui.sid])

  const answerClarify = useCallback(
    (answer: string) => {
      const clarify = overlay.clarify

      if (!clarify) {
        return
      }

      const label = toolTrailLabel('clarify')

      turnController.turnTools = turnController.turnTools.filter(line => !sameToolTrailGroup(label, line))
      patchTurnState({ turnTrail: turnController.turnTools })

      rpc<ClarifyRespondResponse>('clarify.respond', { answer, request_id: clarify.requestId }).then(r => {
        if (!r) {
          return
        }

        if (answer) {
          turnController.persistedToolLabels.add(label)
          appendMessage({
            kind: 'trail',
            role: 'system',
            text: '',
            tools: [buildToolTrailLine('clarify', clarify.question)]
          })
          appendMessage({ role: 'user', text: answer })
          patchUiState({ status: 'running…' })
        } else {
          sys('prompt cancelled')
        }

        patchOverlayState({ clarify: null })
      })
    },
    [appendMessage, overlay.clarify, rpc, sys]
  )

  const paste = useCallback(
    (quiet = false) =>
      rpc<ClipboardPasteResponse>('clipboard.paste', { session_id: getUiState().sid }).then(r => {
        if (!r) {
          return
        }

        if (r.attached) {
          const meta = imageTokenMeta(r)

          return sys(`📎 Image #${r.count} attached from clipboard${meta ? ` · ${meta}` : ''}`)
        }

        if (!quiet) {
          sys(r.message || 'No image found in clipboard')
        }
      }),
    [rpc, sys]
  )

  clipboardPasteRef.current = paste

  const { dispatchSubmission, send, sendQueued, shellExec, submit } = useSubmission({
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
  })

  const prevSidRef = useRef<null | string>(null)
  useEffect(() => {
    const prev = prevSidRef.current
    prevSidRef.current = ui.sid

    if (prev !== null || !ui.sid || ui.busy || composerRefs.queueEditRef.current !== null) {
      return
    }

    const next = composerActions.dequeue()

    if (next) {
      sendQueued(next)
    }
  }, [ui.sid, ui.busy, composerActions, composerRefs, sendQueued])

  const { pagerPageSize } = useInputHandlers({
    actions: {
      answerClarify,
      appendMessage,
      die,
      dispatchSubmission,
      guardBusySessionSwitch: session.guardBusySessionSwitch,
      newSession: session.newSession,
      sys
    },
    composer: { actions: composerActions, refs: composerRefs, state: composerState },
    gateway,
    terminal: { hasSelection, scrollRef, scrollWithSelection, selection, stdout },
    voice: { recording: voiceRecording, setProcessing: setVoiceProcessing, setRecording: setVoiceRecording },
    wheelStep: WHEEL_SCROLL_STEP
  })

  const onEvent = useMemo(
    () =>
      createGatewayEventHandler({
        composer: { dequeue: composerActions.dequeue, queueEditRef: composerRefs.queueEditRef, sendQueued },
        gateway,
        session: {
          STARTUP_RESUME_ID,
          colsRef,
          newSession: session.newSession,
          resetSession: session.resetSession,
          resumeById: session.resumeById,
          setCatalog
        },
        system: { bellOnComplete, stdout, sys },
        transcript: { appendMessage, panel, setHistoryItems }
      }),
    [
      appendMessage,
      bellOnComplete,
      composerActions,
      composerRefs,
      gateway,
      panel,
      sendQueued,
      session.newSession,
      session.resetSession,
      session.resumeById,
      stdout,
      sys
    ]
  )

  onEventRef.current = onEvent

  useEffect(() => {
    const handler = (ev: GatewayEvent) => onEventRef.current(ev)

    const exitHandler = () => {
      patchUiState({ busy: false, sid: null, status: 'gateway exited' })
      turnController.pushActivity('gateway exited · /logs to inspect', 'error')
      sys('error: gateway exited')
    }

    gw.on('event', handler)
    gw.on('exit', exitHandler)
    gw.drain()

    return () => {
      gw.off('event', handler)
      gw.off('exit', exitHandler)
      gw.kill()
    }
  }, [gw, sys])

  useLongRunToolCharms(ui.busy, turn.tools)

  const slash = useMemo(
    () =>
      createSlashHandler({
        composer: {
          enqueue: composerActions.enqueue,
          hasSelection,
          paste,
          queueRef: composerRefs.queueRef,
          selection,
          setInput: composerActions.setInput
        },
        gateway,
        local: {
          catalog,
          getHistoryItems: () => historyItemsRef.current,
          getLastUserMsg: () => lastUserMsgRef.current,
          maybeWarn
        },
        session: {
          closeSession: session.closeSession,
          die,
          guardBusySessionSwitch: session.guardBusySessionSwitch,
          newSession: session.newSession,
          resetVisibleHistory: session.resetVisibleHistory,
          resumeById: session.resumeById,
          setSessionStartedAt
        },
        slashFlightRef,
        transcript: { page, panel, send, setHistoryItems, sys, trimLastExchange: session.trimLastExchange },
        voice: { setVoiceEnabled }
      }),
    [
      catalog,
      composerActions,
      composerRefs,
      die,
      gateway,
      hasSelection,
      maybeWarn,
      page,
      panel,
      paste,
      selection,
      send,
      session,
      sys
    ]
  )

  slashRef.current = slash

  const respondWith = useCallback(
    (method: string, params: Record<string, unknown>, done: () => void) => rpc(method, params).then(r => r && done()),
    [rpc]
  )

  const answerApproval = useCallback(
    (choice: string) =>
      respondWith('approval.respond', { choice, session_id: ui.sid }, () => {
        patchOverlayState({ approval: null })
        patchTurnState({ outcome: choice === 'deny' ? 'denied' : `approved (${choice})` })
        patchUiState({ status: 'running…' })
      }),
    [respondWith, ui.sid]
  )

  const answerSudo = useCallback(
    (pw: string) => {
      if (!overlay.sudo) {
        return
      }

      return respondWith('sudo.respond', { password: pw, request_id: overlay.sudo.requestId }, () => {
        patchOverlayState({ sudo: null })
        patchUiState({ status: 'running…' })
      })
    },
    [overlay.sudo, respondWith]
  )

  const answerSecret = useCallback(
    (value: string) => {
      if (!overlay.secret) {
        return
      }

      return respondWith('secret.respond', { request_id: overlay.secret.requestId, value }, () => {
        patchOverlayState({ secret: null })
        patchUiState({ status: 'running…' })
      })
    },
    [overlay.secret, respondWith]
  )

  const onModelSelect = useCallback((value: string) => {
    patchOverlayState({ modelPicker: false })
    slashRef.current(`/model ${value}`)
  }, [])

  const hasReasoning = Boolean(turn.reasoning.trim())

  const showProgressArea =
    ui.detailsMode === 'hidden'
      ? turn.activity.some(item => item.tone !== 'info')
      : Boolean(
          ui.busy ||
          turn.outcome ||
          turn.streamPendingTools.length ||
          turn.streamSegments.length ||
          turn.subagents.length ||
          turn.tools.length ||
          turn.turnTrail.length ||
          hasReasoning ||
          turn.activity.length
        )

  const appActions = useMemo(
    () => ({
      answerApproval,
      answerClarify,
      answerSecret,
      answerSudo,
      onModelSelect,
      resumeById: session.resumeById,
      setStickyPrompt
    }),
    [answerApproval, answerClarify, answerSecret, answerSudo, onModelSelect, session.resumeById]
  )

  const appComposer = useMemo(
    () => ({
      cols,
      compIdx: composerState.compIdx,
      completions: composerState.completions,
      empty,
      handleTextPaste: composerActions.handleTextPaste,
      input: composerState.input,
      inputBuf: composerState.inputBuf,
      pagerPageSize,
      queueEditIdx: composerState.queueEditIdx,
      queuedDisplay: composerState.queuedDisplay,
      submit,
      updateInput: composerActions.setInput
    }),
    [cols, composerActions, composerState, empty, pagerPageSize, submit]
  )

  const appProgress = useMemo(
    () => ({ ...turn, showProgressArea, showStreamingArea: Boolean(turn.streaming) }),
    [turn, showProgressArea]
  )

  const appStatus = useMemo(
    () => ({
      cwdLabel: shortCwd(ui.info?.cwd || process.env.HERMES_CWD || process.cwd()),
      goodVibesTick,
      sessionStartedAt: ui.sid ? sessionStartedAt : null,
      showStickyPrompt: !!stickyPrompt,
      statusColor: statusColorOf(ui.status, ui.theme.color),
      stickyPrompt,
      voiceLabel: voiceRecording ? 'REC' : voiceProcessing ? 'STT' : `voice ${voiceEnabled ? 'on' : 'off'}`
    }),
    [goodVibesTick, sessionStartedAt, stickyPrompt, ui, voiceEnabled, voiceProcessing, voiceRecording]
  )

  const appTranscript = useMemo(
    () => ({ historyItems, scrollRef, virtualHistory, virtualRows }),
    [historyItems, virtualHistory, virtualRows]
  )

  return { appActions, appComposer, appProgress, appStatus, appTranscript, gateway }
}
