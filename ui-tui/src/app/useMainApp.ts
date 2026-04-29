import { type ScrollBoxHandle, useApp, useHasSelection, useSelection, useStdout, useTerminalTitle } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { STARTUP_RESUME_ID } from '../config/env.js'
import { FULL_RENDER_TAIL_ITEMS, MAX_HISTORY, WHEEL_SCROLL_STEP } from '../config/limits.js'
import { SECTION_NAMES, sectionMode } from '../domain/details.js'
import { attachedImageNotice, imageTokenMeta } from '../domain/messages.js'
import { fmtCwdBranch, shortCwd } from '../domain/paths.js'
import { type GatewayClient } from '../gatewayClient.js'
import type {
  ClarifyRespondResponse,
  ClipboardPasteResponse,
  GatewayEvent,
  TerminalResizeResponse
} from '../gatewayTypes.js'
import { useGitBranch } from '../hooks/useGitBranch.js'
import { useVirtualHistory } from '../hooks/useVirtualHistory.js'
import { appendTranscriptMessage } from '../lib/messages.js'
import { isMac } from '../lib/platform.js'
import { asRpcResult, rpcErrorMessage } from '../lib/rpc.js'
import { terminalParityHints } from '../lib/terminalParity.js'
import { buildToolTrailLine, sameToolTrailGroup, toolTrailLabel } from '../lib/text.js'
import { estimatedMsgHeight, messageHeightKey } from '../lib/virtualHeights.js'
import type { Msg, PanelSection, SlashCatalog } from '../types.js'

import { createGatewayEventHandler } from './createGatewayEventHandler.js'
import { createSlashHandler } from './createSlashHandler.js'
import { getInputSelection } from './inputSelectionStore.js'
import { type GatewayRpc, type TranscriptRow } from './interfaces.js'
import { $overlayState, patchOverlayState } from './overlayStore.js'
import { scrollWithSelectionBy } from './scroll.js'
import { turnController } from './turnController.js'
import { patchTurnState, useTurnSelector } from './turnStore.js'
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
const MAX_HEIGHT_CACHE_BUCKETS = 12

const capHistory = (items: Msg[]): Msg[] => {
  if (items.length <= MAX_HISTORY) {
    return items
  }

  return items[0]?.kind === 'intro' ? [items[0]!, ...items.slice(-(MAX_HISTORY - 1))] : items.slice(-MAX_HISTORY)
}

const statusColorOf = (status: string, t: { error: string; muted: string; ok: string; warn: string }) => {
  if (status === 'ready') {
    return t.ok
  }

  if (status.startsWith('error')) {
    return t.error
  }

  if (status === 'interrupted') {
    return t.warn
  }

  return t.muted
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
  const [turnStartedAt, setTurnStartedAt] = useState<null | number>(null)
  const [goodVibesTick, setGoodVibesTick] = useState(0)
  const [bellOnComplete, setBellOnComplete] = useState(false)

  const ui = useStore($uiState)
  const overlay = useStore($overlayState)

  const turnLiveTailActive = useTurnSelector(state =>
    Boolean(
      state.streaming ||
      state.streamPendingTools.length ||
      state.streamSegments.length ||
      state.reasoning.trim() ||
      state.reasoningActive ||
      state.tools.length ||
      state.subagents.length ||
      state.todos.length
    )
  )

  const slashFlightRef = useRef(0)
  const slashRef = useRef<(cmd: string) => boolean>(() => false)
  const colsRef = useRef(cols)
  const scrollRef = useRef<null | ScrollBoxHandle>(null)
  const onEventRef = useRef<(ev: GatewayEvent) => void>(() => {})
  const clipboardPasteRef = useRef<(quiet?: boolean) => Promise<void> | void>(() => {})
  const submitRef = useRef<(value: string) => void>(() => {})
  const terminalHintsShownRef = useRef(new Set<string>())
  const historyItemsRef = useRef(historyItems)
  const lastUserMsgRef = useRef(lastUserMsg)
  const msgIdsRef = useRef(new WeakMap<Msg, string>())
  const msgIdSeqRef = useRef(0)
  const heightCachesRef = useRef(new Map<string, Map<string, number>>())

  colsRef.current = cols
  historyItemsRef.current = historyItems
  lastUserMsgRef.current = lastUserMsg

  const hasSelection = useHasSelection()
  const selection = useSelection()
  const lastCopiedVersionRef = useRef(-1)

  useEffect(() => {
    selection.setSelectionBgColor(ui.theme.color.selectionBg)
  }, [selection, ui.theme.color.selectionBg])

  // macOS Terminal.app does not forward Cmd+C to fullscreen TUIs that enable
  // mouse tracking, so the only reliable native-feeling path is iTerm-style
  // copy-on-select: once a drag creates a stable TUI selection, write it to
  // the system clipboard while keeping the highlight visible.
  //
  // Subscribe directly via the ink selection bus (not useSyncExternalStore)
  // so React doesn't re-render MainApp on every drag-move tick. The version
  // ref de-dupes against re-entrant notifications.
  useEffect(() => {
    if (!isMac) {
      return
    }

    return selection.subscribe(() => {
      if (!selection.hasSelection()) {
        return
      }

      const state = selection.getState() as { isDragging?: boolean } | null

      if (state?.isDragging) {
        return
      }

      const version = selection.version()

      if (version === lastCopiedVersionRef.current) {
        return
      }

      lastCopiedVersionRef.current = version
      void selection.copySelectionNoClear()
    })
  }, [selection])

  const clearSelection = useCallback(() => {
    selection.clearSelection()
    getInputSelection()?.collapseToEnd()
  }, [selection])

  const composer = useComposerState({
    gw,
    onClipboardPaste: quiet => clipboardPasteRef.current(quiet),
    onImageAttached: info => {
      sys(attachedImageNotice(info))
    },
    submitRef
  })

  const { actions: composerActions, refs: composerRefs, state: composerState } = composer
  const empty = !historyItems.some(msg => msg.kind !== 'intro')

  useEffect(() => {
    void terminalParityHints()
      .then(hints => {
        for (const hint of hints) {
          if (terminalHintsShownRef.current.has(hint.key)) {
            continue
          }

          terminalHintsShownRef.current.add(hint.key)
          turnController.pushActivity(hint.message, hint.tone)
        }
      })
      .catch(() => {})
  }, [])

  const messageId = useCallback((msg: Msg) => {
    const hit = msgIdsRef.current.get(msg)

    if (hit) {
      return hit
    }

    const next = `${messageHeightKey(msg)}:${++msgIdSeqRef.current}`

    msgIdsRef.current.set(msg, next)

    return next
  }, [])

  const virtualRows = useMemo<TranscriptRow[]>(
    () => historyItems.map((msg, index) => ({ index, key: messageId(msg), msg })),
    [historyItems, messageId]
  )

  const detailsLayoutKey = useMemo(() => {
    const thinking = sectionMode('thinking', ui.detailsMode, ui.sections, ui.detailsModeCommandOverride)
    const tools = sectionMode('tools', ui.detailsMode, ui.sections, ui.detailsModeCommandOverride)

    return `${thinking}:${tools}`
  }, [ui.detailsMode, ui.detailsModeCommandOverride, ui.sections])

  const detailsVisible = detailsLayoutKey !== 'hidden:hidden'
  const heightCacheKey = `${ui.sid ?? 'draft'}:${cols}:${ui.compact ? '1' : '0'}:${detailsLayoutKey}`

  const heightCache = useMemo(() => {
    let cache = heightCachesRef.current.get(heightCacheKey)

    if (!cache) {
      cache = new Map()
      heightCachesRef.current.set(heightCacheKey, cache)

      if (heightCachesRef.current.size > MAX_HEIGHT_CACHE_BUCKETS) {
        heightCachesRef.current.delete(heightCachesRef.current.keys().next().value!)
      }
    }

    return cache
  }, [heightCacheKey])

  const estimateRowHeight = useCallback(
    (index: number) =>
      estimatedMsgHeight(virtualRows[index]!.msg, cols, {
        compact: ui.compact,
        details: detailsVisible,
        limitHistory: index < virtualRows.length - FULL_RENDER_TAIL_ITEMS
      }),
    [cols, detailsVisible, ui.compact, virtualRows]
  )

  const syncHeightCache = useCallback(
    (heights: ReadonlyMap<string, number>) => {
      for (const row of virtualRows) {
        const h = heights.get(row.key)

        if (h) {
          heightCache.set(row.key, h)
        }
      }
    },
    [heightCache, virtualRows]
  )

  const virtualHistory = useVirtualHistory(scrollRef, virtualRows, cols, {
    estimateHeight: estimateRowHeight,
    initialHeights: heightCache,
    liveTailActive: turnLiveTailActive,
    onHeightsChange: syncHeightCache
  })

  const scrollWithSelection = useCallback(
    (delta: number) => scrollWithSelectionBy(delta, { scrollRef, selection }),
    [selection]
  )

  const appendMessage = useCallback(
    (msg: Msg) => setHistoryItems(prev => capHistory(appendTranscriptMessage(prev, msg))),
    []
  )

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

  useEffect(() => {
    if (ui.busy) {
      setTurnStartedAt(prev => prev ?? Date.now())
    } else {
      setTurnStartedAt(null)
    }
  }, [ui.busy])

  useConfigSync({ gw, setBellOnComplete, setVoiceEnabled, sid: ui.sid })

  // Tab title: `⚠` waiting on approval/sudo/secret/clarify, `⏳` busy, `✓` idle.
  const model = ui.info?.model?.replace(/^.*\//, '') ?? ''

  const marker = overlay.approval || overlay.sudo || overlay.secret || overlay.clarify ? '⚠' : ui.busy ? '⏳' : '✓'

  const tabCwd = ui.info?.cwd

  useTerminalTitle(model ? `${marker} ${model}${tabCwd ? ` · ${shortCwd(tabCwd, 24)}` : ''}` : 'Hermes')

  useEffect(() => {
    if (!ui.sid || !stdout) {
      return
    }

    let timer: ReturnType<typeof setTimeout> | undefined

    const onResize = () => {
      clearTimeout(timer)
      timer = setTimeout(() => {
        timer = undefined
        void rpc<TerminalResizeResponse>('terminal.resize', { cols: stdout.columns ?? 80, session_id: ui.sid })
      }, 100)
    }

    stdout.on('resize', onResize)

    return () => {
      clearTimeout(timer)
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

  const { dispatchSubmission, send, sendQueued, submit } = useSubmission({
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

  // Drain one queued message whenever the session settles (busy → false):
  // agent turn ends, interrupt, shell.exec finishes, error recovered, or the
  // session first comes up with pre-queued messages. Without this, shell.exec
  // and error paths never emit message.complete, so anything enqueued while
  // `!sleep` / a failed turn was running would stay stuck forever.
  useEffect(() => {
    if (
      !ui.sid ||
      ui.busy ||
      composerRefs.queueEditRef.current !== null ||
      composerRefs.queueRef.current.length === 0
    ) {
      return
    }

    const next = composerActions.dequeue()

    if (next) {
      patchUiState({ busy: true, status: 'running…' })
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
    voice: {
      enabled: voiceEnabled,
      recording: voiceRecording,
      setProcessing: setVoiceProcessing,
      setRecording: setVoiceRecording,
      setVoiceEnabled
    },
    wheelStep: WHEEL_SCROLL_STEP
  })

  const onEvent = useMemo(
    () =>
      createGatewayEventHandler({
        composer: { setInput: composerActions.setInput },
        gateway,
        session: {
          STARTUP_RESUME_ID,
          colsRef,
          newSession: session.newSession,
          resetSession: session.resetSession,
          resumeById: session.resumeById,
          setCatalog
        },
        submission: { submitRef },
        system: { bellOnComplete, stdout, sys },
        transcript: { appendMessage, panel, setHistoryItems },
        voice: {
          setProcessing: setVoiceProcessing,
          setRecording: setVoiceRecording,
          setVoiceEnabled
        }
      }),
    [
      appendMessage,
      bellOnComplete,
      clearSelection,
      composerActions.setInput,
      gateway,
      panel,
      session.newSession,
      session.resetSession,
      session.resumeById,
      setVoiceEnabled,
      setVoiceProcessing,
      setVoiceRecording,
      stdout,
      submitRef,
      sys
    ]
  )

  onEventRef.current = onEvent

  useEffect(() => {
    const handler = (ev: GatewayEvent) => onEventRef.current(ev)

    const exitHandler = () => {
      turnController.reset()
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

  useLongRunToolCharms()

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

  const hasReasoning = useTurnSelector(state => Boolean(state.reasoning.trim()))

  // Per-section overrides win over the global mode — when every section is
  // resolved to hidden, the only thing ToolTrail will surface is the
  // floating-alert backstop (errors/warnings).  Mirror that so we don't
  // render an empty wrapper Box above the streaming area in quiet mode.
  const anyPanelVisible = SECTION_NAMES.some(
    s => sectionMode(s, ui.detailsMode, ui.sections, ui.detailsModeCommandOverride) !== 'hidden'
  )
  const thinkingPanelVisible = sectionMode('thinking', ui.detailsMode, ui.sections, ui.detailsModeCommandOverride) !== 'hidden'

  const showProgressArea = useTurnSelector(state =>
    anyPanelVisible
      ? Boolean(
          ui.busy ||
          state.outcome ||
          state.streamPendingTools.length ||
          state.streamSegments.length ||
          state.subagents.length ||
          state.tools.length ||
          state.todos.length ||
          state.turnTrail.length ||
          (thinkingPanelVisible && hasReasoning) ||
          state.activity.length
        )
      : state.activity.some(item => item.tone !== 'info')
  )

  const appActions = useMemo(
    () => ({
      answerApproval,
      answerClarify,
      answerSecret,
      answerSudo,
      clearSelection,
      onModelSelect,
      resumeById: session.resumeById,
      setStickyPrompt
    }),
    [answerApproval, answerClarify, answerSecret, answerSudo, clearSelection, onModelSelect, session.resumeById]
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

  // Pass current progress through unfrozen — streaming update throttling
  // handles interaction load; progress must stay truthful so panels don't
  // randomly disappear when the live tail scrolls offscreen.
  const appProgress = useMemo(() => ({ showProgressArea }), [showProgressArea])

  const cwd = ui.info?.cwd || process.env.HERMES_CWD || process.cwd()
  const gitBranch = useGitBranch(cwd)

  const appStatus = useMemo(
    () => ({
      cwdLabel: fmtCwdBranch(cwd, gitBranch),
      goodVibesTick,
      sessionStartedAt: ui.sid ? sessionStartedAt : null,
      showStickyPrompt: !!stickyPrompt,
      statusColor: statusColorOf(ui.status, ui.theme.color),
      stickyPrompt,
      turnStartedAt: ui.sid ? turnStartedAt : null,
      // CLI parity: the classic prompt_toolkit status bar shows a red dot
      // on REC (cli.py:_get_voice_status_fragments line 2344).
      voiceLabel: voiceRecording ? '● REC' : voiceProcessing ? '◉ STT' : `voice ${voiceEnabled ? 'on' : 'off'}`
    }),
    [
      cwd,
      gitBranch,
      goodVibesTick,
      sessionStartedAt,
      stickyPrompt,
      turnStartedAt,
      ui,
      voiceEnabled,
      voiceProcessing,
      voiceRecording
    ]
  )

  const appTranscript = useMemo(
    () => ({ historyItems, scrollRef, virtualHistory, virtualRows }),
    [historyItems, virtualHistory, virtualRows]
  )

  return { appActions, appComposer, appProgress, appStatus, appTranscript, gateway }
}
