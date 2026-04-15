import { type ScrollBoxHandle, useApp, useHasSelection, useSelection, useStdout } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { MAX_HISTORY, MOUSE_TRACKING, PASTE_SNIPPET_RE, STARTUP_RESUME_ID, WHEEL_SCROLL_STEP } from './app/constants.js'
import { createGatewayEventHandler } from './app/createGatewayEventHandler.js'
import { createSlashHandler } from './app/createSlashHandler.js'
import { GatewayProvider } from './app/gatewayContext.js'
import {
  imageTokenMeta,
  introMsg,
  looksLikeSlashCommand,
  resolveDetailsMode,
  shortCwd,
  toTranscriptMessages
} from './app/helpers.js'
import { type GatewayRpc, type TranscriptRow } from './app/interfaces.js'
import { $isBlocked, $overlayState, patchOverlayState } from './app/overlayStore.js'
import { $uiState, getUiState, patchUiState } from './app/uiStore.js'
import { useComposerState } from './app/useComposerState.js'
import { useInputHandlers } from './app/useInputHandlers.js'
import { useTurnState } from './app/useTurnState.js'
import { AppLayout } from './components/appLayout.js'
import { INTERPOLATION_RE, ZERO } from './constants.js'
import { type GatewayClient } from './gatewayClient.js'
import type { ConfigFullResponse, ConfigMtimeResponse, GatewayEvent, SessionCreateResponse } from './gatewayTypes.js'
import { useVirtualHistory } from './hooks/useVirtualHistory.js'
import { asRpcResult, rpcErrorMessage } from './lib/rpc.js'
import { buildToolTrailLine, hasInterpolation, sameToolTrailGroup, toolTrailLabel } from './lib/text.js'
import type { Msg, PanelSection, SessionInfo, SlashCatalog } from './types.js'

// ── App ──────────────────────────────────────────────────────────────

export function App({ gw }: { gw: GatewayClient }) {
  const { exit } = useApp()
  const { stdout } = useStdout()
  const [cols, setCols] = useState(stdout?.columns ?? 80)

  useEffect(() => {
    if (!stdout) {
      return
    }

    const sync = () => setCols(stdout.columns ?? 80)
    stdout.on('resize', sync)

    // Enable bracketed paste so image-only clipboard paste reaches the app
    if (stdout.isTTY) {
      stdout.write('\x1b[?2004h')
    }

    return () => {
      stdout.off('resize', sync)

      if (stdout.isTTY) {
        stdout.write('\x1b[?2004l')
      }
    }
  }, [stdout])

  // ── State ────────────────────────────────────────────────────────

  const [historyItems, setHistoryItems] = useState<Msg[]>([])
  const [lastUserMsg, setLastUserMsg] = useState('')
  const [stickyPrompt, setStickyPrompt] = useState('')
  const [catalog, setCatalog] = useState<SlashCatalog | null>(null)
  const [voiceEnabled, setVoiceEnabled] = useState(false)
  const [voiceRecording, setVoiceRecording] = useState(false)
  const [voiceProcessing, setVoiceProcessing] = useState(false)
  const [sessionStartedAt, setSessionStartedAt] = useState(() => Date.now())
  const [bellOnComplete, setBellOnComplete] = useState(false)
  const ui = useStore($uiState)
  const overlay = useStore($overlayState)
  const isBlocked = useStore($isBlocked)

  // ── Refs ─────────────────────────────────────────────────────────

  const slashRef = useRef<(cmd: string) => boolean>(() => false)
  const lastEmptyAt = useRef(0)
  const colsRef = useRef(cols)
  const scrollRef = useRef<ScrollBoxHandle | null>(null)
  const onEventRef = useRef<(ev: GatewayEvent) => void>(() => {})
  const clipboardPasteRef = useRef<(quiet?: boolean) => Promise<void> | void>(() => {})
  const submitRef = useRef<(value: string) => void>(() => {})
  const configMtimeRef = useRef(0)
  const historyItemsRef = useRef(historyItems)
  const lastUserMsgRef = useRef(lastUserMsg)
  const msgIdsRef = useRef(new WeakMap<Msg, string>())
  const nextMsgIdRef = useRef(0)
  colsRef.current = cols
  historyItemsRef.current = historyItems
  lastUserMsgRef.current = lastUserMsg

  // ── Hooks ────────────────────────────────────────────────────────

  const hasSelection = useHasSelection()
  const selection = useSelection()
  const turn = useTurnState()
  const turnActions = turn.actions
  const turnRefs = turn.refs
  const turnState = turn.state

  const composer = useComposerState({
    gw,
    onClipboardPaste: quiet => clipboardPasteRef.current(quiet),
    submitRef
  })

  const composerActions = composer.actions
  const composerRefs = composer.refs
  const composerState = composer.state

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
    () =>
      historyItems.map((msg, index) => ({
        index,
        key: messageId(msg),
        msg
      })),
    [historyItems, messageId]
  )

  const virtualHistory = useVirtualHistory(scrollRef, virtualRows)

  const scrollWithSelection = useCallback(
    (delta: number) => {
      const s = scrollRef.current

      const sel = selection.getState() as {
        anchor?: { row: number }
        focus?: { row: number }
        isDragging?: boolean
      } | null

      if (!s || !sel?.anchor || !sel.focus) {
        s?.scrollBy(delta)

        return
      }

      const top = s.getViewportTop()
      const bottom = top + s.getViewportHeight() - 1

      if (sel.anchor.row < top || sel.anchor.row > bottom) {
        s.scrollBy(delta)

        return
      }

      if (!sel.isDragging && (sel.focus.row < top || sel.focus.row > bottom)) {
        s.scrollBy(delta)

        return
      }

      const max = Math.max(0, s.getScrollHeight() - s.getViewportHeight())
      const cur = s.getScrollTop() + s.getPendingDelta()
      const actual = Math.max(0, Math.min(max, cur + delta)) - cur

      if (actual === 0) {
        return
      }

      if (actual > 0) {
        selection.captureScrolledRows(top, top + actual - 1, 'above')
        sel.isDragging ? selection.shiftAnchor(-actual, top, bottom) : selection.shiftSelection(-actual, top, bottom)
      } else {
        const amount = -actual
        selection.captureScrolledRows(bottom - amount + 1, bottom, 'below')
        sel.isDragging ? selection.shiftAnchor(amount, top, bottom) : selection.shiftSelection(amount, top, bottom)
      }

      s.scrollBy(delta)
    },
    [selection]
  )

  // ── Core actions ─────────────────────────────────────────────────

  const appendMessage = useCallback((msg: Msg) => {
    const cap = (items: Msg[]) =>
      items.length <= MAX_HISTORY
        ? items
        : items[0]?.kind === 'intro'
          ? [items[0]!, ...items.slice(-(MAX_HISTORY - 1))]
          : items.slice(-MAX_HISTORY)

    setHistoryItems(prev => cap([...prev, msg]))
  }, [])

  const sys = useCallback((text: string) => appendMessage({ role: 'system' as const, text }), [appendMessage])

  const page = useCallback((text: string, title?: string) => {
    const lines = text.split('\n')
    patchOverlayState({ pager: { lines, offset: 0, title } })
  }, [])

  const panel = useCallback(
    (title: string, sections: PanelSection[]) => {
      appendMessage({ role: 'system', text: '', kind: 'panel', panelData: { title, sections } })
    },
    [appendMessage]
  )

  const maybeWarn = useCallback(
    (value: any) => {
      if (value?.warning) {
        sys(`warning: ${value.warning}`)
      }
    },
    [sys]
  )

  const applyDisplayConfig = useCallback((cfg: ConfigFullResponse | null) => {
    const display = cfg?.config?.display ?? {}

    setBellOnComplete(!!display?.bell_on_complete)
    patchUiState({
      compact: !!display?.tui_compact,
      detailsMode: resolveDetailsMode(display),
      statusBar: display?.tui_statusbar !== false
    })
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

  // ── Resize RPC ───────────────────────────────────────────────────

  useEffect(() => {
    if (!ui.sid || !stdout) {
      return
    }

    const onResize = () => rpc('terminal.resize', { session_id: ui.sid, cols: stdout.columns ?? 80 })
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
      const nextTrail = turnRefs.turnToolsRef.current.filter(line => !sameToolTrailGroup(label, line))

      turnRefs.turnToolsRef.current = nextTrail
      turnActions.setTurnTrail(nextTrail)

      rpc('clarify.respond', { answer, request_id: clarify.requestId }).then(r => {
        if (!r) {
          return
        }

        if (answer) {
          turnRefs.persistedToolLabelsRef.current.add(label)
          appendMessage({
            role: 'system',
            text: '',
            kind: 'trail',
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
    [appendMessage, overlay.clarify, rpc, sys, turnActions, turnRefs]
  )

  useEffect(() => {
    if (!ui.sid) {
      return
    }

    rpc('voice.toggle', { action: 'status' }).then((r: any) => setVoiceEnabled(!!r?.enabled))
    rpc<ConfigMtimeResponse>('config.get', { key: 'mtime' }).then(r => {
      configMtimeRef.current = Number(r?.mtime ?? 0)
    })
    rpc<ConfigFullResponse>('config.get', { key: 'full' }).then(applyDisplayConfig)
  }, [applyDisplayConfig, rpc, ui.sid])

  useEffect(() => {
    if (!ui.sid) {
      return
    }

    const id = setInterval(() => {
      rpc<ConfigMtimeResponse>('config.get', { key: 'mtime' }).then(r => {
        const next = Number(r?.mtime ?? 0)

        if (configMtimeRef.current && next && next !== configMtimeRef.current) {
          configMtimeRef.current = next
          rpc('reload.mcp', { session_id: ui.sid }).then(r => {
            if (!r) {
              return
            }

            turnActions.pushActivity('MCP reloaded after config change')
          })
          rpc<ConfigFullResponse>('config.get', { key: 'full' }).then(applyDisplayConfig)
        } else if (!configMtimeRef.current && next) {
          configMtimeRef.current = next
        }
      })
    }, 5000)

    return () => clearInterval(id)
  }, [applyDisplayConfig, turnActions, rpc, ui.sid])

  const idle = turnActions.idle
  const clearReasoning = turnActions.clearReasoning

  const die = useCallback(() => {
    gw.kill()
    exit()
  }, [exit, gw])

  const resetSession = useCallback(() => {
    idle()
    clearReasoning()
    setVoiceRecording(false)
    setVoiceProcessing(false)
    patchUiState({
      bgTasks: new Set(),
      info: null,
      sid: null,
      usage: ZERO
    })
    setHistoryItems([])
    setLastUserMsg('')
    setStickyPrompt('')
    composerActions.setPasteSnips([])
    turnActions.setActivity([])
    turnRefs.turnToolsRef.current = []
    turnRefs.lastStatusNoteRef.current = ''
    turnRefs.protocolWarnedRef.current = false
    turnRefs.persistedToolLabelsRef.current.clear()
  }, [clearReasoning, composerActions, idle, turnActions, turnRefs])

  const resetVisibleHistory = useCallback(
    (info: SessionInfo | null = null) => {
      idle()
      clearReasoning()
      setHistoryItems(info ? [introMsg(info)] : [])
      patchUiState({
        info,
        usage: info?.usage ? { ...ZERO, ...info.usage } : ZERO
      })
      setStickyPrompt('')
      composerActions.setPasteSnips([])
      turnActions.setActivity([])
      setLastUserMsg('')
      turnRefs.turnToolsRef.current = []
      turnRefs.persistedToolLabelsRef.current.clear()
    },
    [clearReasoning, composerActions, idle, turnActions, turnRefs]
  )

  const trimLastExchange = useCallback((items: Msg[]) => {
    const q = [...items]

    while (q.at(-1)?.role === 'assistant' || q.at(-1)?.role === 'tool') {
      q.pop()
    }

    if (q.at(-1)?.role === 'user') {
      q.pop()
    }

    return q
  }, [])

  const guardBusySessionSwitch = useCallback(
    (what = 'switch sessions') => {
      if (!getUiState().busy) {
        return false
      }

      sys(`interrupt the current turn before trying to ${what}`)

      return true
    },
    [sys]
  )

  const closeSession = useCallback(
    (targetSid?: string | null) => {
      if (!targetSid) {
        return Promise.resolve(null)
      }

      return rpc('session.close', { session_id: targetSid })
    },
    [rpc]
  )

  // ── Session management ───────────────────────────────────────────

  const newSession = useCallback(
    async (msg?: string) => {
      await closeSession(getUiState().sid)

      return rpc<SessionCreateResponse>('session.create', { cols: colsRef.current }).then(r => {
        if (!r) {
          patchUiState({ status: 'ready' })

          return
        }

        resetSession()
        setSessionStartedAt(Date.now())
        patchUiState({
          info: r.info ?? null,
          sid: r.session_id,
          status: 'ready',
          usage: r.info?.usage ? { ...ZERO, ...r.info.usage } : ZERO
        })

        if (r.info) {
          setHistoryItems([introMsg(r.info)])
        }

        if (r.info?.credential_warning) {
          sys(`warning: ${r.info.credential_warning}`)
        }

        if (msg) {
          sys(msg)
        }
      })
    },
    [closeSession, resetSession, rpc, sys]
  )

  const resumeById = useCallback(
    (id: string) => {
      patchOverlayState({ picker: false })
      patchUiState({ status: 'resuming…' })
      closeSession(getUiState().sid === id ? null : getUiState().sid).then(() =>
        gw
          .request('session.resume', { cols: colsRef.current, session_id: id })
          .then((raw: any) => {
            const r = asRpcResult(raw)

            if (!r) {
              sys('error: invalid response: session.resume')
              patchUiState({ status: 'ready' })

              return
            }

            resetSession()
            setSessionStartedAt(Date.now())
            const resumed = toTranscriptMessages(r.messages)

            setHistoryItems(r.info ? [introMsg(r.info), ...resumed] : resumed)
            patchUiState({
              info: r.info ?? null,
              sid: r.session_id,
              status: 'ready',
              usage: r.info?.usage ? { ...ZERO, ...r.info.usage } : ZERO
            })
          })
          .catch((e: Error) => {
            sys(`error: ${e.message}`)
            patchUiState({ status: 'ready' })
          })
      )
    },
    [closeSession, gw, resetSession, sys]
  )

  // ── Paste pipeline ───────────────────────────────────────────────

  const paste = useCallback(
    (quiet = false) =>
      rpc('clipboard.paste', { session_id: getUiState().sid }).then((r: any) => {
        if (!r) {
          return
        }

        if (r.attached) {
          const meta = imageTokenMeta(r)
          sys(`📎 Image #${r.count} attached from clipboard${meta ? ` · ${meta}` : ''}`)

          return
        }

        quiet || sys(r.message || 'No image found in clipboard')
      }),
    [rpc, sys]
  )

  clipboardPasteRef.current = paste
  const handleTextPaste = composerActions.handleTextPaste

  // ── Send ─────────────────────────────────────────────────────────

  const send = useCallback(
    (text: string) => {
      const expandPasteSnips = (value: string) => {
        const byLabel = new Map<string, string[]>()

        for (const item of composerState.pasteSnips) {
          const list = byLabel.get(item.label)
          list ? list.push(item.text) : byLabel.set(item.label, [item.text])
        }

        return value.replace(PASTE_SNIPPET_RE, token => byLabel.get(token)?.shift() ?? token)
      }

      const startSubmit = (displayText: string, submitText: string) => {
        const sid = getUiState().sid

        if (!sid) {
          sys('session not ready yet')

          return
        }

        if (turnRefs.statusTimerRef.current) {
          clearTimeout(turnRefs.statusTimerRef.current)
          turnRefs.statusTimerRef.current = null
        }

        setLastUserMsg(text)
        appendMessage({ role: 'user', text: displayText })
        patchUiState({ busy: true, status: 'running…' })
        turnRefs.bufRef.current = ''
        turnRefs.interruptedRef.current = false

        gw.request('prompt.submit', { session_id: sid, text: submitText }).catch((e: Error) => {
          sys(`error: ${e.message}`)
          patchUiState({ busy: false, status: 'ready' })
        })
      }

      const sid = getUiState().sid

      if (!sid) {
        sys('session not ready yet')

        return
      }

      gw.request('input.detect_drop', { session_id: sid, text })
        .then((r: any) => {
          if (r?.matched) {
            if (r.is_image) {
              const meta = imageTokenMeta(r)
              turnActions.pushActivity(`attached image: ${r.name}${meta ? ` · ${meta}` : ''}`)
            } else {
              turnActions.pushActivity(`detected file: ${r.name}`)
            }

            startSubmit(r.text || text, expandPasteSnips(r.text || text))

            return
          }

          startSubmit(text, expandPasteSnips(text))
        })
        .catch(() => startSubmit(text, expandPasteSnips(text)))
    },
    [appendMessage, composerState.pasteSnips, gw, turnActions, sys, turnRefs]
  )

  const shellExec = useCallback(
    (cmd: string) => {
      appendMessage({ role: 'user', text: `!${cmd}` })
      patchUiState({ busy: true, status: 'running…' })

      gw.request('shell.exec', { command: cmd })
        .then((raw: any) => {
          const r = asRpcResult(raw)

          if (!r) {
            sys('error: invalid response: shell.exec')

            return
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
        .finally(() => {
          patchUiState({ busy: false, status: 'ready' })
        })
    },
    [appendMessage, gw, sys]
  )

  const openEditor = composerActions.openEditor

  const interpolate = useCallback(
    (text: string, then: (result: string) => void) => {
      patchUiState({ status: 'interpolating…' })
      const matches = [...text.matchAll(new RegExp(INTERPOLATION_RE.source, 'g'))]

      Promise.all(
        matches.map(m =>
          gw
            .request('shell.exec', { command: m[1]! })
            .then((raw: any) => {
              const r = asRpcResult(raw)

              return [r?.stdout, r?.stderr].filter(Boolean).join('\n').trim()
            })
            .catch(() => '(error)')
        )
      ).then(results => {
        let out = text

        for (let i = matches.length - 1; i >= 0; i--) {
          out = out.slice(0, matches[i]!.index!) + results[i] + out.slice(matches[i]!.index! + matches[i]![0].length)
        }

        then(out)
      })
    },
    [gw]
  )

  const sendQueued = useCallback(
    (text: string) => {
      if (text.startsWith('!')) {
        shellExec(text.slice(1).trim())

        return
      }

      if (hasInterpolation(text)) {
        patchUiState({ busy: true })
        interpolate(text, send)

        return
      }

      send(text)
    },
    [interpolate, send, shellExec]
  )

  // ── Dispatch ─────────────────────────────────────────────────────

  const dispatchSubmission = useCallback(
    (full: string) => {
      const live = getUiState()

      if (!full.trim()) {
        return
      }

      if (!live.sid) {
        sys('session not ready yet')

        return
      }

      if (looksLikeSlashCommand(full)) {
        appendMessage({ role: 'system', text: full, kind: 'slash' })
        composerActions.pushHistory(full)
        slashRef.current(full)
        composerActions.clearIn()

        return
      }

      if (full.startsWith('!')) {
        composerActions.clearIn()
        shellExec(full.slice(1).trim())

        return
      }

      const editIdx = composerRefs.queueEditRef.current
      composerActions.clearIn()

      if (editIdx !== null) {
        composerActions.replaceQueue(editIdx, full)
        const picked = composerRefs.queueRef.current.splice(editIdx, 1)[0]
        composerActions.syncQueue()
        composerActions.setQueueEdit(null)

        if (picked && getUiState().busy && live.sid) {
          composerRefs.queueRef.current.unshift(picked)
          composerActions.syncQueue()

          return
        }

        if (picked && live.sid) {
          sendQueued(picked)
        }

        return
      }

      composerActions.pushHistory(full)

      if (getUiState().busy) {
        composerActions.enqueue(full)

        return
      }

      if (hasInterpolation(full)) {
        patchUiState({ busy: true })
        interpolate(full, send)

        return
      }

      send(full)
    },
    [appendMessage, composerActions, composerRefs, interpolate, send, sendQueued, shellExec, sys]
  )

  // ── Input handling ───────────────────────────────────────────────
  const { pagerPageSize } = useInputHandlers({
    actions: {
      answerClarify,
      appendMessage,
      die,
      dispatchSubmission,
      guardBusySessionSwitch,
      newSession,
      sys
    },
    composer: {
      actions: composerActions,
      refs: composerRefs,
      state: composerState
    },
    gateway,
    terminal: {
      hasSelection,
      scrollRef,
      scrollWithSelection,
      selection,
      stdout
    },
    turn: {
      actions: turnActions,
      refs: turnRefs
    },
    voice: {
      recording: voiceRecording,
      setProcessing: setVoiceProcessing,
      setRecording: setVoiceRecording
    },
    wheelStep: WHEEL_SCROLL_STEP
  })

  // ── Gateway events ───────────────────────────────────────────────

  const onEvent = useMemo(
    () =>
      createGatewayEventHandler({
        composer: {
          dequeue: composerActions.dequeue,
          queueEditRef: composerRefs.queueEditRef,
          sendQueued
        },
        gateway,
        session: {
          STARTUP_RESUME_ID,
          colsRef,
          newSession,
          resetSession,
          setCatalog
        },
        system: {
          bellOnComplete,
          stdout,
          sys
        },
        transcript: {
          appendMessage,
          setHistoryItems
        },
        turn: {
          actions: {
            clearReasoning,
            endReasoningPhase: turnActions.endReasoningPhase,
            idle,
            pruneTransient: turnActions.pruneTransient,
            pulseReasoningStreaming: turnActions.pulseReasoningStreaming,
            pushActivity: turnActions.pushActivity,
            pushTrail: turnActions.pushTrail,
            scheduleReasoning: turnActions.scheduleReasoning,
            scheduleStreaming: turnActions.scheduleStreaming,
            setActivity: turnActions.setActivity,
            setReasoningTokens: turnActions.setReasoningTokens,
            setStreaming: turnActions.setStreaming,
            setSubagents: turnActions.setSubagents,
            setToolTokens: turnActions.setToolTokens,
            setTools: turnActions.setTools,
            setTurnTrail: turnActions.setTurnTrail
          },
          refs: {
            activeToolsRef: turnRefs.activeToolsRef,
            bufRef: turnRefs.bufRef,
            interruptedRef: turnRefs.interruptedRef,
            lastStatusNoteRef: turnRefs.lastStatusNoteRef,
            persistedToolLabelsRef: turnRefs.persistedToolLabelsRef,
            protocolWarnedRef: turnRefs.protocolWarnedRef,
            reasoningRef: turnRefs.reasoningRef,
            statusTimerRef: turnRefs.statusTimerRef,
            toolTokenAccRef: turnRefs.toolTokenAccRef,
            toolCompleteRibbonRef: turnRefs.toolCompleteRibbonRef,
            turnToolsRef: turnRefs.turnToolsRef
          }
        }
      }),
    [
      appendMessage,
      bellOnComplete,
      clearReasoning,
      composerActions,
      composerRefs,
      gateway,
      idle,
      newSession,
      resetSession,
      sendQueued,
      sys,
      turnActions,
      turnRefs,
      stdout
    ]
  )

  onEventRef.current = onEvent

  useEffect(() => {
    const handler = (ev: GatewayEvent) => onEventRef.current(ev)

    const exitHandler = () => {
      patchUiState({ busy: false, sid: null, status: 'gateway exited' })
      turnActions.pushActivity('gateway exited · /logs to inspect', 'error')
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
  }, [gw, turnActions, sys])

  // ── Slash commands ───────────────────────────────────────────────

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
          closeSession,
          die,
          guardBusySessionSwitch,
          newSession,
          resetVisibleHistory,
          resumeById,
          setSessionStartedAt
        },
        transcript: {
          page,
          panel,
          send,
          setHistoryItems,
          sys,
          trimLastExchange
        },
        voice: {
          setVoiceEnabled
        }
      }),
    [
      catalog,
      closeSession,
      composerActions,
      composerRefs,
      die,
      gateway,
      guardBusySessionSwitch,
      hasSelection,
      maybeWarn,
      newSession,
      page,
      panel,
      paste,
      resetVisibleHistory,
      resumeById,
      selection,
      send,
      setSessionStartedAt,
      setHistoryItems,
      setVoiceEnabled,
      sys,
      trimLastExchange
    ]
  )

  slashRef.current = slash

  // ── Submit ───────────────────────────────────────────────────────

  const submit = useCallback(
    (value: string) => {
      if (value.startsWith('/') && composerState.completions.length) {
        const row = composerState.completions[composerState.compIdx]

        if (row?.text) {
          const text =
            value.startsWith('/') && row.text.startsWith('/') && composerState.compReplace > 0
              ? row.text.slice(1)
              : row.text

          const next = value.slice(0, composerState.compReplace) + text

          if (next !== value) {
            composerActions.setInput(next)

            return
          }
        }
      }

      if (!value.trim() && !composerState.inputBuf.length) {
        const live = getUiState()
        const now = Date.now()
        const dbl = now - lastEmptyAt.current < 450
        lastEmptyAt.current = now

        if (dbl && live.busy && live.sid) {
          turnActions.interruptTurn({
            appendMessage,
            gw,
            sid: live.sid,
            sys
          })

          return
        }

        if (dbl && composerRefs.queueRef.current.length) {
          const next = composerActions.dequeue()

          if (next && live.sid) {
            composerActions.setQueueEdit(null)
            dispatchSubmission(next)
          }
        }

        return
      }

      lastEmptyAt.current = 0

      if (value.endsWith('\\')) {
        composerActions.setInputBuf(prev => [...prev, value.slice(0, -1)])
        composerActions.setInput('')

        return
      }

      dispatchSubmission([...composerState.inputBuf, value].join('\n'))
    },
    [appendMessage, composerActions, composerRefs, composerState, dispatchSubmission, gw, sys, turnActions]
  )

  submitRef.current = submit

  // ── Derived ──────────────────────────────────────────────────────

  const statusColor =
    ui.status === 'ready'
      ? ui.theme.color.ok
      : ui.status.startsWith('error')
        ? ui.theme.color.error
        : ui.status === 'interrupted'
          ? ui.theme.color.warn
          : ui.theme.color.dim

  const sessionStarted = ui.sid ? sessionStartedAt : null
  const voiceLabel = voiceRecording ? 'REC' : voiceProcessing ? 'STT' : `voice ${voiceEnabled ? 'on' : 'off'}`
  const cwdLabel = shortCwd(ui.info?.cwd || process.env.HERMES_CWD || process.cwd())
  const showStreamingArea = Boolean(turnState.streaming)
  const showStickyPrompt = !!stickyPrompt

  const hasReasoning = Boolean(turnState.reasoning.trim())

  const showProgressArea =
    ui.detailsMode === 'hidden'
      ? turnState.activity.some(item => item.tone !== 'info')
      : Boolean(
          ui.busy ||
          turnState.subagents.length ||
          turnState.tools.length ||
          turnState.turnTrail.length ||
          hasReasoning ||
          turnState.activity.length
        )

  const answerApproval = useCallback(
    (choice: string) => {
      rpc('approval.respond', { choice, session_id: ui.sid }).then(r => {
        if (!r) {
          return
        }

        patchOverlayState({ approval: null })
        sys(choice === 'deny' ? 'denied' : `approved (${choice})`)
        patchUiState({ status: 'running…' })
      })
    },
    [rpc, sys, ui.sid]
  )

  const answerSudo = useCallback(
    (pw: string) => {
      if (!overlay.sudo) {
        return
      }

      rpc('sudo.respond', { request_id: overlay.sudo.requestId, password: pw }).then(r => {
        if (!r) {
          return
        }

        patchOverlayState({ sudo: null })
        patchUiState({ status: 'running…' })
      })
    },
    [overlay.sudo, rpc]
  )

  const answerSecret = useCallback(
    (value: string) => {
      if (!overlay.secret) {
        return
      }

      rpc('secret.respond', { request_id: overlay.secret.requestId, value }).then(r => {
        if (!r) {
          return
        }

        patchOverlayState({ secret: null })
        patchUiState({ status: 'running…' })
      })
    },
    [overlay.secret, rpc]
  )

  const onModelSelect = useCallback((value: string) => {
    patchOverlayState({ modelPicker: false })
    slashRef.current(`/model ${value}`)
  }, [])

  const appActions = useMemo(
    () => ({
      answerApproval,
      answerClarify,
      answerSecret,
      answerSudo,
      onModelSelect,
      resumeById,
      setStickyPrompt
    }),
    [answerApproval, answerClarify, answerSecret, answerSudo, onModelSelect, resumeById, setStickyPrompt]
  )

  const appComposer = useMemo(
    () => ({
      cols,
      compIdx: composerState.compIdx,
      completions: composerState.completions,
      empty,
      handleTextPaste,
      input: composerState.input,
      inputBuf: composerState.inputBuf,
      pagerPageSize,
      queueEditIdx: composerState.queueEditIdx,
      queuedDisplay: composerState.queuedDisplay,
      submit,
      updateInput: composerActions.setInput
    }),
    [
      cols,
      composerActions.setInput,
      composerState.compIdx,
      composerState.completions,
      composerState.input,
      composerState.inputBuf,
      composerState.queueEditIdx,
      composerState.queuedDisplay,
      empty,
      handleTextPaste,
      pagerPageSize,
      submit
    ]
  )

  const appProgress = useMemo(
    () => ({
      activity: turnState.activity,
      reasoning: turnState.reasoning,
      reasoningActive: turnState.reasoningActive,
      reasoningStreaming: turnState.reasoningStreaming,
      reasoningTokens: turnState.reasoningTokens,
      showProgressArea,
      showStreamingArea,
      streaming: turnState.streaming,
      subagents: turnState.subagents,
      toolTokens: turnState.toolTokens,
      tools: turnState.tools,
      turnTrail: turnState.turnTrail
    }),
    [showProgressArea, showStreamingArea, turnState]
  )

  const appStatus = useMemo(
    () => ({
      cwdLabel,
      sessionStartedAt: sessionStarted,
      showStickyPrompt,
      statusColor,
      stickyPrompt,
      voiceLabel
    }),
    [cwdLabel, sessionStarted, showStickyPrompt, statusColor, stickyPrompt, voiceLabel]
  )

  const appTranscript = useMemo(
    () => ({
      historyItems,
      scrollRef,
      virtualHistory,
      virtualRows
    }),
    [historyItems, scrollRef, virtualHistory, virtualRows]
  )

  // ── Render ───────────────────────────────────────────────────────

  return (
    <GatewayProvider value={gateway}>
      <AppLayout
        actions={appActions}
        composer={appComposer}
        mouseTracking={MOUSE_TRACKING}
        progress={appProgress}
        status={appStatus}
        transcript={appTranscript}
      />
    </GatewayProvider>
  )
}
