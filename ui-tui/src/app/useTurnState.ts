import {
  type Dispatch,
  type MutableRefObject,
  type SetStateAction,
  useCallback,
  useEffect,
  useRef,
  useState
} from 'react'

import { isTransientTrailLine, sameToolTrailGroup } from '../lib/text.js'
import type { ActiveTool, ActivityItem, Msg } from '../types.js'

import { REASONING_PULSE_MS, STREAM_BATCH_MS } from './constants.js'
import type { ToolCompleteRibbon } from './interfaces.js'
import { resetOverlayState } from './overlayStore.js'
import { patchUiState } from './uiStore.js'

export interface InterruptTurnOptions {
  appendMessage: (msg: Msg) => void
  gw: { request: (method: string, params?: Record<string, unknown>) => Promise<unknown> }
  sid: string
  sys: (text: string) => void
}

export interface TurnActions {
  clearReasoning: () => void
  endReasoningPhase: () => void
  idle: () => void
  interruptTurn: (options: InterruptTurnOptions) => void
  pruneTransient: () => void
  pulseReasoningStreaming: () => void
  pushActivity: (text: string, tone?: ActivityItem['tone'], replaceLabel?: string) => void
  pushTrail: (line: string) => void
  scheduleReasoning: () => void
  scheduleStreaming: () => void
  setActivity: Dispatch<SetStateAction<ActivityItem[]>>
  setReasoning: Dispatch<SetStateAction<string>>
  setReasoningActive: Dispatch<SetStateAction<boolean>>
  setReasoningStreaming: Dispatch<SetStateAction<boolean>>
  setStreaming: Dispatch<SetStateAction<string>>
  setTools: Dispatch<SetStateAction<ActiveTool[]>>
  setTurnTrail: Dispatch<SetStateAction<string[]>>
}

export interface TurnRefs {
  bufRef: MutableRefObject<string>
  interruptedRef: MutableRefObject<boolean>
  lastStatusNoteRef: MutableRefObject<string>
  persistedToolLabelsRef: MutableRefObject<Set<string>>
  protocolWarnedRef: MutableRefObject<boolean>
  reasoningRef: MutableRefObject<string>
  reasoningStreamingTimerRef: MutableRefObject<ReturnType<typeof setTimeout> | null>
  reasoningTimerRef: MutableRefObject<ReturnType<typeof setTimeout> | null>
  statusTimerRef: MutableRefObject<ReturnType<typeof setTimeout> | null>
  streamTimerRef: MutableRefObject<ReturnType<typeof setTimeout> | null>
  toolCompleteRibbonRef: MutableRefObject<ToolCompleteRibbon | null>
  turnToolsRef: MutableRefObject<string[]>
}

export interface TurnState {
  activity: ActivityItem[]
  reasoning: string
  reasoningActive: boolean
  reasoningStreaming: boolean
  streaming: string
  tools: ActiveTool[]
  turnTrail: string[]
}

export interface UseTurnStateResult {
  actions: TurnActions
  refs: TurnRefs
  state: TurnState
}

export function useTurnState(): UseTurnStateResult {
  const [activity, setActivity] = useState<ActivityItem[]>([])
  const [reasoning, setReasoning] = useState('')
  const [reasoningActive, setReasoningActive] = useState(false)
  const [reasoningStreaming, setReasoningStreaming] = useState(false)
  const [streaming, setStreaming] = useState('')
  const [tools, setTools] = useState<ActiveTool[]>([])
  const [turnTrail, setTurnTrail] = useState<string[]>([])

  const activityIdRef = useRef(0)
  const bufRef = useRef('')
  const interruptedRef = useRef(false)
  const lastStatusNoteRef = useRef('')
  const persistedToolLabelsRef = useRef<Set<string>>(new Set())
  const protocolWarnedRef = useRef(false)
  const reasoningRef = useRef('')
  const reasoningStreamingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const reasoningTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const statusTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const streamTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const toolCompleteRibbonRef = useRef<ToolCompleteRibbon | null>(null)
  const turnToolsRef = useRef<string[]>([])

  const setTrail = (next: string[]) => {
    turnToolsRef.current = next

    return next
  }

  const pulseReasoningStreaming = useCallback(() => {
    if (reasoningStreamingTimerRef.current) {
      clearTimeout(reasoningStreamingTimerRef.current)
    }

    setReasoningActive(true)
    setReasoningStreaming(true)
    reasoningStreamingTimerRef.current = setTimeout(() => {
      reasoningStreamingTimerRef.current = null
      setReasoningStreaming(false)
    }, REASONING_PULSE_MS)
  }, [])

  const scheduleStreaming = useCallback(() => {
    if (streamTimerRef.current) {
      return
    }

    streamTimerRef.current = setTimeout(() => {
      streamTimerRef.current = null
      setStreaming(bufRef.current.trimStart())
    }, STREAM_BATCH_MS)
  }, [])

  const scheduleReasoning = useCallback(() => {
    if (reasoningTimerRef.current) {
      return
    }

    reasoningTimerRef.current = setTimeout(() => {
      reasoningTimerRef.current = null
      setReasoning(reasoningRef.current)
    }, STREAM_BATCH_MS)
  }, [])

  const endReasoningPhase = useCallback(() => {
    if (reasoningStreamingTimerRef.current) {
      clearTimeout(reasoningStreamingTimerRef.current)
      reasoningStreamingTimerRef.current = null
    }

    setReasoningStreaming(false)
    setReasoningActive(false)
  }, [])

  useEffect(
    () => () => {
      if (streamTimerRef.current) {
        clearTimeout(streamTimerRef.current)
      }

      if (reasoningTimerRef.current) {
        clearTimeout(reasoningTimerRef.current)
      }

      if (reasoningStreamingTimerRef.current) {
        clearTimeout(reasoningStreamingTimerRef.current)
      }
    },
    []
  )

  const pushActivity = useCallback((text: string, tone: ActivityItem['tone'] = 'info', replaceLabel?: string) => {
    setActivity(prev => {
      const base = replaceLabel ? prev.filter(item => !sameToolTrailGroup(replaceLabel, item.text)) : prev

      if (base.at(-1)?.text === text && base.at(-1)?.tone === tone) {
        return base
      }

      activityIdRef.current++

      return [...base, { id: activityIdRef.current, text, tone }].slice(-8)
    })
  }, [])

  const pruneTransient = useCallback(() => {
    setTurnTrail(prev => {
      const next = prev.filter(line => !isTransientTrailLine(line))

      return next.length === prev.length ? prev : setTrail(next)
    })
  }, [])

  const pushTrail = useCallback((line: string) => {
    setTurnTrail(prev =>
      prev.at(-1) === line ? prev : setTrail([...prev.filter(item => !isTransientTrailLine(item)), line].slice(-8))
    )
  }, [])

  const clearReasoning = useCallback(() => {
    if (reasoningTimerRef.current) {
      clearTimeout(reasoningTimerRef.current)
      reasoningTimerRef.current = null
    }

    reasoningRef.current = ''
    setReasoning('')
  }, [])

  const idle = useCallback(() => {
    endReasoningPhase()
    setTools([])
    setTurnTrail([])
    patchUiState({ busy: false })
    resetOverlayState()

    if (streamTimerRef.current) {
      clearTimeout(streamTimerRef.current)
      streamTimerRef.current = null
    }

    setStreaming('')
    bufRef.current = ''
  }, [endReasoningPhase])

  const interruptTurn = useCallback(
    ({ appendMessage, gw, sid, sys }: InterruptTurnOptions) => {
      interruptedRef.current = true
      gw.request('session.interrupt', { session_id: sid }).catch(() => {})
      const partial = (streaming || bufRef.current).trimStart()

      if (partial) {
        appendMessage({ role: 'assistant', text: partial + '\n\n*[interrupted]*' })
      } else {
        sys('interrupted')
      }

      idle()
      clearReasoning()
      setActivity([])
      turnToolsRef.current = []
      patchUiState({ status: 'interrupted' })

      if (statusTimerRef.current) {
        clearTimeout(statusTimerRef.current)
      }

      statusTimerRef.current = setTimeout(() => {
        statusTimerRef.current = null
        patchUiState({ status: 'ready' })
      }, 1500)
    },
    [clearReasoning, idle, streaming]
  )

  return {
    actions: {
      clearReasoning,
      endReasoningPhase,
      idle,
      interruptTurn,
      pruneTransient,
      pulseReasoningStreaming,
      pushActivity,
      pushTrail,
      scheduleReasoning,
      scheduleStreaming,
      setActivity,
      setReasoning,
      setReasoningActive,
      setReasoningStreaming,
      setStreaming,
      setTools,
      setTurnTrail
    },
    refs: {
      bufRef,
      interruptedRef,
      lastStatusNoteRef,
      persistedToolLabelsRef,
      protocolWarnedRef,
      reasoningRef,
      reasoningStreamingTimerRef,
      reasoningTimerRef,
      statusTimerRef,
      streamTimerRef,
      toolCompleteRibbonRef,
      turnToolsRef
    },
    state: {
      activity,
      reasoning,
      reasoningActive,
      reasoningStreaming,
      streaming,
      tools,
      turnTrail
    }
  }
}
