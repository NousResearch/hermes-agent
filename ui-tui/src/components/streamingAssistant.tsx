import { useStore } from '@nanostores/react'
import { Box, Text } from '@hermes/ink'
import { memo, useEffect, useState } from 'react'

import type { AppLayoutProgressProps } from '../app/interfaces.js'
import { toggleTodoCollapsed, useTurnSelector } from '../app/turnStore.js'
import { $uiState } from '../app/uiStore.js'
import { fmtDuration } from '../domain/messages.js'
import { appendToolShelfMessage } from '../lib/liveProgress.js'
import type { DetailsMode, Msg, SectionVisibility } from '../types.js'

import { MessageLine } from './messageLine.js'
import { TodoPanel } from './todoPanel.js'

const groupedSegments = (segments: Msg[]): Msg[] =>
  segments.reduce<Msg[]>((acc, msg) => appendToolShelfMessage(acc, msg), [])

const WORKING_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
const WORKING_BAR_WIDTH = 14

const workingBar = (tick: number, width = WORKING_BAR_WIDTH) => {
  const head = tick % width

  return `[${Array.from({ length: width }, (_, index) => (index === head ? '◆' : index < head ? '━' : '─')).join('')}]`
}

const InlineProgress = memo(function InlineProgress({ progress }: { progress: AppLayoutProgressProps }) {
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    if (!progress.busy) {
      return
    }

    setNow(Date.now())
    const id = setInterval(() => setNow(Date.now()), 250)

    return () => clearInterval(id)
  }, [progress.busy, progress.turnStartedAt])

  if (!progress.busy) {
    return null
  }

  const elapsedMs = progress.turnStartedAt ? now - progress.turnStartedAt : 0
  const elapsed = progress.turnStartedAt ? fmtDuration(elapsedMs) : '…'
  const tick = Math.floor(elapsedMs / 250)

  return (
    <Box marginBottom={1} marginTop={1}>
      <Text color={progress.statusColor}>
        {WORKING_FRAMES[tick % WORKING_FRAMES.length]} Working {elapsed} {workingBar(tick)}
        <Text dim> Ctrl+C to interrupt</Text>
      </Text>
    </Box>
  )
})

export const StreamingAssistant = memo(function StreamingAssistant({
  cols,
  compact,
  detailsMode,
  detailsModeCommandOverride,
  progress,
  sections
}: StreamingAssistantProps) {
  const ui = useStore($uiState)
  const streamSegments = useTurnSelector(state => state.streamSegments)
  const streamPendingTools = useTurnSelector(state => state.streamPendingTools)
  const streaming = useTurnSelector(state => state.streaming)
  const activeTools = useTurnSelector(state => state.tools)
  const showStreamingArea = Boolean(streaming)

  if (!progress.busy && !progress.showProgressArea && !showStreamingArea && !activeTools.length) {
    return null
  }

  return (
    <>
      {groupedSegments(streamSegments).map((msg, i) => (
        <MessageLine
          cols={cols}
          compact={compact}
          detailsMode={detailsMode}
          detailsModeCommandOverride={detailsModeCommandOverride}
          key={`seg:${i}`}
          msg={msg}
          sections={sections}
          t={ui.theme}
        />
      ))}

      {!!activeTools.length && (
        <MessageLine
          cols={cols}
          compact={compact}
          detailsMode={detailsMode}
          detailsModeCommandOverride={detailsModeCommandOverride}
          msg={{ kind: 'trail', role: 'system', text: '' }}
          sections={sections}
          t={ui.theme}
          tools={activeTools}
        />
      )}

      {showStreamingArea && (
        <MessageLine
          cols={cols}
          compact={compact}
          detailsMode={detailsMode}
          detailsModeCommandOverride={detailsModeCommandOverride}
          isStreaming
          msg={{
            role: 'assistant',
            text: streaming,
            ...(streamPendingTools.length && { tools: streamPendingTools })
          }}
          sections={sections}
          t={ui.theme}
        />
      )}

      {!showStreamingArea && !!streamPendingTools.length && (
        <MessageLine
          cols={cols}
          compact={compact}
          detailsMode={detailsMode}
          detailsModeCommandOverride={detailsModeCommandOverride}
          msg={{ kind: 'trail', role: 'system', text: '', tools: streamPendingTools }}
          sections={sections}
          t={ui.theme}
        />
      )}

      <LiveTodoPanel />
      <InlineProgress progress={progress} />
    </>
  )
})

export const LiveTodoPanel = memo(function LiveTodoPanel() {
  const ui = useStore($uiState)
  const todos = useTurnSelector(state => state.todos)
  const collapsed = useTurnSelector(state => state.todoCollapsed)

  return <TodoPanel collapsed={collapsed} onToggle={toggleTodoCollapsed} t={ui.theme} todos={todos} />
})

interface StreamingAssistantProps {
  cols: number
  compact?: boolean
  detailsMode: DetailsMode
  detailsModeCommandOverride: boolean
  progress: AppLayoutProgressProps
  sections?: SectionVisibility
}
