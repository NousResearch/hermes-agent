import { useStore } from '@nanostores/react'
import { memo } from 'react'

import type { AppLayoutProgressProps } from '../app/interfaces.js'
import { toggleTodoCollapsed, useTurnSelector } from '../app/turnStore.js'
import { $uiState } from '../app/uiStore.js'
import { appendToolShelfMessage } from '../lib/liveProgress.js'
import type { DetailsMode, Msg, SectionVisibility } from '../types.js'

import { MessageLine } from './messageLine.js'
import { TodoPanel } from './todoPanel.js'

const groupedSegments = (segments: Msg[]): Msg[] =>
  segments.reduce<Msg[]>((acc, msg) => appendToolShelfMessage(acc, msg), [])

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

  if (!progress.showProgressArea && !showStreamingArea && !activeTools.length) {
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
