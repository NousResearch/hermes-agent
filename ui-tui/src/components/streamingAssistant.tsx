import { useStore } from '@nanostores/react'
import { memo } from 'react'

import type { AppLayoutProgressProps } from '../app/interfaces.js'
import { toggleTodoCollapsed, useTurnSelector } from '../app/turnStore.js'
import { $uiState } from '../app/uiStore.js'
import { blockRenders } from '../domain/blockLayout.js'
import { appendToolShelfMessage } from '../lib/liveProgress.js'
import type { ActiveTool, DetailsMode, Msg, SectionVisibility } from '../types.js'

import { MessageLine } from './messageLine.js'
import { TodoPanel } from './todoPanel.js'

const groupedSegments = (segments: Msg[]): Msg[] =>
  segments.reduce<Msg[]>((acc, msg) => appendToolShelfMessage(acc, msg), [])

interface LiveBlock {
  isStreaming?: boolean
  key: string
  msg: Msg
  tools?: ActiveTool[]
}

export const StreamingAssistant = memo(function StreamingAssistant({
  cols,
  compact,
  detailsMode,
  detailsModeCommandOverride,
  prevMsg,
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

  // Flatten the live area into one ordered list so each block's leading gap
  // can be derived from the block directly above it — including the boundary
  // back into settled history (prevMsg). Tracking the predecessor rather than
  // the live text is what keeps the streaming block from jumping when it
  // flushes into a settled segment.
  const blocks: LiveBlock[] = groupedSegments(streamSegments).map((msg, i) => ({ key: `seg:${i}`, msg }))

  if (activeTools.length) {
    blocks.push({ key: 'active-tools', msg: { kind: 'trail', role: 'system', text: '' }, tools: activeTools })
  }

  if (showStreamingArea) {
    blocks.push({
      isStreaming: true,
      key: 'streaming',
      msg: { role: 'assistant', text: streaming, ...(streamPendingTools.length && { tools: streamPendingTools }) }
    })
  } else if (streamPendingTools.length) {
    blocks.push({ key: 'pending-tools', msg: { kind: 'trail', role: 'system', text: '', tools: streamPendingTools } })
  }

  const detailsCtx = { commandOverride: detailsModeCommandOverride, detailsMode, sections }
  let prev = prevMsg

  return (
    <>
      {blocks.map(block => {
        const node = (
          <MessageLine
            cols={cols}
            compact={compact}
            detailsMode={detailsMode}
            detailsModeCommandOverride={detailsModeCommandOverride}
            isStreaming={block.isStreaming}
            key={block.key}
            msg={block.msg}
            prev={prev}
            sections={sections}
            t={ui.theme}
            {...(block.tools ? { tools: block.tools } : {})}
          />
        )

        // Advance the grouping predecessor only past blocks that actually
        // paint, so a trail hidden by /details stays transparent here too
        // (active tools live in the prop, so fold them into the check).
        const checkMsg = block.tools?.length ? { ...block.msg, tools: block.tools.map(tool => tool.name) } : block.msg

        if (blockRenders(checkMsg, detailsCtx)) {
          prev = block.msg
        }

        return node
      })}
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
  prevMsg?: Msg
  progress: AppLayoutProgressProps
  sections?: SectionVisibility
}
