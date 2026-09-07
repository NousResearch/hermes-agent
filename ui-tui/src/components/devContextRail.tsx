import { Box, ScrollBox, Text, useStdout } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { memo, type ReactNode } from 'react'

import { $overlayState } from '../app/overlayStore.js'
import { $turnState } from '../app/turnStore.js'
import { $uiState, $uiTheme } from '../app/uiStore.js'
import {
  devContextHasActivity,
  devContextPlacement,
  devContextRailWidth
} from '../domain/devContext.js'
import type { RailInputs } from '../domain/railInputs.js'
import { truncateDisplay, type WorkspaceHudSnapshot } from '../domain/workspaceHud.js'
import type { RailFlowStatus } from '../hooks/useContextRailInputs.js'
import { countPendingTodos } from '../lib/liveProgress.js'
import { useAmbientRailWidth } from '../sdk/host.js'
import type { Theme } from '../theme.js'
import type { SubagentProgress, TodoItem } from '../types.js'

export interface DevContextRailProps {
  cols: number
  flowStatus?: RailFlowStatus | null
  queuedCount: number
  railInputs?: RailInputs | null
  workspace?: WorkspaceHudSnapshot
}

const SECRET_ASSIGNMENT = /\b(?:token|password|passwd|secret|api[_-]?key|authorization)\s*[:=]\s*[^\s]+/gi
const URL_OR_REMOTE = /(?:https?|ssh):\/\/[^\s]+|git@github\.com:[^\s]+/gi

const safeText = (value: string): string =>
  value
    .replace(URL_OR_REMOTE, '[url]')
    .replace(SECRET_ASSIGNMENT, '[redacted]')
    .split('')
    .map(char => {
      const code = char.codePointAt(0) ?? 0

      return code <= 0x1f || code === 0x7f ? ' ' : char
    })
    .join('')
    .replace(/\s+/g, ' ')
    .trim()

const displayText = (value: string): string => truncateDisplay(safeText(value), 120)

const sourceName = (path: string): string => displayText(path.split(/[\\/]/).pop() || 'repo context')

const formatClock = (at: number): string => {
  const date = new Date(at)

  if (!Number.isFinite(date.getTime())) {
    return 'unknown'
  }

  return `${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`
}

const statusGlyph = (status: SubagentProgress['status']): string => {
  switch (status) {
    case 'completed':
      return '✓'

    case 'error':

    case 'failed':

    case 'timeout':
      return '×'

    case 'interrupted':
      return '⏸'

    case 'queued':
      return '·'

    case 'running':
      return '▶'
  }
}

const statusColor = (status: SubagentProgress['status'], t: Theme): string => {
  switch (status) {
    case 'completed':
      return t.color.ok

    case 'error':

    case 'failed':

    case 'timeout':
      return t.color.error

    case 'interrupted':

    case 'queued':
      return t.color.muted

    case 'running':
      return t.color.accent
  }
}

const todoGlyph = (status: TodoItem['status']): string => {
  switch (status) {
    case 'completed':
      return '✓'

    case 'cancelled':
      return '×'

    case 'in_progress':
      return '▶'

    case 'pending':
      return '·'
  }
}

const todoColor = (status: TodoItem['status'], t: Theme): string => {
  switch (status) {
    case 'completed':
      return t.color.ok

    case 'cancelled':
      return t.color.muted

    case 'in_progress':
      return t.color.accent

    case 'pending':
      return t.color.text
  }
}

const Section = memo(function Section({
  at,
  children,
  marginTop = 1,
  source,
  t,
  title
}: {
  at: number
  children: ReactNode
  marginTop?: number
  source: string
  t: Theme
  title: string
}) {
  return (
    <Box flexDirection="column" flexShrink={0} marginTop={marginTop}>
      <Text bold color={t.color.accent}>
        {title}
      </Text>
      <Text color={t.color.muted} wrap="truncate-end">
        {source} · as of {formatClock(at)}
      </Text>
      {children}
    </Box>
  )
})

const Row = memo(function Row({ color, label, t, value }: { color?: string; label: string; t: Theme; value: string }) {
  return (
    <Text color={color ?? t.color.text} wrap="truncate-end">
      <Text color={t.color.label}>{label}: </Text>
      {displayText(value)}
    </Text>
  )
})

const TodoRow = memo(function TodoRow({ t, todo }: { t: Theme; todo: TodoItem }) {
  return (
    <Text color={todoColor(todo.status, t)} wrap="wrap">
      {todo.parent ? '  ' : ''}
      {todoGlyph(todo.status)}{' '}
      <Text color={todo.status === 'cancelled' ? t.color.muted : t.color.text}>{displayText(todo.content)}</Text>
    </Text>
  )
})

const AgentRow = memo(function AgentRow({ agent, t }: { agent: SubagentProgress; t: Theme }) {
  return (
    <Text color={statusColor(agent.status, t)} wrap="wrap">
      {statusGlyph(agent.status)} <Text color={t.color.text}>{displayText(agent.goal)}</Text>
    </Text>
  )
})

interface DevContextModel {
  activeAgents: SubagentProgress[]
  attention: string | null
  backgroundTaskCount: number
  enabled: boolean
  flowStatus: RailFlowStatus | null | undefined
  hasActivity: boolean
  observedAt: number
  queuedCount: number
  railInputs: RailInputs | null | undefined
  t: Theme
  todos: TodoItem[]
  toolCount: number
  workspace: WorkspaceHudSnapshot | undefined
}

const attentionText = (overlay: ReturnType<typeof $overlayState.get>): string | null => {
  if (overlay.approval) {
    return `approval: ${overlay.approval.command}`
  }

  if (overlay.clarify) {
    return `question: ${overlay.clarify.question}`
  }

  if (overlay.confirm) {
    return `confirm: ${overlay.confirm.title}`
  }

  if (overlay.secret) {
    return `secret required: ${overlay.secret.envVar}`
  }

  if (overlay.sudo) {
    return 'sudo password required'
  }

  if (overlay.billing || overlay.subscription) {
    return 'account action pending'
  }

  return null
}

const useDevContextModel = ({
  flowStatus,
  queuedCount,
  railInputs,
  workspace
}: DevContextRailProps): DevContextModel => {
  const ui = useStore($uiState)
  const turn = useStore($turnState)
  const overlay = useStore($overlayState)
  const t = useStore($uiTheme)
  const activeAgents = turn.subagents.filter(agent => agent.status === 'queued' || agent.status === 'running')
  const attention = attentionText(overlay)
  const observedAt = Date.now()

  return {
    activeAgents,
    attention,
    backgroundTaskCount: ui.bgTasks.size,
    enabled: ui.devContext,
    flowStatus,
    hasActivity: devContextHasActivity(
      turn.todos.length,
      activeAgents.length,
      ui.bgTasks.size,
      turn.tools.length,
      queuedCount,
      Boolean(railInputs),
      Boolean(attention)
    ),
    observedAt,
    queuedCount,
    railInputs,
    t,
    todos: turn.todos,
    toolCount: turn.tools.length,
    workspace
  }
}

const DevContextContent = memo(function DevContextContent({
  activeAgents,
  attention,
  backgroundTaskCount,
  flowStatus,
  observedAt,
  queuedCount,
  railInputs,
  t,
  todos,
  toolCount,
  workspace
}: Omit<DevContextModel, 'enabled' | 'hasActivity'>) {
  const completedTodos = todos.filter(todo => todo.status === 'completed' || todo.status === 'cancelled').length
  const sourceAt = railInputs?.mtimeMs ?? observedAt
  const focusAt = flowStatus?.at ?? sourceAt

  const evidenceVisible =
    Boolean(railInputs) ||
    Boolean(workspace?.branch || workspace?.gitRoot || (workspace?.dirty !== null && workspace?.dirty !== undefined))

  const workValue = [
    activeAgents.length ? `agents ${activeAgents.length}` : '',
    backgroundTaskCount ? `bg ${backgroundTaskCount}` : '',
    toolCount ? `tools ${toolCount}` : '',
    queuedCount ? `queue ${queuedCount}` : ''
  ]
    .filter(Boolean)
    .join(' · ')

  const flowProgress = !railInputs
    ? 'not connected'
    : railInputs.flow.trim() === 'none'
      ? 'not configured'
      : flowStatus?.text || 'not connected'

  return (
    <>
      {attention && (
        <Section at={observedAt} marginTop={0} source="PROMPT" t={t} title="NEEDS ME">
          <Row color={t.color.warn} label="attention" t={t} value={attention} />
        </Section>
      )}

      {todos.length > 0 && (
        <Section at={observedAt} marginTop={attention ? 1 : 0} source="SESSION CHECKLIST" t={t} title="PLAN">
          <Row
            color={completedTodos === todos.length ? t.color.ok : t.color.text}
            label="progress"
            t={t}
            value={`${completedTodos}/${todos.length} done · ${countPendingTodos(todos)} open`}
          />
          {todos.map(todo => (
            <TodoRow key={todo.id} t={t} todo={todo} />
          ))}
        </Section>
      )}

      {railInputs && (
        <Section at={sourceAt} source="REPO INPUTS" t={t} title="PROJECT">
          <Row label="product" t={t} value={railInputs.product || 'not connected'} />
        </Section>
      )}

      {railInputs && (
        <Section at={focusAt} source="FLOW" t={t} title="FOCUS">
          <Row label="tree" t={t} value={railInputs.flow || 'not connected'} />
          <Row label="progress" t={t} value={flowProgress} />
        </Section>
      )}

      {railInputs && railInputs.decisions.length > 0 && (
        <Section at={sourceAt} source={sourceName(railInputs.sourceFile)} t={t} title="DECISIONS">
          {railInputs.decisions.map((decision, index) => (
            <Row key={`${decision}-${index}`} label="decision" t={t} value={decision} />
          ))}
        </Section>
      )}

      {evidenceVisible && (
        <Section at={observedAt} source="GIT + CHECKS" t={t} title="EVIDENCE">
          <Row label="branch" t={t} value={workspace?.branch || 'not connected'} />
          <Row
            label="git"
            t={t}
            value={workspace?.dirty === true ? 'dirty' : workspace?.dirty === false ? 'clean' : 'not connected'}
          />
          <Row label="checks" t={t} value={railInputs?.checks || 'not connected'} />
          <Row label="durable" t={t} value={railInputs?.evidence || 'not connected'} />
        </Section>
      )}

      <Section at={observedAt} source="EXECUTION" t={t} title="WORK">
        <Row label="active" t={t} value={workValue || 'idle'} />
        {activeAgents.map(agent => (
          <AgentRow agent={agent} key={agent.id} t={t} />
        ))}
      </Section>
    </>
  )
})

const DevContextHeader = memo(function DevContextHeader({ t }: { t: Theme }) {
  return (
    <Text bold color={t.color.primary}>
      DEV CONTEXT
    </Text>
  )
})

const contentProps = (model: DevContextModel) => ({
  activeAgents: model.activeAgents,
  attention: model.attention,
  backgroundTaskCount: model.backgroundTaskCount,
  flowStatus: model.flowStatus,
  observedAt: model.observedAt,
  queuedCount: model.queuedCount,
  railInputs: model.railInputs,
  t: model.t,
  todos: model.todos,
  toolCount: model.toolCount,
  workspace: model.workspace
})

export const DevContextRail = memo(function DevContextRail(props: DevContextRailProps) {
  const model = useDevContextModel(props)
  const ambientRailColumns = useAmbientRailWidth('left') + useAmbientRailWidth('right')
  const placement = devContextPlacement(model.enabled, props.cols, ambientRailColumns, model.hasActivity)
  const railWidth = devContextRailWidth(model.enabled, props.cols, ambientRailColumns, model.hasActivity)

  if (placement !== 'side') {
    return null
  }

  return (
    <Box
      borderColor={model.t.color.border}
      borderStyle="round"
      flexDirection="column"
      flexShrink={0}
      overflow="hidden"
      width={railWidth}
    >
      <Box paddingX={1}>
        <DevContextHeader t={model.t} />
      </Box>

      <ScrollBox flexDirection="column" flexGrow={1} flexShrink={1} minHeight={0} paddingX={1}>
        <DevContextContent {...contentProps(model)} />
      </ScrollBox>
    </Box>
  )
})

export const DevContextBottomDock = memo(function DevContextBottomDock(props: DevContextRailProps) {
  const model = useDevContextModel(props)
  const { stdout } = useStdout()
  const ambientRailColumns = useAmbientRailWidth('left') + useAmbientRailWidth('right')
  const placement = devContextPlacement(model.enabled, props.cols, ambientRailColumns, model.hasActivity)

  if (placement !== 'bottom') {
    return null
  }

  return (
    <Box
      borderColor={model.t.color.border}
      borderStyle="round"
      flexDirection="column"
      flexShrink={0}
      marginTop={1}
      paddingX={1}
      width={Math.max(1, props.cols - 2)}
    >
      <DevContextHeader t={model.t} />
      <ScrollBox
        flexDirection="column"
        flexGrow={1}
        flexShrink={1}
        maxHeight={Math.max(3, (stdout?.rows ?? 24) - 8)}
        minHeight={0}
      >
        <DevContextContent {...contentProps(model)} />
      </ScrollBox>
    </Box>
  )
})
