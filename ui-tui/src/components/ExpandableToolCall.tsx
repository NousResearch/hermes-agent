import { Box, NoSelect, Text } from '@hermes/ink'
import { useMemo } from 'react'
import type { ActiveTool } from '../types.js'
import type { Theme } from '../theme.js'
import type { TreeBranch, TreeRails } from './treeRow.js'
import { TreeTextRow } from './treeRow.js'

// Minimal tool spinner frames
const TOOL_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

function ToolSpinner({ color }: { color: string }) {
  const frame = TOOL_FRAMES[Math.floor(Date.now() / 120) % TOOL_FRAMES.length]
  return <Text color={color}>{frame}</Text>
}

export function ExpandableToolCall({
  tool,
  isExpanded,
  onToggle,
  t,
  details = [],
  branch = 'mid',
  rails = [] as TreeRails,
  hasInlineSubagents = false,
}: {
  tool: ActiveTool
  isExpanded: boolean
  onToggle: () => void
  t: Theme
  details?: Array<{ color: string; content: React.ReactNode; dimColor?: boolean; key: string }>
  branch?: TreeBranch
  rails?: TreeRails
  hasInlineSubagents?: boolean
}) {
  const handleRowClick = (e: React.MouseEvent) => {
    if (e.button !== 0) return
    if (e.shiftKey || e.ctrlKey || e.metaKey) {
      e.stopPropagation()
      onToggle()
    } else {
      e.stopPropagation()
      onToggle()
    }
  }

  const summary = tool.summary ?? formatToolSummary(tool)

  return (
    <Box flexDirection="column">
      {/* Collapsed header row */}
      <TreeTextRow
        branch={branch}
        color={t.color.cornsilk}
        dimColor={false}
        rails={rails}
        t={t}
        content={
          <Box
            onClick={handleRowClick}
            style={{ cursor: 'pointer' }}
          >
            <Text color={t.color.amber}>{isExpanded ? '▾' : '▸'}</Text>
            <Text> </Text>
            <ToolSpinner color={t.color.amber} />
            <Text> {summary}</Text>
            {tool.duration != null ? (
              <Text color={t.color.dim}> ({tool.duration}s)</Text>
            ) : tool.startedAt != null ? (
              <Text color={t.color.dim}> ({fmtElapsed(Date.now() - tool.startedAt)})</Text>
            ) : null}
            {tool.error ? (
              <Text color={t.color.error}> [error]</Text>
            ) : null}
          </Box>
        }
      />

      {/* Expanded details */}
      {isExpanded && details.length > 0 && (
        <Box pl={4} pt={1} pb={1}>
          {details.map((detail, detailIndex) => (
            <TreeTextRow
              branch={detailIndex === details.length - 1 && !hasInlineSubagents ? 'last' : 'mid'}
              color={detail.color}
              dimColor={detail.dimColor}
              rails={[] as TreeRails}
              key={detail.key}
              t={t}
              content={typeof detail.content === 'string' ? (
                <Text dim={detail.dimColor}>{detail.content}</Text>
              ) : (
                detail.content
              )}
            />
          ))}
          {/* Show less action */}
          <Box pt={1}>
            <NoSelect dim>
              <Text color={t.color.amber}>▸</Text>
              <Text> </Text>
              <Text
                color={t.color.amber}
                underline
                onClick={(e: React.MouseEvent) => {
                  e.stopPropagation()
                  onToggle()
                }}
              >
                Show less
              </Text>
            </NoSelect>
          </Box>
        </Box>
      )}
    </Box>
  )
}

function fmtElapsed(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000))
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  const r = s % 60
  return r > 0 ? `${m}m ${r}s` : `${m}m`
}

function formatToolSummary(tool: ActiveTool): string {
  const name = tool.name
  const contextPreview = (tool.context ?? '')
    .split('\n')[0]
    .trim()
    .slice(0, 60)
  return contextPreview ? `${name}(${contextPreview})` : name
}
