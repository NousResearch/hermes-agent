import { Box, Text, useStdout } from 'ink'

import { caduceus, logo, LOGO_WIDTH } from '../banner.js'
import { flat } from '../lib/text.js'
import type { Theme } from '../theme.js'
import type { SessionInfo } from '../types.js'

export function ArtLines({ lines }: { lines: [string, string][] }) {
  return (
    <>
      {lines.map(([c, text], i) => (
        <Text color={c} key={i}>
          {text}
        </Text>
      ))}
    </>
  )
}

export function Banner({ t }: { t: Theme }) {
  const cols = useStdout().stdout?.columns ?? 80

  return (
    <Box flexDirection="column" marginBottom={1}>
      {cols >= LOGO_WIDTH ? (
        <ArtLines lines={logo(t.color)} />
      ) : (
        <Text bold color={t.color.gold}>
          {t.brand.icon} NOUS HERMES
        </Text>
      )}
      <Text color={t.color.dim}>{t.brand.icon} Nous Research · Messenger of the Digital Gods</Text>
    </Box>
  )
}

export function SessionPanel({ info, sid, t }: { info: SessionInfo; sid?: string | null; t: Theme }) {
  const cols = useStdout().stdout?.columns ?? 100
  const wide = cols >= 90
  const leftW = wide ? 34 : 0
  const w = wide ? cols - leftW - 12 : cols - 10
  const cwd = info.cwd || process.cwd()
  const strip = (s: string) => (s.endsWith('_tools') ? s.slice(0, -6) : s)
  const title = `${t.brand.name}${info.version ? ` v${info.version}` : ''}${info.release_date ? ` (${info.release_date})` : ''}`

  const truncLine = (pfx: string, items: string[]) => {
    let line = ''

    for (const item of items.sort()) {
      const next = line ? `${line}, ${item}` : item

      if (pfx.length + next.length > w) {
        return line ? `${line}, …+${items.length - line.split(', ').length}` : `${item}, …`
      }

      line = next
    }

    return line
  }

  const section = (title: string, data: Record<string, string[]>, max = 8, overflowLabel = 'more…') => {
    const entries = Object.entries(data).sort()
    const shown = entries.slice(0, max)
    const overflow = entries.length - max

    return (
      <Box flexDirection="column" marginTop={1}>
        <Text bold color={t.color.amber}>
          Available {title}
        </Text>
        {shown.map(([k, vs]) => (
          <Text key={k} wrap="truncate">
            <Text color={t.color.dim}>{strip(k)}: </Text>
            <Text color={t.color.cornsilk}>{truncLine(strip(k) + ': ', vs)}</Text>
          </Text>
        ))}
        {overflow > 0 && <Text color={t.color.dim}>(and {overflow} {overflowLabel})</Text>}
      </Box>
    )
  }

  return (
    <Box borderColor={t.color.bronze} borderStyle="round" marginBottom={1} paddingX={2} paddingY={1}>
      {wide && (
        <Box flexDirection="column" marginRight={2} width={leftW}>
          <ArtLines lines={caduceus(t.color)} />
          <Text />
          <Text color={t.color.amber}>
            {info.model.split('/').pop()}
            <Text color={t.color.dim}> · Nous Research</Text>
          </Text>
          <Text color={t.color.dim} wrap="truncate-end">{cwd}</Text>
          {sid && <Text color={t.color.dim}>Session: {sid}</Text>}
        </Box>
      )}
      <Box flexDirection="column" width={w}>
        <Box justifyContent="center" marginBottom={1}>
          <Text bold color={t.color.gold}>{title}</Text>
        </Box>
        {section('Tools', info.tools, 8, 'more toolsets…')}
        {section('Skills', info.skills)}
        <Text />
        <Text color={t.color.cornsilk}>
          {flat(info.tools).length} tools{' · '}
          {flat(info.skills).length} skills
          {' · '}
          <Text color={t.color.dim}>/help for commands</Text>
        </Text>
        {typeof info.update_behind === 'number' && info.update_behind > 0 && (
          <Text bold color="yellow">
            ⚠ {info.update_behind} {info.update_behind === 1 ? 'commit' : 'commits'} behind
            <Text bold={false} color="yellow" dimColor> — run </Text>
            <Text bold color="yellow">{info.update_command || 'hermes update'}</Text>
            <Text bold={false} color="yellow" dimColor> to update</Text>
          </Text>
        )}
      </Box>
    </Box>
  )
}
