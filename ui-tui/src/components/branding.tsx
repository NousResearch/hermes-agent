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
      <Text />
      <Text>
        <Text color={t.color.amber}>{t.brand.icon} Nous Research</Text>
        <Text color={t.color.dim}> · Messenger of the Digital Gods</Text>
      </Text>
    </Box>
  )
}

export function SessionPanel({ info, t }: { info: SessionInfo; t: Theme }) {
  const cols = useStdout().stdout?.columns ?? 100
  const wide = cols >= 90
  const w = wide ? cols - 46 : cols - 10
  const strip = (s: string) => (s.endsWith('_tools') ? s.slice(0, -6) : s)

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

  const section = (title: string, data: Record<string, string[]>, max = 8) => {
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
        {overflow > 0 && <Text color={t.color.dim}>(and {overflow} more…)</Text>}
      </Box>
    )
  }

  return (
    <Box borderColor={t.color.bronze} borderStyle="round" marginBottom={1} paddingX={2} paddingY={1}>
      {wide && (
        <Box flexDirection="column" marginRight={2} width={34}>
          <ArtLines lines={caduceus(t.color)} />
          <Text />
          <Text color={t.color.dim}>Nous Research</Text>
        </Box>
      )}
      <Box flexDirection="column" width={w}>
        <Text bold color={t.color.gold}>
          {t.brand.icon} {t.brand.name}
        </Text>
        {section('Tools', info.tools)}
        {section('Skills', info.skills)}
        <Text />
        <Text color={t.color.cornsilk}>
          {flat(info.tools).length} tools{' · '}
          {flat(info.skills).length} skills
          {' · '}
          <Text color={t.color.dim}>/help for commands</Text>
        </Text>
        <Text color={t.color.dim}>
          {info.model.split('/').pop()}
          {' · '}Ctrl+C to interrupt
        </Text>
      </Box>
    </Box>
  )
}
