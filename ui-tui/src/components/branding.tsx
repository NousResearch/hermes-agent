import { Box, Text, useStdout } from '@hermes/ink'

import { artWidth, caduceus, CADUCEUS_WIDTH, logo, LOGO_WIDTH } from '../banner.js'
import { flat } from '../lib/text.js'
import type { Theme } from '../theme.js'
import type { PanelSection, SessionInfo } from '../types.js'

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
  const logoLines = logo(t.color, t.bannerLogo || undefined)

  return (
    <Box flexDirection="column" marginBottom={1}>
      {cols >= (t.bannerLogo ? artWidth(logoLines) : LOGO_WIDTH) ? (
        <ArtLines lines={logoLines} />
      ) : (
        <Text bold color={t.color.primary}>
          {t.brand.icon} NOUS HERMES
        </Text>
      )}

      <Text color={t.color.muted}>{t.brand.icon} Nous Research · Messenger of the Digital Gods</Text>
    </Box>
  )
}

export function SessionPanel({ info, sid, t }: SessionPanelProps) {
  const cols = useStdout().stdout?.columns ?? 100
  const heroLines = caduceus(t.color, t.bannerHero || undefined)
  const leftW = Math.min((artWidth(heroLines) || CADUCEUS_WIDTH) + 4, Math.floor(cols * 0.4))
  const wide = cols >= 90 && leftW + 40 < cols
  const w = Math.max(20, wide ? cols - leftW - 14 : cols - 12)
  const lineBudget = Math.max(12, w - 2)
  const strip = (s: string) => (s.endsWith('_tools') ? s.slice(0, -6) : s)

  const truncLine = (pfx: string, items: string[]) => {
    let line = ''
    let shown = 0

    for (const item of [...items].sort()) {
      const next = line ? `${line}, ${item}` : item

      if (pfx.length + next.length > lineBudget) {
        return line ? `${line}, …+${items.length - shown}` : `${item}, …`
      }

      line = next
      shown++
    }

    return line
  }

  const section = (title: string, data: Record<string, string[]>, max = 8, overflowLabel = 'more…') => {
    const entries = Object.entries(data).sort()
    const shown = entries.slice(0, max)
    const overflow = entries.length - max

    return (
      <Box flexDirection="column" marginTop={1}>
        <Text bold color={t.color.accent}>
          Available {title}
        </Text>

        {shown.map(([k, vs]) => (
          <Text key={k} wrap="truncate">
            <Text color={t.color.muted}>{strip(k)}: </Text>
            <Text color={t.color.text}>{truncLine(strip(k) + ': ', vs)}</Text>
          </Text>
        ))}

        {overflow > 0 && (
          <Text color={t.color.muted}>
            (and {overflow} {overflowLabel})
          </Text>
        )}
      </Box>
    )
  }

  return (
    <Box borderColor={t.color.border} borderStyle="round" marginBottom={1} paddingX={2} paddingY={1}>
      {wide && (
        <Box flexDirection="column" marginRight={2} width={leftW}>
          <ArtLines lines={heroLines} />
          <Text />

          <Text color={t.color.accent}>
            {info.model.split('/').pop()}
            <Text color={t.color.muted}> · Nous Research</Text>
          </Text>

          <Text color={t.color.muted} wrap="truncate-end">
            {info.cwd || process.cwd()}
          </Text>

          {sid && (
            <Text>
              <Text color={t.color.sessionLabel}>Session: </Text>
              <Text color={t.color.sessionBorder}>{sid}</Text>
            </Text>
          )}
        </Box>
      )}

      <Box flexDirection="column" width={w}>
        <Box justifyContent="center" marginBottom={1}>
          <Text bold color={t.color.primary}>
            {t.brand.name}
            {info.version ? ` v${info.version}` : ''}
            {info.release_date ? ` (${info.release_date})` : ''}
          </Text>
        </Box>

        {section('Tools', info.tools, 8, 'more toolsets…')}
        {section('Skills', info.skills)}

        {info.mcp_servers && info.mcp_servers.length > 0 && (
          <Box flexDirection="column" marginTop={1}>
            <Text bold color={t.color.accent}>
              MCP Servers
            </Text>

            {info.mcp_servers.map(s => (
              <Text key={s.name} wrap="truncate">
                <Text color={t.color.muted}>{`  ${s.name} `}</Text>
                <Text color={t.color.muted}>{`[${s.transport}]`}</Text>
                <Text color={t.color.muted}>: </Text>
                {s.connected ? (
                  <Text color={t.color.text}>
                    {s.tools} tool{s.tools === 1 ? '' : 's'}
                  </Text>
                ) : (
                  <Text color={t.color.error}>failed</Text>
                )}
              </Text>
            ))}
          </Box>
        )}

        <Text />

        <Text color={t.color.text}>
          {flat(info.tools).length} tools{' · '}
          {flat(info.skills).length} skills
          {info.mcp_servers?.length ? ` · ${info.mcp_servers.length} MCP` : ''}
          {' · '}
          <Text color={t.color.muted}>/help for commands</Text>
        </Text>

        {typeof info.update_behind === 'number' && info.update_behind > 0 && (
          <Text bold color={t.color.warn}>
            ! {info.update_behind} {info.update_behind === 1 ? 'commit' : 'commits'} behind
            <Text bold={false} color={t.color.warn} dimColor>
              {' '}
              - run{' '}
            </Text>
            <Text bold color={t.color.warn}>
              {info.update_command || 'hermes update'}
            </Text>
            <Text bold={false} color={t.color.warn} dimColor>
              {' '}
              to update
            </Text>
          </Text>
        )}
      </Box>
    </Box>
  )
}

export function Panel({ sections, t, title }: PanelProps) {
  return (
    <Box borderColor={t.color.border} borderStyle="round" flexDirection="column" paddingX={2} paddingY={1}>
      <Box justifyContent="center" marginBottom={1}>
        <Text bold color={t.color.primary}>
          {title}
        </Text>
      </Box>

      {sections.map((sec, si) => (
        <Box flexDirection="column" key={si} marginTop={si > 0 ? 1 : 0}>
          {sec.title && (
            <Text bold color={t.color.accent}>
              {sec.title}
            </Text>
          )}

          {sec.rows?.map(([k, v], ri) => (
            <Text key={ri} wrap="truncate">
              <Text color={t.color.muted}>{k.padEnd(20)}</Text>
              <Text color={t.color.text}>{v}</Text>
            </Text>
          ))}

          {sec.items?.map((item, ii) => (
            <Text color={t.color.text} key={ii} wrap="truncate">
              {item}
            </Text>
          ))}

          {sec.text && <Text color={t.color.muted}>{sec.text}</Text>}
        </Box>
      ))}
    </Box>
  )
}

interface PanelProps {
  sections: PanelSection[]
  t: Theme
  title: string
}

interface SessionPanelProps {
  info: SessionInfo
  sid?: string | null
  t: Theme
}
