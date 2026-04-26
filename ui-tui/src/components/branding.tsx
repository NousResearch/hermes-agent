import { Box, Text, useStdout } from '@hermes/ink'

import { artWidth, logo, LOGO_WIDTH } from '../banner.js'
import { flat } from '../lib/text.js'
import type { Theme } from '../theme.js'
import type { PanelSection, SessionInfo } from '../types.js'

export const MOTTO = 'The bridge remembers. The engine dreams.'

const AURORA_IDENTITY = 'Aurora Proto'

type LaunchTone = 'accent' | 'dim' | 'muted' | 'plain' | 'title' | 'warn'

export interface LaunchCardLine {
  text: string
  tone?: LaunchTone
}

export interface LaunchCardModel {
  lines: LaunchCardLine[]
  mode: 'launch' | 'resume'
}

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
        <Text bold color={t.color.gold}>
          ░▒▓ AURORA PROTO ▓▒░
        </Text>
      )}

      <Text color={t.color.dim}>{t.brand.icon} {MOTTO}</Text>
    </Box>
  )
}

export function buildLaunchCardModel(info: SessionInfo, sid?: null | string): LaunchCardModel {
  const toolsets = Object.keys(info.tools ?? {}).length
  const skillRings = Object.keys(info.skills ?? {}).length
  const tools = flat(info.tools ?? {}).length
  const skills = flat(info.skills ?? {}).length
  const mcp = info.mcp_servers ?? []
  const onlineMcp = mcp.filter(server => server.connected).length
  const cwd = info.cwd || process.cwd()
  const model = info.model.split('/').pop() || info.model
  const mode: LaunchCardModel['mode'] = info.resumed_session_id ? 'resume' : 'launch'

  const lines: LaunchCardLine[] = [
    { text: `☤ ${AURORA_IDENTITY}`, tone: 'title' },
    { text: MOTTO, tone: 'dim' },
    { text: '' },
    { text: `${model}${info.version ? ` · v${info.version}` : ''}${info.release_date ? ` · ${info.release_date}` : ''}`, tone: 'accent' },
    { text: `▣ workspace  ${cwd}`, tone: 'plain' }
  ]

  if (sid) {
    lines.push({ text: `◇ session    ${sid}`, tone: 'muted' })
  }

  lines.push({ text: '' })

  if (mode === 'resume') {
    lines.push(
      { text: 'THREAD RE-ENTRY', tone: 'accent' },
      { text: `◇ trace      ${info.resumed_session_id}`, tone: 'plain' },
      {
        text: `⟁ restored   ${info.resume_message_count ?? 0} messages restored`,
        tone: 'plain'
      },
      { text: '→ continue from the last living edge', tone: 'title' },
      { text: '' }
    )
  } else {
    lines.push(
      { text: 'LAUNCH SEQUENCE', tone: 'accent' },
      { text: '◐ cognition  online', tone: 'plain' },
      { text: '◇ memory     listening', tone: 'plain' },
      { text: '' }
    )
  }

  lines.push(
    { text: 'AURORA BODY', tone: 'accent' },
    { text: `⚙ tools      ${tools} indexed across ${toolsets} toolsets`, tone: 'plain' },
    { text: `✦ skills     ${skills} procedures across ${skillRings} rings`, tone: 'plain' },
    {
      text: `⌁ mcp        ${onlineMcp}/${mcp.length} servers linked`,
      tone: mcp.length && onlineMcp < mcp.length ? 'warn' : 'plain'
    },
    { text: '☉ hands      terminal · file · browser', tone: 'plain' },
    { text: '◇ memory     chorus · sessions · skills', tone: 'plain' },
    { text: '' },
    { text: 'MISSION', tone: 'accent' },
    { text: 'Build the cockpit we actually want to live inside.', tone: 'plain' },
    { text: '' },
    { text: mode === 'resume' ? '/palette command bridge · /status body · /resume thread' : '/palette command bridge · /tools inventory · /skills shelf', tone: 'dim' }
  )

  if (typeof info.update_behind === 'number' && info.update_behind > 0) {
    lines.push({ text: `! ${info.update_behind} commits behind · ${info.update_command || 'hermes update'}`, tone: 'warn' })
  }

  return { lines, mode }
}

const launchColor = (t: Theme, tone: LaunchTone = 'plain') => {
  switch (tone) {
    case 'accent':
      return t.color.amber

    case 'dim':
      return t.color.dim

    case 'muted':
      return t.color.sessionBorder

    case 'title':
      return t.color.gold

    case 'warn':
      return t.color.warn

    case 'plain':

    default:
      return t.color.cornsilk
  }
}

export function SessionPanel({ info, sid, t }: SessionPanelProps) {
  const cols = useStdout().stdout?.columns ?? 100
  const w = Math.max(20, cols - 12)
  const model = buildLaunchCardModel(info, sid)

  return (
    <Box borderColor={t.color.bronze} borderStyle="round" marginBottom={1} paddingX={2} paddingY={1}>
      <Box flexDirection="column" width={w}>
        {model.lines.map((line, i) =>
          line.text ? (
            <Text bold={line.tone === 'title' || line.tone === 'accent'} color={launchColor(t, line.tone)} key={i} wrap="truncate">
              {line.text}
            </Text>
          ) : (
            <Text key={i}> </Text>
          )
        )}
      </Box>
    </Box>
  )
}

export function Panel({ sections, t, title }: PanelProps) {
  return (
    <Box borderColor={t.color.bronze} borderStyle="round" flexDirection="column" paddingX={2} paddingY={1}>
      <Box justifyContent="center" marginBottom={1}>
        <Text bold color={t.color.gold}>
          {title}
        </Text>
      </Box>

      {sections.map((sec, si) => (
        <Box flexDirection="column" key={si} marginTop={si > 0 ? 1 : 0}>
          {sec.title && (
            <Text bold color={t.color.amber}>
              {sec.title}
            </Text>
          )}

          {sec.rows?.map(([k, v], ri) => (
            <Text key={ri} wrap="truncate">
              <Text color={t.color.dim}>{k.padEnd(20)}</Text>
              <Text color={t.color.cornsilk}>{v}</Text>
            </Text>
          ))}

          {sec.items?.map((item, ii) => (
            <Text color={t.color.cornsilk} key={ii} wrap="truncate">
              {item}
            </Text>
          ))}

          {sec.text && <Text color={t.color.dim}>{sec.text}</Text>}
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
