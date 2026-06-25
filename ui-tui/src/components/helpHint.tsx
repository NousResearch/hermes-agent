import { Box, Text, useInput } from '@hermes/ink'
import { useMemo } from 'react'

import { HOTKEYS } from '../content/hotkeys.js'
import type { Theme } from '../theme.js'

const COMMON_COMMANDS: [string, string][] = [
  ['/help', 'full list of commands + hotkeys'],
  ['/clear', 'start a new session'],
  ['/resume', 'switch live or resume past sessions'],
  ['/details', 'control transcript detail level'],
  ['/copy', 'copy selection or last assistant message'],
  ['/statusbar', 'cycle status bar position (top/bottom/off)'],
  ['/theme', 'switch active theme'],
  ['/quit', 'exit hermes']
]

const NAV_HOTKEYS: [string, string][] = [
  ['↑/↓', 'navigate history / session list'],
  ['Tab', 'cycle mode (build/plan/ask/agent)'],
  ['Ctrl+N', 'new session'],
  ['Ctrl+D', 'quit'],
  ['Ctrl+C', 'interrupt (twice to exit)'],
  ['Ctrl+M', 'cycle recent model'],
  ['Ctrl+R', 'recent commands (frecency)'],
  ['F1 / Ctrl+/', 'toggle this help overlay']
]

export function HelpHint({ t }: { t: Theme }) {
  const labelW = Math.max(
    ...COMMON_COMMANDS.map(([k]) => k.length),
    ...NAV_HOTKEYS.slice(0, 4).map(([k]) => k.length)
  )

  const pad = (s: string) => s + ' '.repeat(Math.max(0, labelW - s.length + 2))

  return (
    <Box alignItems="flex-start" bottom="100%" flexDirection="column" left={0} position="absolute" right={0}>
      <Box
        alignSelf="flex-start"
        borderColor={t.color.primary}
        borderStyle="round"
        flexDirection="column"
        marginBottom={1}
        opaque
        paddingX={1}
      >
        <Text>
          <Text bold color={t.color.primary}>
            ? quick help
          </Text>
          <Text color={t.color.muted}>
            {'  ·  F1 for the full panel  ·  backspace to dismiss'}
          </Text>
        </Text>

        <Box marginTop={1}>
          <Text bold color={t.color.accent}>
            Common commands
          </Text>
        </Box>

        {COMMON_COMMANDS.map(([k, v]) => (
          <Text key={k}>
            <Text color={t.color.label}>{pad(k)}</Text>
            <Text color={t.color.muted}>{v}</Text>
          </Text>
        ))}

        <Box marginTop={1}>
          <Text bold color={t.color.accent}>
            Hotkeys
          </Text>
        </Box>

        {NAV_HOTKEYS.slice(0, 4).map(([k, v]) => (
          <Text key={k}>
            <Text color={t.color.label}>{pad(k)}</Text>
            <Text color={t.color.muted}>{v}</Text>
          </Text>
        ))}
      </Box>
    </Box>
  )
}

// Full-screen help overlay. Always available via F1 / Ctrl+/, never
// requires the user to type ? in the input. Critical for TDAH: discoverable
// without remembering slash commands. Esc closes.
export function HelpOverlay({ t, onClose }: { t: Theme; onClose: () => void }) {
  useInput((_input, key) => {
    if (key.escape) {
      onClose()
    }
  })

  // Split the full HOTKEYS list into rough sections so the panel reads
  // top-to-bottom: navigation, sessions, model/agent, then everything else.
  const sections = useMemo(() => {
    const nav = HOTKEYS.filter(k => /arrow|tab|esc|enter|backspace/i.test(k[0]))
    const sessions = HOTKEYS.filter(k => /ctrl\+[nd]/i.test(k[0]))
    const model = HOTKEYS.filter(k => /ctrl\+m|model|f[1-9]/i.test(k[0]))

    const other = HOTKEYS.filter(
      k => !nav.includes(k) && !sessions.includes(k) && !model.includes(k)
    )

    return [
      { title: 'Navigation', items: nav },
      { title: 'Sessions', items: sessions },
      { title: 'Model / Agent', items: model },
      { title: 'Other', items: other }
    ]
  }, [])

  const labelW = Math.max(
    ...HOTKEYS.map(([k]) => k.length),
    ...COMMON_COMMANDS.map(([k]) => k.length)
  )

  const pad = (s: string) => s + ' '.repeat(Math.max(0, labelW - s.length + 2))

  return (
    <Box
      alignItems="center"
      bottom={0}
      flexDirection="column"
      justifyContent="center"
      left={0}
      position="absolute"
      right={0}
      top={0}
    >
      <Box
        borderColor={t.color.primary}
        borderStyle="round"
        flexDirection="column"
        maxHeight="80%"
        maxWidth="80%"
        opaque
        paddingX={2}
        paddingY={1}
      >
        <Text>
          <Text bold color={t.color.primary}>
            ? hermes tui
          </Text>
          <Text color={t.color.muted}>
            {'  ·  F1 / Ctrl+/ to toggle  ·  Esc to close'}
          </Text>
        </Text>

        <Box marginTop={1}>
          <Text bold color={t.color.accent}>
            Common commands
          </Text>
        </Box>

        {COMMON_COMMANDS.map(([k, v]) => (
          <Text key={k}>
            <Text color={t.color.label}>{pad(k)}</Text>
            <Text color={t.color.muted}>{v}</Text>
          </Text>
        ))}

        {sections.map(({ title, items }) =>
          items.length > 0 ? (
            <Box flexDirection="column" key={title} marginTop={1}>
              <Text bold color={t.color.accent}>
                {title}
              </Text>
              {items.map(([k, v]) => (
                <Text key={k}>
                  <Text color={t.color.label}>{pad(k)}</Text>
                  <Text color={t.color.muted}>{v}</Text>
                </Text>
              ))}
            </Box>
          ) : null
        )}
      </Box>
    </Box>
  )
}
