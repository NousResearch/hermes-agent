import { Box, Text } from '@hermes/ink'

import { HOTKEYS } from '../content/hotkeys.js'
import type { Theme } from '../theme.js'

const COMMON_COMMANDS: [string, string][] = [
  ['/help', 'full list of commands + hotkeys'],
  ['/clear', 'start a new session'],
  ['/resume', 'resume a prior session'],
  ['/details', 'control transcript detail level'],
  ['/copy', 'copy selection or last assistant message'],
  ['/quit', 'exit hermes']
]

const HOTKEY_PREVIEW = HOTKEYS.slice(0, 8)

export function HelpHint({ t }: { t: Theme }) {
  const labelW = Math.max(
    ...COMMON_COMMANDS.map(([k]) => k.length),
    ...HOTKEY_PREVIEW.map(([k]) => k.length)
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
            {'  ·  type /help for the full panel  ·  backspace to dismiss'}
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

        {HOTKEY_PREVIEW.map(([k, v]) => (
          <Text key={k}>
            <Text color={t.color.label}>{pad(k)}</Text>
            <Text color={t.color.muted}>{v}</Text>
          </Text>
        ))}
      </Box>
    </Box>
  )
}
