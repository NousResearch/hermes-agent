import { Box, Text } from '@hermes/ink'

import { getHotkeys } from '../content/hotkeys.js'
import { useTranslations } from '../i18n/index.js'
import type { Theme } from '../theme.js'

export function HelpHint({ t }: { t: Theme }) {
  const copy = useTranslations()

  const COMMON_COMMANDS: [string, string][] = [
    ['/help', copy.help.full],
    ['/clear', copy.help.clear],
    ['/resume', copy.help.resume],
    ['/details', copy.help.details],
    ['/copy', copy.help.copy],
    ['/quit', copy.help.quit]
  ]

  const HOTKEY_PREVIEW = getHotkeys().slice(0, 8)
  const labelW = Math.max(...COMMON_COMMANDS.map(([k]) => k.length), ...HOTKEY_PREVIEW.map(([k]) => k.length))

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
            {copy.help.title}
          </Text>
          <Text color={t.color.muted}>{copy.help.hint}</Text>
        </Text>

        <Box marginTop={1}>
          <Text bold color={t.color.accent}>
            {copy.help.commands}
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
            {copy.help.hotkeys}
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
