import { Box, Text } from '@hermes/ink'

import { HOTKEYS } from '../content/hotkeys.js'
import { useI18n, type TranslationKey } from '../i18n/index.js'
import type { Theme } from '../theme.js'

const COMMON_COMMAND_KEYS: [string, TranslationKey][] = [
  ['/help', 'help.fullList'],
  ['/clear', 'help.newSession'],
  ['/resume', 'help.resumeSession'],
  ['/details', 'help.detailsDesc'],
  ['/copy', 'help.copyDesc'],
  ['/quit', 'help.exitDesc']
]

const HOTKEY_PREVIEW = HOTKEYS.slice(0, 8)

export function HelpHint({ t }: { t: Theme }) {
  const { t: ti } = useI18n()
  const COMMON_COMMANDS: [string, string][] = COMMON_COMMAND_KEYS.map(([k, key]) => [k, ti(key)])
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
            {ti('help.quickHelp')}
          </Text>
          <Text color={t.color.muted}>
            {ti('help.dismissHint')}
          </Text>
        </Text>

        <Box marginTop={1}>
          <Text bold color={t.color.accent}>
            {ti('help.commonCommands')}
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
            {ti('help.hotkeys')}
          </Text>
        </Box>

        {HOTKEY_PREVIEW.map(([k, key]) => (
          <Text key={k}>
            <Text color={t.color.label}>{pad(k)}</Text>
            <Text color={t.color.muted}>{ti(key as TranslationKey)}</Text>
          </Text>
        ))}
      </Box>
    </Box>
  )
}
