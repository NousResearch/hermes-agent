import { Box, Text } from '@hermes/ink'

import { hotkeys } from '../content/hotkeys.js'
import { useT } from '../i18n/useT.js'
import type { Theme } from '../theme.js'

export function HelpHint({ nativeMode = false, t }: { nativeMode?: boolean; t: Theme }) {
  const T = useT()

  const commands: [string, string][] = [
    ['/help', T.help.commands.help],
    ['/clear', T.help.commands.clear],
    ['/resume', T.help.commands.resume],
    ['/details', T.help.commands.details],
    ['/copy', T.help.commands.copy],
    ['/quit', T.help.commands.quit]
  ]

  const hotkeyPreview = hotkeys().slice(0, 8)
  const labelW = Math.max(...commands.map(([k]) => k.length), ...hotkeyPreview.map(([k]) => k.length))

  const pad = (s: string) => s + ' '.repeat(Math.max(0, labelW - s.length + 2))

  return (
    <Box
      alignItems="flex-start"
      {...(nativeMode ? {} : { bottom: '100%', left: 0, position: 'absolute' as const, right: 0 })}
      flexDirection="column"
    >
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
            {T.help.quickHelp}
          </Text>
          <Text color={t.color.muted}>{T.help.quickHelpTail}</Text>
        </Text>

        <Box marginTop={1}>
          <Text bold color={t.color.accent}>
            {T.help.commonCommands}
          </Text>
        </Box>

        {commands.map(([k, v]) => (
          <Text key={k}>
            <Text color={t.color.label}>{pad(k)}</Text>
            <Text color={t.color.muted}>{v}</Text>
          </Text>
        ))}

        <Box marginTop={1}>
          <Text bold color={t.color.accent}>
            {T.help.hotkeys}
          </Text>
        </Box>

        {hotkeyPreview.map(([k, v]) => (
          <Text key={k}>
            <Text color={t.color.label}>{pad(k)}</Text>
            <Text color={t.color.muted}>{v}</Text>
          </Text>
        ))}
      </Box>
    </Box>
  )
}
