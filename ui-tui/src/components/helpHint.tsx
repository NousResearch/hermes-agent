import { Box, Text } from '@hermes/ink'

import { HOTKEYS } from '../content/hotkeys.js'
import type { Theme } from '../theme.js'

const COMMON_COMMANDS: [string, string][] = [
  ['/help', '命令与快捷键完整列表'],
  ['/clear', '开始新会话'],
  ['/resume', '切换或恢复过去的会话'],
  ['/details', '控制对话详细级别'],
  ['/copy', '复制选择内容或上一条助手消息'],
  ['/quit', '退出 Hermes']
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
            ? 快速帮助
          </Text>
          <Text color={t.color.muted}>
            {'  ·  输入 /help 查看完整面板  ·  Backspace 关闭'}
          </Text>
        </Text>

        <Box marginTop={1}>
          <Text bold color={t.color.accent}>
            常用命令
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
            快捷键
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
