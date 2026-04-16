import { Box, Text } from '@hermes/ink'
import { useStore } from '@nanostores/react'

import { useGateway } from '../app/gatewayContext.js'
import type { AppOverlaysProps } from '../app/interfaces.js'
import { $overlayState, patchOverlayState } from '../app/overlayStore.js'
import { $uiState } from '../app/uiStore.js'

import { FloatBox } from './appChrome.js'
import { MaskedPrompt } from './maskedPrompt.js'
import { ModelPicker } from './modelPicker.js'
import { ApprovalPrompt, ClarifyPrompt } from './prompts.js'
import { SessionPicker } from './sessionPicker.js'

export function AppOverlays({
  cols,
  compIdx,
  completions,
  onApprovalChoice,
  onClarifyAnswer,
  onModelSelect,
  onPickerSelect,
  onSecretSubmit,
  onSudoSubmit,
  pagerPageSize
}: AppOverlaysProps) {
  const { gw } = useGateway()
  const overlay = useStore($overlayState)
  const ui = useStore($uiState)

  if (
    !(
      overlay.approval ||
      overlay.clarify ||
      overlay.modelPicker ||
      overlay.pager ||
      overlay.picker ||
      overlay.secret ||
      overlay.sudo ||
      completions.length
    )
  ) {
    return null
  }

  const start = Math.max(0, compIdx - 8)

  return (
    <Box alignItems="flex-start" bottom="100%" flexDirection="column" left={0} position="absolute" right={0}>
      {overlay.clarify && (
        <FloatBox color={ui.theme.color.bronze}>
          <ClarifyPrompt
            cols={cols}
            onAnswer={onClarifyAnswer}
            onCancel={() => onClarifyAnswer('')}
            req={overlay.clarify}
            t={ui.theme}
          />
        </FloatBox>
      )}

      {overlay.approval && (
        <FloatBox color={ui.theme.color.bronze}>
          <ApprovalPrompt onChoice={onApprovalChoice} req={overlay.approval} t={ui.theme} />
        </FloatBox>
      )}

      {overlay.sudo && (
        <FloatBox color={ui.theme.color.bronze}>
          <MaskedPrompt cols={cols} icon="🔐" label="sudo password required" onSubmit={onSudoSubmit} t={ui.theme} />
        </FloatBox>
      )}

      {overlay.secret && (
        <FloatBox color={ui.theme.color.bronze}>
          <MaskedPrompt
            cols={cols}
            icon="🔑"
            label={overlay.secret.prompt}
            onSubmit={onSecretSubmit}
            sub={`for ${overlay.secret.envVar}`}
            t={ui.theme}
          />
        </FloatBox>
      )}

      {overlay.picker && (
        <FloatBox color={ui.theme.color.bronze}>
          <SessionPicker
            gw={gw}
            onCancel={() => patchOverlayState({ picker: false })}
            onSelect={onPickerSelect}
            t={ui.theme}
          />
        </FloatBox>
      )}

      {overlay.modelPicker && (
        <FloatBox color={ui.theme.color.bronze}>
          <ModelPicker
            gw={gw}
            onCancel={() => patchOverlayState({ modelPicker: false })}
            onSelect={onModelSelect}
            sessionId={ui.sid}
            t={ui.theme}
          />
        </FloatBox>
      )}

      {overlay.pager && (
        <FloatBox color={ui.theme.color.bronze}>
          <Box flexDirection="column" paddingX={1} paddingY={1}>
            {overlay.pager.title && (
              <Box justifyContent="center" marginBottom={1}>
                <Text bold color={ui.theme.color.gold}>
                  {overlay.pager.title}
                </Text>
              </Box>
            )}

            {overlay.pager.lines.slice(overlay.pager.offset, overlay.pager.offset + pagerPageSize).map((line, i) => (
              <Text key={i}>{line}</Text>
            ))}

            <Box marginTop={1}>
              <Text color={ui.theme.color.dim}>
                {overlay.pager.offset + pagerPageSize < overlay.pager.lines.length
                  ? `Enter/Space for more · q to close (${Math.min(overlay.pager.offset + pagerPageSize, overlay.pager.lines.length)}/${overlay.pager.lines.length})`
                  : `end · q to close (${overlay.pager.lines.length} lines)`}
              </Text>
            </Box>
          </Box>
        </FloatBox>
      )}

      {!!completions.length && (
        <FloatBox color={ui.theme.color.gold}>
          <Box flexDirection="column" width={Math.max(28, cols - 6)}>
            {completions.slice(start, compIdx + 8).map((item, i) => {
              const active = start + i === compIdx

              return (
                <Box
                  backgroundColor={active ? ui.theme.color.completionCurrentBg : undefined}
                  flexDirection="row"
                  key={`${start + i}:${item.text}:${item.display}:${item.meta ?? ''}`}
                  width="100%"
                >
                  <Text bold={active} color={ui.theme.color.bronze}>
                    {' '}
                    {item.display}
                  </Text>
                  {item.meta ? <Text color={ui.theme.color.dim}> {item.meta}</Text> : null}
                </Box>
              )
            })}
          </Box>
        </FloatBox>
      )}
    </Box>
  )
}
