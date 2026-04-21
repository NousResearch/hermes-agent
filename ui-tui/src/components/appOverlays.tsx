import { Box, Text } from '@hermes/ink'
import { useStore } from '@nanostores/react'

import { useGateway } from '../app/gatewayContext.js'
import type { AppOverlaysProps } from '../app/interfaces.js'
import { $overlayState, patchOverlayState } from '../app/overlayStore.js'
import { $uiState } from '../app/uiStore.js'

import { FloatBox } from './appChrome.js'
import { MaskedPrompt } from './maskedPrompt.js'
import { ModelPicker } from './modelPicker.js'
import { ApprovalPrompt, ClarifyPrompt, ConfirmPrompt } from './prompts.js'
import { SessionPicker } from './sessionPicker.js'
import { SkillsHub } from './skillsHub.js'

const COMPLETION_WINDOW = 16

export function PromptZone({
  cols,
  onApprovalChoice,
  onClarifyAnswer,
  onSecretSubmit,
  onSudoSubmit
}: Pick<AppOverlaysProps, 'cols' | 'onApprovalChoice' | 'onClarifyAnswer' | 'onSecretSubmit' | 'onSudoSubmit'>) {
  const overlay = useStore($overlayState)
  const ui = useStore($uiState)

  if (overlay.approval) {
    return (
      <Box flexDirection="column" flexShrink={0} paddingX={1} paddingY={1}>
        <ApprovalPrompt onChoice={onApprovalChoice} req={overlay.approval} t={ui.theme} />
      </Box>
    )
  }

  if (overlay.confirm) {
    const req = overlay.confirm

    const onConfirm = () => {
      patchOverlayState({ confirm: null })
      req.onConfirm()
    }

    const onCancel = () => patchOverlayState({ confirm: null })

    return (
      <Box flexDirection="column" flexShrink={0} paddingX={1} paddingY={1}>
        <ConfirmPrompt onCancel={onCancel} onConfirm={onConfirm} req={req} t={ui.theme} />
      </Box>
    )
  }

  if (overlay.clarify) {
    return (
      <Box flexDirection="column" flexShrink={0} paddingX={1} paddingY={1}>
        <ClarifyPrompt
          cols={cols}
          onAnswer={onClarifyAnswer}
          onCancel={() => onClarifyAnswer('')}
          req={overlay.clarify}
          t={ui.theme}
        />
      </Box>
    )
  }

  if (overlay.sudo) {
    return (
      <Box flexDirection="column" flexShrink={0} paddingX={1} paddingY={1}>
        <MaskedPrompt cols={cols} icon="🔐" label="sudo password required" onSubmit={onSudoSubmit} t={ui.theme} />
      </Box>
    )
  }

  if (overlay.secret) {
    return (
      <Box flexDirection="column" flexShrink={0} paddingX={1} paddingY={1}>
        <MaskedPrompt
          cols={cols}
          icon="🔑"
          label={overlay.secret.prompt}
          onSubmit={onSecretSubmit}
          sub={`for ${overlay.secret.envVar}`}
          t={ui.theme}
        />
      </Box>
    )
  }

  return null
}

export function FloatingOverlays({
  cols,
  compIdx,
  completions,
  onModelSelect,
  onPickerSelect,
  pagerPageSize
}: Pick<AppOverlaysProps, 'cols' | 'compIdx' | 'completions' | 'onModelSelect' | 'onPickerSelect' | 'pagerPageSize'>) {
  const { gw } = useGateway()
  const overlay = useStore($overlayState)
  const ui = useStore($uiState)

  const hasAny = overlay.modelPicker || overlay.pager || overlay.picker || overlay.skillsHub || completions.length

  if (!hasAny) {
    return null
  }

  // Fixed viewport centered on compIdx — previously the slice end was
  // compIdx + 8 so the dropdown grew from 8 rows to 16 as the user scrolled
  // down, bouncing the height on every keystroke.
  const viewportSize = Math.min(COMPLETION_WINDOW, completions.length)

  const start = Math.max(0, Math.min(compIdx - Math.floor(COMPLETION_WINDOW / 2), completions.length - viewportSize))

  return (
    <Box alignItems="flex-start" bottom="100%" flexDirection="column" left={0} position="absolute" right={0}>
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

      {overlay.skillsHub && (
        <FloatBox color={ui.theme.color.bronze}>
          <SkillsHub gw={gw} onClose={() => patchOverlayState({ skillsHub: false })} t={ui.theme} />
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
            {completions.slice(start, start + viewportSize).map((item, i) => {
              const active = start + i === compIdx

              return (
                <Box
                  backgroundColor={active ? ui.theme.color.completionCurrentBg : undefined}
                  flexDirection="row"
                  key={`${start + i}:${item.text}:${item.display}:${item.meta ?? ''}`}
                  width="100%"
                >
                  <Text bold color={ui.theme.color.label}>
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
