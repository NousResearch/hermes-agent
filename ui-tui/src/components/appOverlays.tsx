import { Box, Text } from '@hermes/ink'
import { useStore } from '@nanostores/react'

import { useGateway } from '../app/gatewayContext.js'
import type { AppOverlaysProps } from '../app/interfaces.js'
import { $overlayState, patchOverlayState } from '../app/overlayStore.js'
import { $uiSessionId, $uiTheme, $uiState } from '../app/uiStore.js'

import { ActiveSessionSwitcher } from './activeSessionSwitcher.js'
import { FloatBox } from './appChrome.js'
import { BillingOverlay } from './billingOverlay.js'
import { MaskedPrompt } from './maskedPrompt.js'
import { ModelPicker } from './modelPicker.js'
import { OverlayHint } from './overlayControls.js'
import { PluginsHub } from './pluginsHub.js'
import { ApprovalPrompt, ClarifyPrompt, ConfirmPrompt } from './prompts.js'
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
  const theme = useStore($uiTheme)

  if (overlay.approval) {
    return (
      <Box flexDirection="column" flexShrink={0} paddingX={1} paddingY={1}>
        <ApprovalPrompt cols={cols} onChoice={onApprovalChoice} req={overlay.approval} t={theme} />
      </Box>
    )
  }

  if (overlay.billing) {
    const current = overlay.billing

    const onPatch = (next: Partial<typeof current>) =>
      patchOverlayState(prev => (prev.billing ? { ...prev, billing: { ...prev.billing, ...next } } : prev))

    const onClose = () => patchOverlayState({ billing: null })

    return (
      <Box flexDirection="column" flexShrink={0} paddingX={1} paddingY={1}>
        <BillingOverlay onClose={onClose} onPatch={onPatch} overlay={current} t={theme} />
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
        <ConfirmPrompt onCancel={onCancel} onConfirm={onConfirm} req={req} t={theme} />
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
          t={theme}
        />
      </Box>
    )
  }

  if (overlay.sudo) {
    return (
      <Box flexDirection="column" flexShrink={0} paddingX={1} paddingY={1}>
        <MaskedPrompt cols={cols} icon="🔐" label="sudo password required" onSubmit={onSudoSubmit} t={theme} />
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
          t={theme}
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
  onActiveSessionSelect,
  onActiveSessionClose,
  onModelSelect,
  onNewLiveSession,
  onNewPromptSession,
  onResumeSelect,
  pagerPageSize
}: Pick<
  AppOverlaysProps,
  | 'cols'
  | 'compIdx'
  | 'completions'
  | 'onActiveSessionSelect'
  | 'onActiveSessionClose'
  | 'onModelSelect'
  | 'onNewLiveSession'
  | 'onNewPromptSession'
  | 'onResumeSelect'
  | 'pagerPageSize'
>) {
  const { gw } = useGateway()
  const overlay = useStore($overlayState)
  const sid = useStore($uiSessionId)
  const theme = useStore($uiTheme)
  const ui = useStore($uiState)

  const hasAny =
    overlay.modelPicker ||
    overlay.pager ||
    overlay.sessions ||
    overlay.skillsHub ||
    overlay.pluginsHub ||
    completions.length

  if (!hasAny) {
    return null
  }

  // Fixed viewport centered on compIdx — previously the slice end was
  // compIdx + 8 so the dropdown grew from 8 rows to 16 as the user scrolled
  // down, bouncing the height on every keystroke.
  const viewportSize = Math.min(COMPLETION_WINDOW, completions.length)

  const start = Math.max(0, Math.min(compIdx - Math.floor(COMPLETION_WINDOW / 2), completions.length - viewportSize))

  // When the status bar is pinned to the bottom (below the composer),
  // bottom="100%" pushes the overlay into/behind the status bar rows.
  // Anchor from the top of the relative container instead so the
  // overlay stays visible above the composer input.
  const overlayPositioning =
    ui.statusBar === 'bottom'
      ? { left: 0 as const, position: 'absolute' as const, right: 0 as const, top: 0 as const }
      : { bottom: '100%' as const, left: 0 as const, position: 'absolute' as const, right: 0 as const }

  return (
    <Box alignItems="flex-start" flexDirection="column" {...overlayPositioning}>
      {overlay.sessions && (
        <FloatBox color={theme.color.border}>
          <ActiveSessionSwitcher
            currentSessionId={sid}
            gw={gw}
            onCancel={() => patchOverlayState({ sessions: false })}
            onClose={onActiveSessionClose}
            onNew={onNewLiveSession}
            onNewPrompt={onNewPromptSession}
            onResume={onResumeSelect}
            onSelect={onActiveSessionSelect}
            t={theme}
          />
        </FloatBox>
      )}

      {overlay.modelPicker && (
        <FloatBox color={theme.color.border}>
          <ModelPicker
            gw={gw}
            onCancel={() => patchOverlayState({ modelPicker: false })}
            onSelect={onModelSelect}
            sessionId={sid}
            t={theme}
          />
        </FloatBox>
      )}

      {overlay.skillsHub && (
        <FloatBox color={theme.color.border}>
          <SkillsHub gw={gw} onClose={() => patchOverlayState({ skillsHub: false })} t={theme} />
        </FloatBox>
      )}

      {overlay.pluginsHub && (
        <FloatBox color={theme.color.border}>
          <PluginsHub gw={gw} onClose={() => patchOverlayState({ pluginsHub: false })} t={theme} />
        </FloatBox>
      )}

      {overlay.pager && (
        <FloatBox color={theme.color.border}>
          <Box flexDirection="column" paddingX={1} paddingY={1}>
            {overlay.pager.title && (
              <Box justifyContent="center" marginBottom={1}>
                <Text bold color={theme.color.primary}>
                  {overlay.pager.title}
                </Text>
              </Box>
            )}

            {overlay.pager.lines.slice(overlay.pager.offset, overlay.pager.offset + pagerPageSize).map((line, i) => (
              <Text key={i}>{line}</Text>
            ))}

            <Box marginTop={1}>
              <OverlayHint t={theme}>
                {overlay.pager.offset + pagerPageSize < overlay.pager.lines.length
                  ? `↑↓/jk line · Enter/Space/PgDn page · b/PgUp back · g/G top/bottom · Esc/q close (${Math.min(overlay.pager.offset + pagerPageSize, overlay.pager.lines.length)}/${overlay.pager.lines.length})`
                  : `end · ↑↓/jk · b/PgUp back · g top · Esc/q close (${overlay.pager.lines.length} lines)`}
              </OverlayHint>
            </Box>
          </Box>
        </FloatBox>
      )}

      {!!completions.length && (
        <FloatBox color={theme.color.primary}>
          <Box flexDirection="column" width={Math.max(28, cols - 6)}>
            {completions.slice(start, start + viewportSize).map((item, i) => {
              const active = start + i === compIdx

              return (
                <Box
                  backgroundColor={active ? theme.color.completionCurrentBg : theme.color.completionBg}
                  flexDirection="row"
                  key={`${start + i}:${item.text}:${item.display}:${item.meta ?? ''}`}
                  width="100%"
                >
                  {/* flexShrink=0 — when meta overflows the row, Ink/Yoga
                      otherwise shaves the last char off the display column
                      (e.g. /goal renders as /goa). */}
                  <Box flexShrink={0}>
                    <Text bold color={theme.color.label}>
                      {' '}
                      {item.display}
                    </Text>
                  </Box>
                  {item.meta ? (
                    <Text
                      backgroundColor={active ? theme.color.completionMetaCurrentBg : theme.color.completionMetaBg}
                      color={theme.color.muted}
                    >
                      {' '}
                      {item.meta}
                    </Text>
                  ) : null}
                </Box>
              )
            })}
          </Box>
        </FloatBox>
      )}
    </Box>
  )
}
