import { AlternateScreen, Box, NoSelect, ScrollBox, Text } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { memo } from 'react'

import { useGateway } from '../app/gatewayContext.js'
import type { AppLayoutProps } from '../app/interfaces.js'
import { $isBlocked, $overlayState, patchOverlayState } from '../app/overlayStore.js'
import { $uiState } from '../app/uiStore.js'
import { PLACEHOLDER } from '../content/placeholders.js'
import { inputVisualHeight, stableComposerColumns } from '../lib/inputMetrics.js'

import { AgentsOverlay } from './agentsOverlay.js'
import { GoodVibesHeart, StatusRule, StickyPromptTracker, TranscriptScrollbar } from './appChrome.js'
import { FloatingOverlays, PromptZone } from './appOverlays.js'
import { Banner, Panel, SessionPanel } from './branding.js'
import { MessageLine } from './messageLine.js'
import { QueuedMessages } from './queuedMessages.js'
import { LiveTodoPanel, StreamingAssistant } from './streamingAssistant.js'
import { TextInput } from './textInput.js'

const TranscriptPane = memo(function TranscriptPane({
  actions,
  composer,
  progress,
  transcript
}: Pick<AppLayoutProps, 'actions' | 'composer' | 'progress' | 'transcript'>) {
  const ui = useStore($uiState)

  return (
    <>
      <ScrollBox flexDirection="column" flexGrow={1} flexShrink={1} ref={transcript.scrollRef} stickyScroll>
        <Box flexDirection="column" paddingX={1}>
          <LiveTodoPanel />

          {transcript.virtualHistory.topSpacer > 0 ? <Box height={transcript.virtualHistory.topSpacer} /> : null}

          {transcript.virtualRows.slice(transcript.virtualHistory.start, transcript.virtualHistory.end).map(row => (
            <Box flexDirection="column" key={row.key} ref={transcript.virtualHistory.measureRef(row.key)}>
              {row.msg.kind === 'intro' ? (
                <Box flexDirection="column" paddingTop={1}>
                  <Banner t={ui.theme} />

                  {row.msg.info?.version && <SessionPanel info={row.msg.info} sid={ui.sid} t={ui.theme} />}
                </Box>
              ) : row.msg.kind === 'panel' && row.msg.panelData ? (
                <Panel sections={row.msg.panelData.sections} t={ui.theme} title={row.msg.panelData.title} />
              ) : (
                <MessageLine
                  cols={composer.cols}
                  compact={ui.compact}
                  detailsMode={ui.detailsMode}
                  detailsModeCommandOverride={ui.detailsModeCommandOverride}
                  msg={row.msg}
                  sections={ui.sections}
                  t={ui.theme}
                />
              )}
            </Box>
          ))}

          {transcript.virtualHistory.bottomSpacer > 0 ? <Box height={transcript.virtualHistory.bottomSpacer} /> : null}

          <StreamingAssistant
            cols={composer.cols}
            compact={ui.compact}
            detailsMode={ui.detailsMode}
            detailsModeCommandOverride={ui.detailsModeCommandOverride}
            progress={progress}
            sections={ui.sections}
          />
        </Box>
      </ScrollBox>

      <NoSelect flexShrink={0} marginLeft={1}>
        <TranscriptScrollbar scrollRef={transcript.scrollRef} t={ui.theme} />
      </NoSelect>

      <StickyPromptTracker
        messages={transcript.historyItems}
        offsets={transcript.virtualHistory.offsets}
        onChange={actions.setStickyPrompt}
        scrollRef={transcript.scrollRef}
      />
    </>
  )
})

const ComposerPane = memo(function ComposerPane({
  actions,
  composer,
  status
}: Pick<AppLayoutProps, 'actions' | 'composer' | 'status'>) {
  const ui = useStore($uiState)
  const isBlocked = useStore($isBlocked)
  const sh = (composer.inputBuf[0] ?? composer.input).startsWith('!')
  const pw = sh ? 2 : 3
  const inputColumns = stableComposerColumns(composer.cols, pw)
  const inputHeight = inputVisualHeight(composer.input, inputColumns)

  return (
    <NoSelect flexDirection="column" flexShrink={0} fromLeftEdge paddingX={1}>
      <QueuedMessages
        cols={composer.cols}
        queued={composer.queuedDisplay}
        queueEditIdx={composer.queueEditIdx}
        t={ui.theme}
      />

      {ui.bgTasks.size > 0 && (
        <Text color={ui.theme.color.dim}>
          {ui.bgTasks.size} background {ui.bgTasks.size === 1 ? 'task' : 'tasks'} running
        </Text>
      )}

      {status.showStickyPrompt ? (
        <Text color={ui.theme.color.dim} wrap="truncate-end">
          <Text color={ui.theme.color.label}>↳ </Text>

          {status.stickyPrompt}
        </Text>
      ) : (
        <Text> </Text>
      )}

      <StatusRulePane at="top" composer={composer} status={status} />

      <Box flexDirection="column" marginTop={ui.statusBar === 'top' ? 0 : 1} position="relative">
        <FloatingOverlays
          cols={composer.cols}
          compIdx={composer.compIdx}
          completions={composer.completions}
          onModelSelect={actions.onModelSelect}
          onPickerSelect={actions.resumeById}
          pagerPageSize={composer.pagerPageSize}
        />

        {!isBlocked && (
          <>
            {composer.inputBuf.map((line, i) => (
              <Box key={i}>
                <Box width={3}>
                  <Text color={ui.theme.color.dim}>{i === 0 ? `${ui.theme.brand.prompt} ` : '  '}</Text>
                </Box>

                <Text color={ui.theme.color.cornsilk}>{line || ' '}</Text>
              </Box>
            ))}

            <Box position="relative">
              <Box width={pw}>
                {sh ? (
                  <Text color={ui.theme.color.shellDollar}>$ </Text>
                ) : (
                  <Text bold color={ui.theme.color.prompt}>
                    {composer.inputBuf.length ? '  ' : `${ui.theme.brand.prompt} `}
                  </Text>
                )}
              </Box>

              <Box flexGrow={0} flexShrink={0} height={inputHeight} position="relative" width={inputColumns}>
                {/* Reserve the transcript scrollbar gutter too so typing never rewraps when the scrollbar column repaints. */}
                <TextInput
                  columns={inputColumns}
                  onChange={composer.updateInput}
                  onPaste={composer.handleTextPaste}
                  onSubmit={composer.submit}
                  placeholder={composer.empty ? PLACEHOLDER : ui.busy ? 'Ctrl+C to interrupt…' : ''}
                  value={composer.input}
                />

                <Box position="absolute" right={0}>
                  <GoodVibesHeart t={ui.theme} tick={status.goodVibesTick} />
                </Box>
              </Box>
            </Box>
          </>
        )}
      </Box>

      {!composer.empty && !ui.sid && <Text color={ui.theme.color.dim}>⚕ {ui.status}</Text>}

      <StatusRulePane at="bottom" composer={composer} status={status} />
    </NoSelect>
  )
})

const AgentsOverlayPane = memo(function AgentsOverlayPane() {
  const { gw } = useGateway()
  const ui = useStore($uiState)
  const overlay = useStore($overlayState)

  return (
    <AgentsOverlay
      gw={gw}
      initialHistoryIndex={overlay.agentsInitialHistoryIndex}
      onClose={() => patchOverlayState({ agents: false, agentsInitialHistoryIndex: 0 })}
      t={ui.theme}
    />
  )
})

const StatusRulePane = memo(function StatusRulePane({
  at,
  composer,
  status
}: Pick<AppLayoutProps, 'composer' | 'status'> & { at: 'bottom' | 'top' }) {
  const ui = useStore($uiState)

  if (ui.statusBar !== at) {
    return null
  }

  return (
    <Box marginTop={at === 'top' ? 1 : 0}>
      <StatusRule
        bgCount={ui.bgTasks.size}
        busy={ui.busy}
        cols={composer.cols}
        cwdLabel={status.cwdLabel}
        model={ui.info?.model ?? ''}
        modelFast={ui.info?.fast || ui.info?.service_tier === 'priority'}
        modelReasoningEffort={ui.info?.reasoning_effort}
        sessionStartedAt={status.sessionStartedAt}
        showCost={ui.showCost}
        status={ui.status}
        statusColor={status.statusColor}
        t={ui.theme}
        turnStartedAt={status.turnStartedAt}
        usage={ui.usage}
        voiceLabel={status.voiceLabel}
      />
    </Box>
  )
})

export const AppLayout = memo(function AppLayout({
  actions,
  composer,
  mouseTracking,
  progress,
  status,
  transcript
}: AppLayoutProps) {
  const overlay = useStore($overlayState)

  return (
    <AlternateScreen mouseTracking={mouseTracking}>
      <Box flexDirection="column" flexGrow={1}>
        <Box flexDirection="row" flexGrow={1}>
          {overlay.agents ? (
            <AgentsOverlayPane />
          ) : (
            <TranscriptPane actions={actions} composer={composer} progress={progress} transcript={transcript} />
          )}
        </Box>

        {!overlay.agents && (
          <>
            <PromptZone
              cols={composer.cols}
              onApprovalChoice={actions.answerApproval}
              onClarifyAnswer={actions.answerClarify}
              onSecretSubmit={actions.answerSecret}
              onSudoSubmit={actions.answerSudo}
            />

            <ComposerPane actions={actions} composer={composer} status={status} />
          </>
        )}
      </Box>
    </AlternateScreen>
  )
})
