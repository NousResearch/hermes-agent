import { AlternateScreen, Box, NoSelect, ScrollBox, Text } from '@hermes/ink'
import { useStore } from '@nanostores/react'
import { Fragment, memo, useEffect, useMemo, useRef } from 'react'

import { useGateway } from '../app/gatewayContext.js'
import type { AppLayoutProps } from '../app/interfaces.js'
import { $isBlocked, $overlayState, patchOverlayState } from '../app/overlayStore.js'
import { $petBox } from '../app/petFlashStore.js'
import { $uiState } from '../app/uiStore.js'
import { usePet } from '../app/usePet.js'
import { INLINE_MODE, SHOW_FPS, TERMUX_TUI_MODE } from '../config/env.js'
import { PLACEHOLDER } from '../content/placeholders.js'
import { prevRenderedMsg } from '../domain/blockLayout.js'
import { buildStatusCapsule } from '../domain/statusCapsule.js'
import {
  COMPOSER_PROMPT_GAP_WIDTH,
  composerPromptWidth,
  inputVisualHeight,
  stableComposerColumns
} from '../lib/inputMetrics.js'
import { PerfPane } from '../lib/perfPane.js'
import { composerPromptText } from '../lib/prompt.js'

import { AgentsOverlay } from './agentsOverlay.js'
import { GoodVibesHeart, StatusRule, StickyPromptTracker, TranscriptScrollbar } from './appChrome.js'
import { FloatingOverlays, PromptZone } from './appOverlays.js'
import { Banner, Panel, SessionPanel } from './branding.js'
import { FpsOverlay } from './fpsOverlay.js'
import { HelpHint } from './helpHint.js'
import { Journey } from './journey.js'
import { MessageLine } from './messageLine.js'
import { PetKitty, PetSprite } from './petSprite.js'
import { QueuedMessages } from './queuedMessages.js'
import { LiveTodoPanel, StreamingAssistant } from './streamingAssistant.js'
import { TextInput, type TextInputMouseApi } from './textInput.js'

// Box geometry, kept here so the transcript's reservation math matches the
// rendered overlay exactly.
const PET_BOTTOM = 3 // rows the pet floats above the screen bottom (over the composer)
const PET_PAD_LEFT = 2
const PET_RIGHT = 1
const PET_GUTTER_GAP = 1
const KITTY_PLACEHOLDER = '\u{10eeee}'
// Below this many columns of remaining text width, the right gutter is too
// cramped, so the transcript collapses to reserving bottom rows instead.
const MIN_GUTTER_BODY_COLS = 72

// Petdex mascot — a small floating overlay riding the bottom-right corner just
// above the status bar, with a little top/left breathing room. It reserves no
// layout rows (the transcript scrolls underneath); instead it publishes its
// footprint so the transcript can keep its text clear of it (right gutter on
// wide terminals, reserved bottom rows on narrow ones). Renders nothing unless
// a pet is installed + enabled.
export const PetPane = memo(function PetPane() {
  const { enabled, grid, kitty } = usePet()

  // Footprint in cells. For kitty we count real placeholder cells (zero-width
  // diacritics make string length lie); for half-blocks it's the grid shape.
  const { width, height } = useMemo(() => {
    if (kitty) {
      return {
        height: kitty.placeholder.length,
        width: Math.max(0, ...kitty.placeholder.map(row => [...row].filter(ch => ch === KITTY_PLACEHOLDER).length))
      }
    }

    if (grid) {
      return { height: grid.length, width: Math.max(0, ...grid.map(row => row.length)) }
    }

    return { height: 0, width: 0 }
  }, [grid, kitty])

  const active = enabled && width > 0 && height > 0

  useEffect(() => {
    $petBox.set(
      active
        ? {
            // Bottom PET_BOTTOM rows sit over the composer, so the transcript
            // only needs to clear the rest in the row-reservation (band) mode.
            height: Math.max(0, height - PET_BOTTOM),
            width: width + PET_PAD_LEFT + PET_RIGHT + PET_GUTTER_GAP
          }
        : null
    )

    return () => $petBox.set(null)
  }, [active, height, width])

  if (!active) {
    return null
  }

  return (
    <NoSelect bottom={PET_BOTTOM} flexShrink={0} paddingLeft={PET_PAD_LEFT} paddingTop={1} position="absolute" right={PET_RIGHT}>
      {kitty ? <PetKitty color={kitty.color} placeholder={kitty.placeholder} /> : null}
      {!kitty && grid ? <PetSprite grid={grid} /> : null}
    </NoSelect>
  )
})

const PromptPrefix = memo(function PromptPrefix({
  bold = false,
  color,
  promptText,
  width
}: {
  bold?: boolean
  color: string
  promptText: string
  width: number
}) {
  const glyphWidth = Math.max(1, width - COMPOSER_PROMPT_GAP_WIDTH)

  return (
    <Box width={width}>
      <Box width={glyphWidth}>
        <Text bold={bold} color={color}>
          {promptText}
        </Text>
      </Box>
      <Box width={COMPOSER_PROMPT_GAP_WIDTH} />
    </Box>
  )
})

const TranscriptPane = memo(function TranscriptPane({
  actions,
  composer,
  progress,
  transcript
}: Pick<AppLayoutProps, 'actions' | 'composer' | 'progress' | 'transcript'>) {
  const ui = useStore($uiState)
  const petBox = useStore($petBox)

  // Keep transcript text clear of the floating pet, responsively:
  //  - wide terminals: reserve a right gutter so lines wrap to the pet's left
  //    (as long as enough width is left for comfortable reading);
  //  - narrow terminals: keep full width and reserve bottom rows instead, so
  //    the newest lines sit above the pet rather than getting cramped.
  const useGutter = !!petBox && composer.cols - petBox.width >= MIN_GUTTER_BODY_COLS
  const bodyCols = useGutter && petBox ? composer.cols - petBox.width : composer.cols
  const petBandRows = petBox && !useGutter ? petBox.height : 0

  // LiveTodoPanel rides as a child of the latest user-message row so it
  // visually belongs to the prompt and follows it during scroll. -1 when
  // empty → row.index === -1 is always false → no render.
  const lastUserIdx = useMemo(() => {
    const items = transcript.historyItems

    for (let i = items.length - 1; i >= 0; i--) {
      if (items[i].role === 'user') {
        return i
      }
    }

    return -1
  }, [transcript.historyItems])

  // Index of the first user-role message; every later user message gets a
  // small dash above it so multi-turn transcripts visually segment by
  // turn. -1 when no user message has been sent yet → no separator ever
  // renders.
  const firstUserIdx = useMemo(
    () => transcript.historyItems.findIndex(m => m.role === 'user'),
    [transcript.historyItems]
  )

  return (
    <>
      <ScrollBox
        flexDirection="column"
        flexGrow={1}
        flexShrink={1}
        onClick={(e: { cellIsBlank?: boolean }) => {
          if (e.cellIsBlank) {
            actions.clearSelection()
          }
        }}
        ref={transcript.scrollRef}
        stickyScroll
      >
        <Box flexDirection="column" paddingX={1}>
          {transcript.virtualHistory.topSpacer > 0 ? <Box height={transcript.virtualHistory.topSpacer} /> : null}

          {transcript.virtualRows.slice(transcript.virtualHistory.start, transcript.virtualHistory.end).map(row => (
            <Box flexDirection="column" key={row.key} ref={transcript.virtualHistory.measureRef(row.key)}>
              {row.msg.role === 'user' && firstUserIdx >= 0 && row.index > firstUserIdx && (
                <Box marginTop={1}>
                  <Text color={ui.theme.color.border}>───</Text>
                </Box>
              )}

              {row.msg.kind === 'intro' ? (
                <Box flexDirection="column" paddingTop={1}>
                  <Banner maxWidth={Math.max(1, composer.cols - 2)} t={ui.theme} />

                  {row.msg.info && (
                    <SessionPanel
                      info={row.msg.info}
                      maxWidth={Math.max(1, composer.cols - 2)}
                      sid={ui.sid}
                      t={ui.theme}
                    />
                  )}
                </Box>
              ) : row.msg.kind === 'panel' && row.msg.panelData ? (
                <Panel sections={row.msg.panelData.sections} t={ui.theme} title={row.msg.panelData.title} />
              ) : (
                <MessageLine
                  cols={bodyCols}
                  compact={ui.compact}
                  detailsMode={ui.detailsMode}
                  detailsModeCommandOverride={ui.detailsModeCommandOverride}
                  msg={row.msg}
                  prev={prevRenderedMsg(i => transcript.virtualRows[i]?.msg, row.index, {
                    commandOverride: ui.detailsModeCommandOverride,
                    detailsMode: ui.detailsMode,
                    sections: ui.sections
                  })}
                  sections={ui.sections}
                  t={ui.theme}
                />
              )}

              {row.index === lastUserIdx && <LiveTodoPanel />}
            </Box>
          ))}

          {transcript.virtualHistory.bottomSpacer > 0 ? <Box height={transcript.virtualHistory.bottomSpacer} /> : null}

          <StreamingAssistant
            cols={bodyCols}
            compact={ui.compact}
            detailsMode={ui.detailsMode}
            detailsModeCommandOverride={ui.detailsModeCommandOverride}
            prevMsg={transcript.historyItems[transcript.historyItems.length - 1]}
            progress={progress}
            sections={ui.sections}
          />

          {/* Narrow terminals: reserve rows so the newest lines sit above the pet. */}
          {petBandRows > 0 ? <Box height={petBandRows} /> : null}
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

  const promptText = composerPromptText(
    ui.theme.brand.prompt,
    ui.info?.profile_name,
    sh,
    TERMUX_TUI_MODE,
    composer.cols
  )

  const promptWidth = composerPromptWidth(promptText)
  const promptBlank = ' '.repeat(promptWidth)
  const inputColumns = stableComposerColumns(composer.cols, promptWidth, TERMUX_TUI_MODE)
  const inputHeight = inputVisualHeight(composer.input, inputColumns)
  const inputMouseRef = useRef<null | TextInputMouseApi>(null)

  const captureInputDrag = (e: GutterMouseEvent) => {
    if (e.button !== 0) {
      return
    }

    e.stopImmediatePropagation?.()
    inputMouseRef.current?.startAtBeginning()
  }

  // Drag origin matches the input box's top-left, so localRow / localCol
  // map directly into TextInput coords (after backing out the prompt cell).
  const dragFromPromptRow = (e: GutterMouseEvent) => {
    if (e.button !== 0) {
      return
    }

    e.stopImmediatePropagation?.()
    inputMouseRef.current?.dragAt(e.localRow ?? 0, (e.localCol ?? 0) - promptWidth)
  }

  // Spacer rows live on a different vertical origin; only the column is
  // parent-aligned with the input. Force row=0 so vertical drags can't
  // jump the cursor to the wrong wrapped line.
  const dragFromSpacer = (e: GutterMouseEvent) => {
    if (e.button !== 0) {
      return
    }

    e.stopImmediatePropagation?.()
    inputMouseRef.current?.dragAt(0, (e.localCol ?? 0) - promptWidth)
  }

  const endInputDrag = () => inputMouseRef.current?.end()

  return (
    <NoSelect
      flexDirection="column"
      flexShrink={0}
      fromLeftEdge
      onClick={(e: { cellIsBlank?: boolean }) => {
        if (e.cellIsBlank) {
          actions.clearSelection()
        }
      }}
      paddingX={1}
    >
      <QueuedMessages
        cols={composer.cols}
        queued={composer.queuedDisplay}
        queueEditIdx={composer.queueEditIdx}
        t={ui.theme}
      />

      {ui.bgTasks.size > 0 && (
        <Text color={ui.theme.color.muted}>
          {ui.bgTasks.size} background {ui.bgTasks.size === 1 ? 'task' : 'tasks'} running
        </Text>
      )}

      {status.showStickyPrompt ? (
        <Text color={ui.theme.color.muted} wrap="truncate-end">
          <Text color={ui.theme.color.label}>↳ </Text>

          {status.stickyPrompt}
        </Text>
      ) : (
        <Box height={1} onMouseDown={captureInputDrag} onMouseDrag={dragFromSpacer} onMouseUp={endInputDrag} />
      )}

      <StatusRulePane at="top" composer={composer} status={status} />

      <Box flexDirection="column" marginTop={ui.statusBar === 'top' ? 0 : 1} position="relative">
        <FloatingOverlays
          cols={composer.cols}
          compIdx={composer.compIdx}
          completions={composer.completions}
          onActiveSessionClose={actions.closeLiveSession}
          onActiveSessionSelect={actions.activateLiveSession}
          onModelSelect={actions.onModelSelect}
          onNewLiveSession={actions.newLiveSession}
          onNewPromptSession={actions.newPromptSession}
          onResumeSelect={actions.resumeById}
          pagerPageSize={composer.pagerPageSize}
        />

        {composer.input === '?' && !composer.inputBuf.length && <HelpHint t={ui.theme} />}

        {!isBlocked && (
          <>
            {composer.inputBuf.map((line, i) => (
              <Box key={i}>
                <Box width={promptWidth}>
                  {i === 0 ? (
                    <PromptPrefix color={ui.theme.color.muted} promptText={promptText} width={promptWidth} />
                  ) : (
                    <Text color={ui.theme.color.muted}>{promptBlank}</Text>
                  )}
                </Box>

                <Text color={ui.theme.color.text}>{line || ' '}</Text>
              </Box>
            ))}

            <Box
              onMouseDown={captureInputDrag}
              onMouseDrag={dragFromPromptRow}
              onMouseUp={endInputDrag}
              position="relative"
              width={Math.max(1, composer.cols - 2)}
            >
              <Box width={promptWidth}>
                {sh ? (
                  <PromptPrefix color={ui.theme.color.shellDollar} promptText={promptText} width={promptWidth} />
                ) : composer.inputBuf.length ? (
                  <Text color={ui.theme.color.prompt}>{promptBlank}</Text>
                ) : (
                  <PromptPrefix bold color={ui.theme.color.prompt} promptText={promptText} width={promptWidth} />
                )}
              </Box>

              <Box flexGrow={0} flexShrink={0} height={inputHeight} width={inputColumns}>
                {/* Reserve the transcript scrollbar gutter too so typing never rewraps when the scrollbar column repaints. */}
                <TextInput
                  columns={inputColumns}
                  mouseApiRef={inputMouseRef}
                  onChange={composer.updateInput}
                  onPaste={composer.handleTextPaste}
                  onSubmit={composer.submit}
                  placeholder={composer.empty ? PLACEHOLDER : ui.busy ? 'Ctrl+C to interrupt…' : ''}
                  value={composer.input}
                  voiceRecordKey={composer.voiceRecordKey}
                />
              </Box>

              <Box position="absolute" right={0}>
                <GoodVibesHeart t={ui.theme} tick={status.goodVibesTick} />
              </Box>
            </Box>
          </>
        )}
      </Box>

      {!composer.empty && !ui.sid && <Text color={ui.theme.color.muted}>⚕ {ui.status}</Text>}

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

const JourneyPane = memo(function JourneyPane() {
  const { gw } = useGateway()
  const ui = useStore($uiState)

  return <Journey gw={gw} onClose={() => patchOverlayState({ journey: false })} t={ui.theme} />
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

  const statusCapsule = ui.mechanismStatusBar
    ? buildStatusCapsule({
        autoSession: !ui.sid && /forging|resuming|recovering|starting agent/i.test(ui.status),
        busy: ui.busy,
        lastTurnEndedAt: status.lastTurnEndedAt,
        prior: ui.statusCapsule,
        sessionStartedAt: status.sessionStartedAt,
        status: ui.status,
        turnStartedAt: status.turnStartedAt,
        usage: ui.usage
      })
    : null

  return (
    <Box marginTop={at === 'top' ? 1 : 0}>
      <StatusRule
        bgCount={ui.bgTasks.size}
        busy={ui.busy}
        cols={composer.cols}
        cwdLabel={status.cwdLabel}
        indicatorStyle={ui.indicatorStyle}
        lastTurnEndedAt={status.lastTurnEndedAt}
        liveSessionCount={ui.liveSessionCount}
        model={ui.info?.model ?? ''}
        modelFast={ui.info?.fast || ui.info?.service_tier === 'priority'}
        modelReasoningEffort={ui.info?.reasoning_effort}
        notice={ui.notice}
        onSessionCountClick={() => patchOverlayState({ sessions: true })}
        sessionStartedAt={status.sessionStartedAt}
        status={ui.status}
        statusCapsule={statusCapsule}
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
  const ui = useStore($uiState)

  // Inline mode skips AlternateScreen so the host terminal's native
  // scrollback captures rows scrolled off the top; composer + progress
  // stay anchored via normal flex-column flow.
  const Shell = INLINE_MODE ? Fragment : AlternateScreen
  const shellProps = INLINE_MODE ? {} : { mouseTracking }

  return (
    <Shell {...shellProps}>
      <Box flexDirection="column" flexGrow={1} position="relative">
        <Box flexDirection="row" flexGrow={1}>
          {overlay.agents ? (
            <PerfPane id="agents">
              <AgentsOverlayPane />
            </PerfPane>
          ) : overlay.journey ? (
            <PerfPane id="journey">
              <JourneyPane />
            </PerfPane>
          ) : (
            <PerfPane id="transcript">
              <TranscriptPane actions={actions} composer={composer} progress={progress} transcript={transcript} />
            </PerfPane>
          )}
        </Box>

        {!overlay.agents && !overlay.journey && (
          <>
            <PerfPane id="prompt">
              <PromptZone
                cols={composer.cols}
                onApprovalChoice={actions.answerApproval}
                onClarifyAnswer={actions.answerClarify}
                onSecretSubmit={actions.answerSecret}
                onSudoSubmit={actions.answerSudo}
              />
            </PerfPane>

            <PerfPane id="composer">
              <ComposerPane actions={actions} composer={composer} status={status} />
            </PerfPane>

            {SHOW_FPS && (
              <Box flexShrink={0} justifyContent="flex-end" paddingRight={1}>
                <FpsOverlay t={ui.theme} />
              </Box>
            )}
          </>
        )}

        {!overlay.agents && <PetPane />}
      </Box>
    </Shell>
  )
})

type GutterMouseEvent = {
  button: number
  localCol?: number
  localRow?: number
  stopImmediatePropagation?: () => void
}
