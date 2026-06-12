/**
 * App — the Solid view shell (spec v4 §2 `view/App.tsx`). Header + a content zone
 * that is either the PAGER overlay (long slash output) or the normal
 * transcript + input zone; the input zone is one of: blocking prompt, resume
 * picker (/sessions), generic picker (model/skills), or the composer. Fully
 * themed (§7.5).
 *
 *   header     flexShrink:0            (top chrome line)
 *   content    flexGrow:1, minHeight:0 — Pager OR (transcript + input zone)
 *   transcript flexGrow:1, minHeight:0 (the one <scrollbox>; §8 #2 gotchas)
 *   input zone flexShrink:0            (PromptOverlay | SessionPicker | Picker | Composer)
 *
 * Overlays REPLACE rather than stack (a `<Switch>`), so the composer remounts +
 * refocuses when an overlay closes; the key that closed an overlay can't leak
 * into it because the close is deferred a tick.
 */
import { Match, Switch } from 'solid-js'

import { deferClose } from '../logic/defer.ts'
import type { PromptHistory as ComposerHistory } from '../logic/history.ts'
import type { PasteStore } from '../logic/pastes.ts'
import { actionCommand, promptHistoryEntries } from '../logic/promptHistory.ts'
import type { SessionStore } from '../logic/store.ts'
import { AgentsTray, type AgentsTrayApi } from './agentsTray.tsx'
import { Composer } from './composer.tsx'
import { DimensionsProvider } from './dimensions.tsx'
import { Header } from './header.tsx'
import { AgentsDashboard } from './overlays/agentsDashboard.tsx'
import { Pager } from './overlays/pager.tsx'
import { Picker } from './overlays/picker.tsx'
import { PromptHistory } from './overlays/promptHistory.tsx'
import { SessionPicker, type SessionPickerOps } from './overlays/sessionPicker.tsx'
import { PromptOverlay } from './prompts/promptOverlay.tsx'
import { SessionInfoProvider } from './sessionInfo.tsx'
import { StatusBar } from './statusBar.tsx'
import { StatusLine } from './statusLine.tsx'
import { useTheme } from './theme.tsx'
import { Transcript } from './transcript.tsx'

export interface AppProps {
  readonly store: SessionStore
  readonly onSubmit?: (text: string) => void
  readonly onType?: (text: string) => void
  readonly onRespond?: (method: string, params: Record<string, unknown>) => void
  readonly onResume?: (sessionId: string) => void
  /** Gateway calls for the resume picker (session.list/peek/title). */
  readonly sessionOps?: SessionPickerOps
  /** Fired after the resume picker closes WITHOUT a pick (boot path: the
   *  entry creates a fresh session when none exists yet). */
  readonly onSessionPickerClosed?: () => void
  readonly sessionId?: () => string | undefined
  readonly history?: ComposerHistory
  readonly onImagePaste?: () => void
  readonly pasteStore?: PasteStore
}

const NOOP = () => {}
const NOOP_RESPOND = () => {}
const NOOP_RESUME = () => {}
const NO_SESSION = () => undefined
/** Inert picker ops for headless mounts that pass no gateway (tests). */
const NOOP_OPS: SessionPickerOps = {
  list: () => Promise.resolve({ sessions: [], truncated: false }),
  peek: () => Promise.resolve({}),
  rename: () => Promise.resolve()
}

export function App(props: AppProps) {
  const theme = useTheme()
  // Background-agents tray focus plumbing (Epic 2.7): Down on an empty composer
  // hands focus to the tray; Esc from the tray hands it back. Plain refs — the
  // tray persists across composer remounts (overlay close re-registers focus).
  let trayApi: AgentsTrayApi | undefined
  let focusComposer: (() => void) | undefined
  const blocked = () => props.store.state.prompt !== undefined
  const pager = () => props.store.state.pager
  const dashboard = () => props.store.state.dashboard
  const sessionPicker = () => props.store.state.sessionPicker
  const picker = () => props.store.state.picker
  const promptHistory = () => props.store.state.promptHistory
  // Defer the close so the key that closed an overlay (Esc/q/Enter) can't land in
  // the freshly-remounted composer (see deferClose).
  const closePager = () => deferClose(() => props.store.closePager())
  const closeDashboard = () => deferClose(() => props.store.closeDashboard())
  // close WITHOUT a pick — the boot path may create a fresh session here.
  const closeSessionPicker = () =>
    deferClose(() => {
      props.store.closeSessionPicker()
      props.onSessionPickerClosed?.()
    })
  const closePicker = () => deferClose(() => props.store.closePicker())
  const closePromptHistory = () => deferClose(() => props.store.closePromptHistory())
  // Esc+Esc viewer trigger (Epic 5): only when this session HAS user prompts —
  // an empty session opens nothing (no empty modal).
  const openPromptHistory = () => {
    if (promptHistoryEntries(props.store.state.messages).length > 0) props.store.openPromptHistory()
  }
  const resume = (id: string) => {
    ;(props.onResume ?? NOOP_RESUME)(id)
    // a PICK closes without the no-pick callback (the resume owns the session)
    deferClose(() => props.store.closeSessionPicker())
  }

  return (
    <DimensionsProvider>
      {/* live session chrome (cwd, model, …) for deep nodes — e.g. the file-tool
          renderer relativizes paths against `info.cwd` (Epic 2.3). */}
      <SessionInfoProvider info={() => props.store.state.info}>
        <box style={{ flexDirection: 'column', flexGrow: 1, paddingTop: 1, paddingLeft: 1, paddingRight: 1 }}>
          {/* a bottom rule under the header bookends the transcript with the status
          bar's top rule — frames the chrome as intentional (item 8). */}
          <box border={['bottom']} borderColor={theme().color.border} style={{ flexShrink: 0 }}>
            <Header store={props.store} />
          </box>
          {/* content zone: a full-screen overlay (pager / agents dashboard) OR the transcript + input zone */}
          <Switch
            fallback={
              <>
                <Transcript store={props.store} />
                {/* transient busy face floats at the bottom of the transcript area */}
                <StatusLine store={props.store} />
                {/* input region — a top-edge rule separates the status bar + textbox from the
                transcript above; the status bar sits directly ABOVE the composer (item 14). */}
                <box
                  border={['top']}
                  borderColor={theme().color.border}
                  style={{ flexShrink: 0, flexDirection: 'column' }}
                >
                  <StatusBar store={props.store} />
                  <Switch
                    fallback={
                      <Composer
                        onSubmit={props.onSubmit ?? NOOP}
                        onType={props.onType}
                        completions={() => props.store.state.completions ?? []}
                        completionFrom={() => props.store.state.completionFrom}
                        onDismiss={() => props.store.clearCompletions()}
                        history={props.history}
                        onImagePaste={props.onImagePaste}
                        pasteStore={props.pasteStore}
                        onFocusDown={() => trayApi?.focusTray() ?? false}
                        registerFocus={fn => (focusComposer = fn)}
                        onDoubleEsc={openPromptHistory}
                      />
                    }
                  >
                    <Match when={blocked()}>
                      <PromptOverlay
                        store={props.store}
                        onRespond={props.onRespond ?? NOOP_RESPOND}
                        sessionId={props.sessionId ?? NO_SESSION}
                      />
                    </Match>
                    <Match when={sessionPicker()}>
                      {sp => (
                        <SessionPicker
                          ops={props.sessionOps ?? NOOP_OPS}
                          initialTab={sp().tab}
                          currentCwd={() => props.store.state.info.cwd}
                          onResume={resume}
                          onClose={closeSessionPicker}
                        />
                      )}
                    </Match>
                    <Match when={picker()}>
                      {p => (
                        <Picker
                          title={p().title}
                          items={p().items}
                          onPick={value => {
                            p().onPick(value)
                            closePicker()
                          }}
                          onClose={closePicker}
                        />
                      )}
                    </Match>
                    <Match when={promptHistory()}>
                      <PromptHistory
                        entries={promptHistoryEntries(props.store.state.messages)}
                        onAction={action => (props.onSubmit ?? NOOP)(actionCommand(action))}
                        onClose={closePromptHistory}
                      />
                    </Match>
                  </Switch>
                  {/* background-agents tray (Epic 2.7): renders nothing with no
                      running agents; a one-line indicator otherwise; expands while
                      focused (composer Down). Enter opens the dashboard on that row. */}
                  <AgentsTray
                    subagents={props.store.state.subagents}
                    onOpen={id => props.store.openDashboard(id)}
                    onExit={() => focusComposer?.()}
                    bind={api => (trayApi = api)}
                  />
                </box>
              </>
            }
          >
            <Match when={pager()}>{p => <Pager title={p().title} text={p().text} onClose={closePager} />}</Match>
            <Match when={dashboard()}>
              <AgentsDashboard
                subagents={props.store.state.subagents}
                onClose={closeDashboard}
                preselect={props.store.state.dashboardAgent}
              />
            </Match>
          </Switch>
        </box>
      </SessionInfoProvider>
    </DimensionsProvider>
  )
}
