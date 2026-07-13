export { default as useStderr } from './hooks/use-stderr.js'
export { default as useStdout } from './hooks/use-stdout.js'
export { Ansi } from './ink/Ansi.js'
export { evictInkCaches, type EvictLevel, type InkCacheSizes } from './ink/cache-eviction.js'
export { AlternateScreen } from './ink/components/AlternateScreen.js'
export { default as Box } from './ink/components/Box.js'
export { default as Link } from './ink/components/Link.js'
export { default as Newline } from './ink/components/Newline.js'
export { NoSelect } from './ink/components/NoSelect.js'
export { RawAnsi } from './ink/components/RawAnsi.js'
export { default as ScrollBox } from './ink/components/ScrollBox.js'
export { default as Spacer } from './ink/components/Spacer.js'
export { default as Text } from './ink/components/Text.js'
export { default as useApp } from './ink/hooks/use-app.js'
export { useCursorAdvance } from './ink/hooks/use-cursor-advance.js'
export { useDeclaredCursor } from './ink/hooks/use-declared-cursor.js'
export { type RunExternalProcess, useExternalProcess, withInkSuspended } from './ink/hooks/use-external-process.js'
export { default as useInput } from './ink/hooks/use-input.js'
export { useHasSelection, useSelection } from './ink/hooks/use-selection.js'
export { default as useStdin } from './ink/hooks/use-stdin.js'
export { useTabStatus } from './ink/hooks/use-tab-status.js'
export { useTerminalFocus } from './ink/hooks/use-terminal-focus.js'
export { useTerminalTitle } from './ink/hooks/use-terminal-title.js'
export { useTerminalViewport } from './ink/hooks/use-terminal-viewport.js'
export { default as measureElement } from './ink/measure-element.js'
export { scrollFastPathStats, type ScrollFastPathStats } from './ink/render-node-to-output.js'
export { createRoot, forceRedraw, default as render, renderSync } from './ink/root.js'
export { stringWidth } from './ink/stringWidth.js'
export { isXtermJs } from './ink/terminal.js'
export type { MouseTrackingMode } from './ink/termio/dec.js'
export { wrapAnsi } from './ink/wrapAnsi.js'

// NOTE: Do not re-export from 'ink-text-input' here.
//
// 'ink-text-input' depends on the npm 'ink' package; pulling it in from
// this re-export drags an entire second copy of ink (and its async
// top-level init chain) into any caller that bundles `@hermes/ink` from
// source. esbuild's `__esm` helper then deadlocks on the circular
// async init between the two ink graphs — the dashboard TUI bundle
// stalls at startup with only 141 bytes of ANSI reset output, blank
// screen forever (#31227).
//
// Consumers that actually want the upstream ink-text-input widget must
// import it via the dedicated subpath:
//
//     import TextInput from '@hermes/ink/text-input'
//
// which still resolves through this package's `./text-input` export,
// just outside the entry-exports surface that gets inlined by callers.
