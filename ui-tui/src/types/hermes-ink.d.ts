import type * as React from 'react'

declare module '@hermes/ink' {
  export type Key = {
    readonly ctrl: boolean
    readonly meta: boolean
    readonly super: boolean
    readonly shift: boolean
    readonly alt: boolean
    readonly upArrow: boolean
    readonly downArrow: boolean
    readonly leftArrow: boolean
    readonly rightArrow: boolean
    readonly return: boolean
    readonly backspace: boolean
    readonly delete: boolean
    readonly escape: boolean
    readonly tab: boolean
    readonly pageUp: boolean
    readonly pageDown: boolean
    readonly wheelUp: boolean
    readonly wheelDown: boolean
    readonly home: boolean
    readonly end: boolean
    readonly [key: string]: boolean
  }

  export type InputEvent = {
    readonly input: string
    readonly key: Key
    readonly keypress: { readonly raw?: string }
  }

  export type InputHandler = (input: string, key: Key, event: InputEvent) => void

  export type FrameEvent = {
    readonly durationMs: number
    readonly phases?: {
      readonly renderer: number
      readonly diff: number
      readonly optimize: number
      readonly write: number
      readonly patches: number
      readonly optimizedPatches: number
      readonly writeBytes: number
      readonly backpressure: boolean
      readonly prevFrameDrainMs: number
      readonly yoga: number
      readonly commit: number
      readonly yogaVisited: number
      readonly yogaMeasured: number
      readonly yogaCacheHits: number
      readonly yogaLive: number
    }
    readonly flickers: ReadonlyArray<{
      readonly desiredHeight: number
      readonly availableHeight: number
      readonly reason: 'resize' | 'offscreen' | 'clear'
    }>
  }

  export type RenderOptions = {
    readonly stdin?: NodeJS.ReadStream
    readonly stdout?: NodeJS.WriteStream
    readonly stderr?: NodeJS.WriteStream
    readonly exitOnCtrlC?: boolean
    readonly patchConsole?: boolean
    readonly onFrame?: (event: FrameEvent) => void
  }

  export type Instance = {
    readonly rerender: (node: React.ReactNode) => void
    readonly unmount: () => void
    readonly waitUntilExit: () => Promise<void>
    readonly cleanup: () => void
  }

  export type ScrollBoxHandle = {
    readonly scrollTo: (y: number) => void
    readonly scrollBy: (dy: number) => void
    readonly scrollToElement: (el: unknown, offset?: number) => void
    readonly scrollToBottom: () => void
    readonly getScrollTop: () => number
    readonly getPendingDelta: () => number
    readonly getScrollHeight: () => number
    readonly getViewportHeight: () => number
    readonly getViewportTop: () => number
    readonly getLastManualScrollAt: () => number
    readonly isSticky: () => boolean
    readonly subscribe: (listener: () => void) => () => void
    readonly setClampBounds: (min: number | undefined, max: number | undefined) => void
  }

  export const Box: React.ComponentType<any>
  export const AlternateScreen: React.ComponentType<any>
  export const Ansi: React.ComponentType<any>
  export const Link: React.ComponentType<{
    readonly children?: React.ReactNode
    readonly fallback?: React.ReactNode
    readonly url: string
  }>
  export const NoSelect: React.ComponentType<any>
  export const ScrollBox: React.ComponentType<any>
  export const Text: React.ComponentType<any>
  export const TextInput: React.ComponentType<any>
  export const stringWidth: (s: string) => number
  export function isXtermJs(): boolean

  export type ScrollFastPathStats = {
    captured: number
    taken: number
    declined: {
      noPrevScreen: number
      heightDeltaMismatch: number
      other: number
    }
    lastDeclineReason?: string
    lastHeightDelta?: number
    lastHintDelta?: number
    lastScrollHeight?: number
    lastPrevHeight?: number
  }
  export const scrollFastPathStats: ScrollFastPathStats

  export type EvictLevel = 'all' | 'half'
  export type InkCacheSizes = {
    readonly lineWidth: number
    readonly slice: number
    readonly width: number
    readonly wrap: number
  }
  export function evictInkCaches(level?: EvictLevel): InkCacheSizes

  export function render(node: React.ReactNode, options?: NodeJS.WriteStream | RenderOptions): Instance

  export function useApp(): { readonly exit: (error?: Error) => void }
  export type RunExternalProcess = () => Promise<void>
  export function useExternalProcess(): (run: RunExternalProcess) => Promise<void>
  export function withInkSuspended(run: RunExternalProcess): Promise<void>
  export function useInput(handler: InputHandler, options?: { readonly isActive?: boolean }): void
  export function useSelection(): {
    readonly copySelection: () => Promise<string>
    readonly copySelectionNoClear: () => Promise<string>
    readonly clearSelection: () => void
    readonly hasSelection: () => boolean
    readonly getState: () => unknown
    readonly subscribe: (cb: () => void) => () => void
    readonly shiftAnchor: (dRow: number, minRow: number, maxRow: number) => void
    readonly shiftSelection: (dRow: number, minRow: number, maxRow: number) => void
    readonly moveFocus: (move: unknown) => void
    readonly captureScrolledRows: (firstRow: number, lastRow: number, side: 'above' | 'below') => void
    readonly setSelectionBgColor: (color: string) => void
  }
  export function useHasSelection(): boolean
  export function useStdout(): { readonly stdout?: NodeJS.WriteStream }
  export function useTerminalFocus(): boolean
  export function useTerminalTitle(title: string | null): void
  export function useDeclaredCursor(args: {
    readonly line: number
    readonly column: number
    readonly active: boolean
  }): (el: unknown) => void
  export function useStdin(): {
    readonly stdin: NodeJS.ReadStream
    readonly setRawMode: (value: boolean) => void
    readonly isRawModeSupported: boolean
    readonly exitOnCtrlC: boolean
    readonly inputEmitter: NodeJS.EventEmitter
    readonly querier: unknown
  }
}
