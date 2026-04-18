import type * as React from 'react'

declare module '@hermes/ink' {
  export type Key = {
    readonly ctrl: boolean
    readonly meta: boolean
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

  export type RenderOptions = {
    readonly stdin?: NodeJS.ReadStream
    readonly stdout?: NodeJS.WriteStream
    readonly stderr?: NodeJS.WriteStream
    readonly exitOnCtrlC?: boolean
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
    readonly isSticky: () => boolean
    readonly subscribe: (listener: () => void) => () => void
  }

  export const Box: React.ComponentType<any>
  export const AlternateScreen: React.ComponentType<any>
  export const Ansi: React.ComponentType<any>
  export const Link: React.ComponentType<{ readonly url: string; readonly children?: React.ReactNode; readonly fallback?: React.ReactNode }>
  export const NoSelect: React.ComponentType<any>
  export const ScrollBox: React.ComponentType<any>
  export const Text: React.ComponentType<any>
  export const TextInput: React.ComponentType<any>
  export const stringWidth: (s: string) => number

  export function render(node: React.ReactNode, options?: NodeJS.WriteStream | RenderOptions): Instance

  export function useApp(): { readonly exit: (error?: Error) => void }
  export type RunExternalProcess = () => Promise<void>
  export function useExternalProcess(): (run: RunExternalProcess) => Promise<void>
  export function withInkSuspended(run: RunExternalProcess): Promise<void>
  export function useInput(handler: InputHandler, options?: { readonly isActive?: boolean }): void
  export function useSelection(): {
    readonly copySelection: () => string
    readonly copySelectionNoClear: () => string
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
