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
    readonly home: boolean
    readonly end: boolean
    readonly [key: string]: boolean
  }

  export type InputHandler = (input: string, key: Key) => void

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

  export const Box: React.ComponentType<any>
  export const Text: React.ComponentType<any>
  export const TextInput: React.ComponentType<any>
  export const stringWidth: (s: string) => number

  export function render(node: React.ReactNode, options?: NodeJS.WriteStream | RenderOptions): Instance

  export function useApp(): { readonly exit: (error?: Error) => void }
  export function useInput(handler: InputHandler, options?: { readonly isActive?: boolean }): void
  export function useStdout(): { readonly stdout?: NodeJS.WriteStream }
  export function useTerminalFocus(): boolean
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
