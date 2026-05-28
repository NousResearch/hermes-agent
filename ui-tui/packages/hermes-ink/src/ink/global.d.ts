declare module 'signal-exit' {
  export type ExitHandler = (code: number | null, signal: NodeJS.Signals | null) => void

  export interface Options {
    alwaysLast?: boolean
  }

  export function onExit(handler: ExitHandler, options?: Options): () => void
}
