import wrapAnsiModule from 'wrap-ansi'

type WrapAnsiOptions = {
  hard?: boolean
  wordWrap?: boolean
  trim?: boolean
}

const wrapAnsiBun = typeof Bun !== 'undefined' && typeof Bun.wrapAnsi === 'function' ? Bun.wrapAnsi : null
const wrapAnsiNpm =
  typeof wrapAnsiModule === 'function'
    ? wrapAnsiModule
    : (wrapAnsiModule as { default: (input: string, columns: number, options?: WrapAnsiOptions) => string }).default

const wrapAnsi: (input: string, columns: number, options?: WrapAnsiOptions) => string = wrapAnsiBun ?? wrapAnsiNpm

export { wrapAnsi }
