import { createContext } from 'react'

import { EventEmitter } from '../events/emitter.js'
import type { TerminalQuerier } from '../terminal-querier.js'

export type Props = {
  readonly stdin: NodeJS.ReadStream
  readonly setRawMode: (value: boolean) => void
  readonly isRawModeSupported: boolean
  readonly exitOnCtrlC: boolean
  readonly inputEmitter: EventEmitter
  readonly querier: TerminalQuerier | null
}

const StdinContext = createContext<Props>({
  stdin: process.stdin,
  inputEmitter: new EventEmitter(),
  setRawMode() {},
  isRawModeSupported: false,
  exitOnCtrlC: true,
  querier: null
})

StdinContext.displayName = 'StdinContext'
export default StdinContext
