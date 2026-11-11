/** Shared list keys; shifted arrows belong to transcript scrolling. */
export interface ListKey {
  ctrl?: boolean
  downArrow?: boolean
  meta?: boolean
  shift?: boolean
  super?: boolean
  upArrow?: boolean
}

export const isListUp = (input: string, key: ListKey): boolean =>
  !key.shift && (!!key.upArrow || (!!key.ctrl && !key.meta && !key.super && (input === 'p' || input === '\x10')))

export const isListDown = (input: string, key: ListKey): boolean =>
  !key.shift && (!!key.downArrow || (!!key.ctrl && !key.meta && !key.super && (input === 'n' || input === '\x0e')))
