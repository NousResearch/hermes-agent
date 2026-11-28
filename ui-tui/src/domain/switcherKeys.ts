/** What a keypress means in the session switcher, before any state is touched. */
export type SwitcherIntent =
  | 'activate'
  | 'arm-delete'
  | 'cancel'
  | 'clear-query'
  | 'close-live'
  | 'delete-confirm'
  | 'delete-dismiss'
  | 'down'
  | 'new'
  | 'none'
  | 'open-search'
  | 'pick-model'
  | 'refresh'
  | 'search-cancel'
  | 'search-down'
  | 'search-up'
  | 'toggle-group'
  | 'up'

export type SwitcherRowKind = 'header' | 'history' | 'live' | 'new'

export interface SwitcherKeyState {
  busy: boolean
  canMoveDown: boolean
  canMoveUp: boolean
  confirmingDelete: boolean
  draftHasText: boolean
  hasQuery: boolean
  kind: SwitcherRowKind
  searching: boolean
}

export interface SwitcherKey {
  ctrl?: boolean
  downArrow?: boolean
  escape?: boolean
  meta?: boolean
  return?: boolean
  tab?: boolean
  upArrow?: boolean
}

const CTRL_OFFSET = 96

const ctrlChar = (letter: string) => String.fromCharCode(letter.charCodeAt(0) - CTRL_OFFSET)

interface Input {
  ch: string
  key: SwitcherKey
  lower: string
  s: SwitcherKeyState
}

const ctrl = ({ ch, key, lower }: Input, letter: string) =>
  Boolean(key.ctrl) && (lower === letter || ch === ctrlChar(letter))

const plain = ({ key }: Input) => !key.ctrl && !key.meta

// On the New row letters belong to the prompt's TextInput, so row hotkeys only apply elsewhere.
const onRow = (i: Input) => plain(i) && i.s.kind !== 'new'

/**
 * Ordered rules, first match wins; an order is a priority. The armed delete prompt and the
 * search box capture every key before any hotkey sees it.
 */
const RULES: ReadonlyArray<readonly [(i: Input) => boolean, SwitcherIntent | ((i: Input) => SwitcherIntent)]> = [
  [i => i.s.busy, 'none'],
  [i => i.s.confirmingDelete, i => (i.lower === 'd' ? 'delete-confirm' : 'delete-dismiss')],
  [
    i => i.s.searching,
    i => (i.key.escape ? 'search-cancel' : i.key.upArrow ? 'search-up' : i.key.downArrow ? 'search-down' : 'none')
  ],
  [i => Boolean(i.key.escape), i => (i.s.hasQuery ? 'clear-query' : 'cancel')],
  [i => ctrl(i, 'n'), 'new'],
  [i => ctrl(i, 'r'), 'refresh'],
  [i => Boolean(i.key.tab), i => (i.s.kind === 'new' ? 'pick-model' : 'none')],
  [i => ctrl(i, 'd'), i => (i.s.kind === 'live' ? 'close-live' : 'none')],
  [i => i.lower === 'd' && plain(i) && i.s.kind === 'history', 'arm-delete'],
  [i => i.ch === '/' && onRow(i), 'open-search'],
  [i => i.lower === 'g' && onRow(i), 'toggle-group'],
  [i => i.s.kind === 'new' && i.s.draftHasText, 'none'],
  [i => Boolean(i.key.upArrow) && i.s.canMoveUp, 'up'],
  [i => Boolean(i.key.downArrow) && i.s.canMoveDown, 'down'],
  [i => Boolean(i.key.return), 'activate']
]

export function switcherKeyIntent(s: SwitcherKeyState, ch: string | undefined, key: SwitcherKey): SwitcherIntent {
  const input: Input = { ch: ch ?? '', key, lower: ch?.toLowerCase() ?? '', s }
  const rule = RULES.find(([matches]) => matches(input))

  if (!rule) {
    return 'none'
  }

  const intent = rule[1]

  return typeof intent === 'function' ? intent(input) : intent
}
