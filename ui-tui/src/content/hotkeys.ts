import type { TranslationKey } from '../i18n/index.js'
import { isMac, isRemoteShell } from '../lib/platform.js'

const action = isMac ? 'Cmd' : 'Ctrl'
const paste = isMac ? 'Cmd' : 'Alt'

const copyHotkeys: [string, TranslationKey][] = isMac
  ? [
      ['Cmd+C', 'hotkey.copySelection'],
      ['Ctrl+C', 'hotkey.interruptClearExit']
    ]
  : isRemoteShell()
    ? [
        ['Cmd+C', 'hotkey.forwardCopySelection'],
        ['Ctrl+C', 'hotkey.copySelectionInterruptClearExit']
      ]
    : [['Ctrl+C', 'hotkey.copySelectionInterruptClearExit']]

export const HOTKEYS: [string, TranslationKey][] = [
  ...copyHotkeys,
  [action + '+D', 'hotkey.quit'],
  [action + '+G / Alt+G', 'hotkey.openEditor'],
  [action + '+L', 'hotkey.redraw'],
  [paste + '+V / /paste', 'hotkey.pasteTextOrImage'],
  ['Tab', 'hotkey.applyCompletion'],
  ['↑/↓', 'hotkey.navigateOrEdit'],
  ['Ctrl+X', 'hotkey.deleteQueuedMsg'],
  [action + '+A/E', 'hotkey.lineStartEnd'],
  [action + '+Z / ' + action + '+Y', 'hotkey.undoRedo'],
  [action + '+W', 'hotkey.deleteWord'],
  [action + '+U/K', 'hotkey.deleteToLineEnds'],
  [action + '+←/→', 'hotkey.jumpWord'],
  ['Home/End', 'hotkey.lineStartEnd'],
  ['Shift+Enter / Alt+Enter', 'hotkey.insertNewline'],
  ['\\\\+Enter', 'hotkey.multilineCont'],
  ['!<cmd>', 'hotkey.runShellCmd'],
  ['{!<cmd>}', 'hotkey.inlineShellCmd']
]
