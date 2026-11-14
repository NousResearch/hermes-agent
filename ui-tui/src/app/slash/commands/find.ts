import { patchOverlayState } from '../../overlayStore.js'
import type { SlashCommand } from '../types.js'

export const findCommands: SlashCommand[] = [
  {
    help: 'search this transcript: type to narrow, ↑↓ older/newer match, ↵ stay there, esc back',
    name: 'find',
    run: (arg, ctx) => patchOverlayState({ find: { getRows: ctx.local.getHistoryItems, query: arg.trim() } })
  }
]
