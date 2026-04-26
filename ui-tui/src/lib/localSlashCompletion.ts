import type { CompletionResponse, GatewayCompletionItem } from '../gatewayTypes.js'

import type { CommandPaletteCommand } from './commandPalette.js'

const normalizeSlash = (value: string) => `/${value.replace(/^\/+/, '')}`

const queryFromText = (text: string) => text.trimStart().replace(/^\/+/, '').toLowerCase()

export function localSlashCompletionItems(commands: CommandPaletteCommand[], text: string): GatewayCompletionItem[] {
  const query = queryFromText(text)

  return commands.flatMap(cmd => {
    const names = [cmd.name, ...(cmd.aliases ?? [])]
    const matches = !query || names.some(name => name.toLowerCase().startsWith(query))

    if (!matches) {
      return []
    }

    const command = normalizeSlash(cmd.name)
    const aliasText = cmd.aliases?.length ? ` · ${cmd.aliases.map(normalizeSlash).join(', ')}` : ''

    return [
      {
        display: command,
        meta: `${cmd.help ?? ''}${aliasText}`.trim() || undefined,
        text: command
      }
    ]
  })
}

export function mergeLocalSlashCompletions(
  response: null | CompletionResponse,
  commands: CommandPaletteCommand[],
  text: string
): CompletionResponse {
  const remoteItems = response?.items ?? []
  const seen = new Set(remoteItems.map(item => normalizeSlash(item.text).toLowerCase()))

  const localItems = localSlashCompletionItems(commands, text).filter(item => {
    const key = normalizeSlash(item.text).toLowerCase()

    if (seen.has(key)) {
      return false
    }

    seen.add(key)

    return true
  })

  return {
    ...response,
    items: [...remoteItems, ...localItems],
    replace_from: response?.replace_from ?? 1
  }
}
