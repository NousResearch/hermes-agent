import type { SlashCatalog } from '../types.js'

export interface CommandPaletteCommand {
  aliases?: string[]
  help?: string
  name: string
  usage?: string
}

export interface CommandPaletteItem {
  aliases: string[]
  command: string
  description: string
  lane: 'catalog' | 'core'
  rune: '☤' | '◇'
  searchText: string
  usage: string
}

const normalizeCommand = (value: string) => `/${value.trim().replace(/^\/+/, '')}`

const unique = (items: string[]) => Array.from(new Set(items.filter(Boolean)))

export function buildCommandPaletteItems(
  commands: CommandPaletteCommand[],
  catalog: null | SlashCatalog
): CommandPaletteItem[] {
  const byCommand = new Map<string, CommandPaletteItem>()
  const catalogHelp = new Map<string, string>()

  for (const [cmd, desc] of catalog?.pairs ?? []) {
    catalogHelp.set(normalizeCommand(cmd), desc)
  }

  const catalogAliases = new Map<string, string[]>()

  for (const [alias, target] of Object.entries(catalog?.canon ?? {})) {
    const cleanAlias = alias.replace(/^\/+/, '')
    const cleanTarget = normalizeCommand(target)

    if (cleanAlias && normalizeCommand(cleanAlias) !== cleanTarget) {
      catalogAliases.set(cleanTarget, [...(catalogAliases.get(cleanTarget) ?? []), cleanAlias])
    }
  }

  for (const cmd of commands) {
    const command = normalizeCommand(cmd.name)
    const aliases = unique([...(cmd.aliases ?? []), ...(catalogAliases.get(command) ?? [])].map(a => a.replace(/^\/+/, '')))
    const description = cmd.help ?? catalogHelp.get(command) ?? ''
    const usage = cmd.usage ?? command

    byCommand.set(command, {
      aliases,
      command,
      description,
      lane: 'core',
      rune: '☤',
      searchText: unique([command, cmd.name, description, usage, ...aliases]).join(' ').toLowerCase(),
      usage
    })
  }

  for (const [cmd, desc] of catalog?.pairs ?? []) {
    const command = normalizeCommand(cmd)

    if (!byCommand.has(command)) {
      const name = command.slice(1)
      const aliases = unique((catalogAliases.get(command) ?? []).map(a => a.replace(/^\/+/, '')))

      byCommand.set(command, {
        aliases,
        command,
        description: desc,
        lane: 'catalog',
        rune: '◇',
        searchText: unique([command, name, desc, command, ...aliases]).join(' ').toLowerCase(),
        usage: command
      })
    }
  }

  return Array.from(byCommand.values())
}

const itemMatchesPart = (item: CommandPaletteItem, part: string) => {
  if (part.length === 1) {
    return item.command.slice(1).startsWith(part) || item.aliases.some(alias => alias === part || alias.startsWith(part))
  }

  return item.searchText.includes(part)
}

export function filterCommandPaletteItems(items: CommandPaletteItem[], query: string): CommandPaletteItem[] {
  const parts = query
    .trim()
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)

  if (!parts.length) {
    return items
  }

  return items.filter(item => parts.every(part => itemMatchesPart(item, part)))
}
