export interface ParsedSlashCommand {
  arg: string
  cmd: string
  name: string
}

export const looksLikeSlashCommand = (text: string) => {
  if (!text.startsWith('/')) {
    return false
  }

  const first = text.split(/\s+/, 1)[0] || ''

  return !first.slice(1).includes('/')
}

export const parseSlashCommand = (cmd: string): ParsedSlashCommand => {
  const [rawName = '', ...rest] = cmd.slice(1).split(/\s+/)

  return {
    arg: rest.join(' '),
    cmd,
    name: rawName.toLowerCase()
  }
}
