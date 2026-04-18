export const looksLikeSlashCommand = (text: string) => /^\/[^\s/]*(?:\s|$)/.test(text)

export const parseSlashCommand = (cmd: string) => {
  const [name = '', ...rest] = cmd.slice(1).split(/\s+/)

  return { arg: rest.join(' '), cmd, name: name.toLowerCase() }
}
