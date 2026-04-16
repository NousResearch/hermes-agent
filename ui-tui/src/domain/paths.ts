export const shortCwd = (cwd: string, max = 28) => {
  const p = process.env.HOME && cwd.startsWith(process.env.HOME) ? `~${cwd.slice(process.env.HOME.length)}` : cwd

  return p.length <= max ? p : `…${p.slice(-(max - 1))}`
}
