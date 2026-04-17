export const shortCwd = (cwd: string, max = 28) => {
  const h = process.env.HOME
  const p = h && cwd.startsWith(h) ? `~${cwd.slice(h.length)}` : cwd

  return p.length <= max ? p : `…${p.slice(-(max - 1))}`
}
