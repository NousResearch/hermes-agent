export const shortCwd = (cwd: string, max = 28) => {
  const h = process.env.HOME
  const p = h && cwd.startsWith(h) ? `~${cwd.slice(h.length)}` : cwd

  return p.length <= max ? p : `…${p.slice(-(max - 1))}`
}

export const fmtCwdBranch = (cwd: string, branch: null | string, max = 40) => {
  if (!branch) {
    return shortCwd(cwd, max)
  }

  const b = branch.length > 16 ? `…${branch.slice(-15)}` : branch
  const tag = ` (${b})`

  return `${shortCwd(cwd, Math.max(8, max - tag.length))}${tag}`
}
