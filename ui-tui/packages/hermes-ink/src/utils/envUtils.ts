export function isEnvTruthy(envVar: string | boolean | undefined): boolean {
  if (!envVar) {
    return false
  }

  if (typeof envVar === 'boolean') {
    return envVar
  }

  const v = envVar.toLowerCase().trim()

  return ['1', 'true', 'yes', 'on'].includes(v)
}
