export const INTERPOLATION_RE = /\{!(.+?)\}/g

export const hasInterpolation = (s: string) => {
  INTERPOLATION_RE.lastIndex = 0

  return INTERPOLATION_RE.test(s)
}
