export const INTERPOLATION_RE = /\{!(.+?)\}/g;
export const hasInterpolation = (s) => /\{!.+?\}/.test(s);
