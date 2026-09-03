import { resolvePreset as P, MODE_FRAMES as T } from "./engine.es.js";
const E = 64, X = 10240, H = [
  [1, 8],
  [2, 16],
  [4, 32],
  [64, 128]
], U = [
  [0, 8, 2, 10],
  [12, 4, 14, 6],
  [3, 11, 1, 9],
  [15, 7, 13, 5]
];
function p(s) {
  return Math.max(0, Math.min(1, s));
}
function Y(s, t, a) {
  const o = p(a);
  s[t] = 1 - (1 - s[t]) * (1 - o);
}
function C(s, t, a) {
  return (a ? 1 - p(s) : p(s)) * p(t);
}
function W(s, t, a, o, n, h, m, y) {
  const i = o.x * n + h, f = o.y * n + m, l = Math.max(0.55, o.r * n), d = C(o.white, o.a ?? 1, y), M = Math.max(0, Math.floor(i - l - 0.5)), g = Math.min(t - 1, Math.ceil(i + l + 0.5)), u = Math.max(0, Math.floor(f - l - 0.5)), x = Math.min(a - 1, Math.ceil(f + l + 0.5));
  for (let c = u; c <= x; c++)
    for (let e = M; e <= g; e++) {
      const r = Math.hypot(e + 0.5 - i, c + 0.5 - f), A = p(l + 0.7 - r);
      A > 0 && Y(s, c * t + e, d * A);
    }
}
function Z(s, t, a, o, n, h, m, y) {
  const i = o.x1 * n + h, f = o.y1 * n + m, l = o.x2 * n + h, d = o.y2 * n + m, M = Math.max(1, Math.ceil(Math.hypot(l - i, d - f) * 1.5)), g = C(o.white, o.a ?? 1, y);
  for (let u = 0; u <= M; u++) {
    const x = u / M, c = Math.round(i + (l - i) * x), e = Math.round(f + (d - f) * x);
    c >= 0 && c < t && e >= 0 && e < a && Y(s, e * t + c, g);
  }
}
function q(s, t = {}) {
  const a = Math.max(4, Math.floor(t.columns ?? 16)), o = Math.max(2, Math.floor(t.rows ?? 8)), n = a * 2, h = o * 4, m = Math.min((n - 1) / E, (h - 1) / E), y = (n - E * m) / 2, i = (h - E * m) / 2, f = (t.theme ?? "dark") === "dark", l = p(t.threshold ?? 0.18), d = t.paint ?? ((r) => r), M = P(s, E), g = Math.max(0, t.time ?? 0) * M.speed * Math.max(0, t.speed ?? 1), u = T[M.mode](E, g, M.opts), x = new Float32Array(n * h);
  for (const r of u.lines) Z(x, n, h, r, m, y, i, f);
  for (const r of u.dots) W(x, n, h, r, m, y, i, f);
  const c = [], e = [];
  for (let r = 0; r < o; r++) {
    let A = "";
    const R = [];
    for (let k = 0; k < a; k++) {
      let B = 0, I = 0, _ = 0;
      for (let S = 0; S < 4; S++)
        for (let b = 0; b < 2; b++) {
          const w = k * 2 + b, O = r * 4 + S, L = x[O * n + w];
          I = Math.max(I, L), _ += L;
          const F = (U[O % 4][w % 4] / 15 - 0.5) * 0.16;
          L >= l + F && (B |= H[S][b]);
        }
      const v = p(I * 0.72 + _ / 8 * 0.28);
      R.push(v);
      const D = B === 0 ? " " : String.fromCodePoint(X + B);
      A += d(D, v);
    }
    c.push(A), e.push(R);
  }
  return { lines: c, intensities: e };
}
export {
  q as renderTuiOrb
};
