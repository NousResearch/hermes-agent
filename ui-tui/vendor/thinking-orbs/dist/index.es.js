import { jsx as R } from "react/jsx-runtime";
import { useState as O, useEffect as M, useRef as F } from "react";
import { resolvePreset as C, MODE_DRAWS as I } from "./engine.es.js";
import { STATE_TO_MODE as N } from "./engine.es.js";
function P(t) {
  let e = t;
  for (; e; ) {
    const n = e.getAttribute("data-theme");
    if (n === "dark") return !0;
    if (n === "light") return !1;
    if (e.classList.contains("dark")) return !0;
    if (e.classList.contains("light")) return !1;
    e = e.parentElement;
  }
  return null;
}
function W() {
  return typeof matchMedia > "u" || matchMedia("(prefers-color-scheme: dark)").matches;
}
function _(t, e) {
  const [n, r] = O(!0);
  return M(() => {
    if (t === "dark") {
      r(!0);
      return;
    }
    if (t === "light") {
      r(!1);
      return;
    }
    const i = () => {
      const c = P(e.current);
      r(c ?? W());
    };
    i();
    const o = typeof matchMedia < "u" ? matchMedia("(prefers-color-scheme: dark)") : null, d = () => i();
    o == null || o.addEventListener("change", d);
    let s = null;
    return typeof MutationObserver < "u" && e.current && (s = new MutationObserver(i), s.observe(document.documentElement, {
      attributes: !0,
      attributeFilter: ["class", "data-theme"],
      subtree: !0
    })), () => {
      o == null || o.removeEventListener("change", d), s == null || s.disconnect();
    };
  }, [t, e]), n;
}
function j() {
  const [t, e] = O(!1);
  return M(() => {
    if (typeof matchMedia > "u") return;
    const n = matchMedia("(prefers-reduced-motion: reduce)");
    e(n.matches);
    const r = (i) => e(i.matches);
    return n.addEventListener("change", r), () => n.removeEventListener("change", r);
  }, []), t;
}
const q = {
  working: "Working…",
  searching: "Searching…",
  solving: "Solving…",
  listening: "Listening…",
  connecting: "Connecting…",
  weaving: "Weaving…",
  composing: "Composing…",
  breathing: "Thinking…",
  shaping: "Shaping…"
};
function H({
  state: t = "working",
  size: e = 64,
  theme: n = "auto",
  speed: r = 1,
  paused: i = !1,
  style: o,
  "aria-label": d,
  ...s
}) {
  const c = F(null), y = _(n, c), E = j();
  return M(() => {
    const u = c.current;
    if (!u) return;
    const l = Math.min(2, typeof devicePixelRatio < "u" && devicePixelRatio || 1);
    u.width = Math.round(e * l), u.height = Math.round(e * l);
    const f = u.getContext("2d");
    if (!f) return;
    const { mode: T, speed: x, opts: A } = C(t, e), D = I[T], w = x * r, h = (k) => {
      f.setTransform(l, 0, 0, l, 0, 0), f.clearRect(0, 0, e, e), D(f, e, k, y, A);
    };
    if (E) {
      h(0.6);
      return;
    }
    let g = 0, m = !1;
    const L = () => {
      h(performance.now() / 1e3 * w), m && (g = requestAnimationFrame(L));
    }, v = () => {
      m || i || (m = !0, g = requestAnimationFrame(L));
    }, p = () => {
      m = !1, cancelAnimationFrame(g);
    };
    h(performance.now() / 1e3 * w);
    let b = !0;
    const a = typeof IntersectionObserver < "u" ? new IntersectionObserver(([k]) => {
      b = k.isIntersecting, b && document.visibilityState !== "hidden" ? v() : p();
    }) : null;
    a == null || a.observe(u);
    const S = () => {
      document.visibilityState === "hidden" ? p() : b && v();
    };
    return document.addEventListener("visibilitychange", S), a || v(), () => {
      p(), a == null || a.disconnect(), document.removeEventListener("visibilitychange", S);
    };
  }, [t, e, y, r, i, E]), /* @__PURE__ */ R(
    "canvas",
    {
      ref: c,
      role: "img",
      "aria-label": d ?? q[t],
      style: { width: e, height: e, display: "block", ...o },
      ...s
    }
  );
}
export {
  I as MODE_DRAWS,
  N as STATE_TO_MODE,
  H as ThinkingOrb,
  C as resolvePreset
};
