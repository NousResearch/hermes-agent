import { useEffect, useState } from "react";

/**
 * ?debug=inset diagnostic overlay (mobile verification aid).
 *
 * Renders a small fixed chip with the live viewport geometry so the inset
 * fixes can be verified on a real device without a computer:
 *
 *   layout : innerHeight (the layout viewport / ICB height)
 *   visible: visualViewport.height (what the browser actually shows)
 *   raw    : layout − visible — the region the browser is hiding
 *            (keyboard and/or URL bar); what the tracker must compensate
 *   inset  : the applied `--hermes-mobile-bottom-inset`
 *   shell  : the live height of the app shell (`.hermes-dvh-surface`) —
 *            the mobile layout shrinks the fixed column to
 *            `100dvh - inset` via `--app-h`, so `shell ≤ visible` means
 *            the whole UI (prompt row included) sits above the
 *            keyboard/URL bar.
 *
 * The chip is `position: fixed; top: 0` — the top of the screen, far from
 * the bottom keyboard/URL bar — so it stays readable while the keyboard is
 * up. It is rendered by main.tsx only when `?debug=inset` is in the URL,
 * so it never ships visible in normal use.
 */
export function InsetDebugSentinel() {
  const [line, setLine] = useState("…");

  useEffect(() => {
    const sample = () => {
      const vv = window.visualViewport;
      const inset =
        getComputedStyle(document.documentElement).getPropertyValue(
          "--hermes-mobile-bottom-inset",
        ).trim() || "0px";
      const layout = window.innerHeight;
      const visible = vv ? Math.round(vv.height) : null;
      const raw = visible !== null ? layout - visible : null;
      const shell = document.querySelector(".hermes-dvh-surface");
      const shellH = shell ? Math.round(shell.getBoundingClientRect().height) : null;
      const fits =
        visible !== null && shellH !== null
          ? (shellH <= visible ? "FITS" : "CLIPPED")
          : "?";
      // Terminal geometry (written by ChatPage when xterm is connected) —
      // distinguishes "host didn't shrink" from "grid didn't re-fit" from
      // "PTY didn't re-render at the new row count".
      const t = (
        window as unknown as {
          __HERMES_TERM_DEBUG__?: {
            rows: number;
            cols: number;
            fs: number;
            hostTop: number;
            hostBot: number;
            hostH: number;
            docTop: number;
            offBottom: number;
            kbReserve: number;
          };
        }
      ).__HERMES_TERM_DEBUG__;
      // Fresh host geometry, measured here on every sample (independent of
      // whether ChatPage's refit path actually ran).
      const termHost = document.querySelector(".hermes-chat-xterm-host");
      const hostH = termHost
        ? Math.round(termHost.getBoundingClientRect().height)
        : null;
      const termLine =
        hostH !== null || t
          ? `\nhost h${hostH ?? "n/a"} top${t?.hostTop ?? "?"} bot${t?.hostBot ?? "?"} · grid ${
              t ? `${t.rows}r x ${t.cols}c` : "n/a (no refit yet)"
            } @${t ? t.fs : "?"}px · offBot ${t ? t.offBottom : "?"} · docTop ${
              t ? t.docTop : "?"
            } · kbRow ${t ? t.kbReserve : "?"}`
          : "";
      setLine(
        `layout ${layout} · visible ${visible ?? "n/a"} · raw ${raw ?? "n/a"}\n` +
          `inset ${inset} · shell ${shellH ?? "n/a"} · ${fits}` +
          termLine,
      );
    };
    sample();
    const timer = window.setInterval(sample, 500);
    return () => window.clearInterval(timer);
  }, []);

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        zIndex: 9999,
        pointerEvents: "none",
        padding: "2px 8px",
        fontSize: "11px",
        lineHeight: 1.4,
        whiteSpace: "pre-wrap",
        fontFamily: "ui-monospace, Menlo, monospace",
        color: "#0f0",
        background: "rgba(0, 0, 0, 0.75)",
      }}
    >
      {line}
    </div>
  );
}