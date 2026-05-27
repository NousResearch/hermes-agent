import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

const BACKEND = process.env.HERMES_DASHBOARD_URL ?? "http://127.0.0.1:9119";

/**
 * In production the Python `hermes dashboard` server injects a one-shot
 * session token into `index.html` (see `hermes_cli/web_server.py`). The
 * Vite dev server serves its own `index.html`, so unless we forward that
 * token, every protected `/api/*` call 401s.
 *
 * This plugin fetches the running dashboard's `index.html` on each dev page
 * load, scrapes the `window.__HERMES_SESSION_TOKEN__` assignment, and
 * re-injects it into the dev HTML. No-op in production builds.
 */
function hermesDevToken(): Plugin {
  const TOKEN_RE = /window\.__HERMES_SESSION_TOKEN__\s*=\s*"([^"]+)"/;
  const EMBEDDED_RE =
    /window\.__HERMES_DASHBOARD_EMBEDDED_CHAT__\s*=\s*(true|false)/;
  const LEGACY_TUI_RE =
    /window\.__HERMES_DASHBOARD_TUI__\s*=\s*(true|false)/;

  return {
    name: "hermes:dev-session-token",
    apply: "serve",
    async transformIndexHtml() {
      try {
        const res = await fetch(BACKEND, { headers: { accept: "text/html" } });
        const html = await res.text();
        const match = html.match(TOKEN_RE);
        if (!match) {
          console.warn(
            `[hermes] Could not find session token in ${BACKEND} — ` +
              `is \`hermes dashboard\` running? /api calls will 401.`,
          );
          return;
        }
        const embeddedMatch = html.match(EMBEDDED_RE);
        const legacyMatch = html.match(LEGACY_TUI_RE);
        const embeddedJs = embeddedMatch
          ? embeddedMatch[1]
          : legacyMatch
            ? legacyMatch[1]
            : "false";
        return [
          {
            tag: "script",
            injectTo: "head",
            children:
              `window.__HERMES_SESSION_TOKEN__="${match[1]}";` +
              `window.__HERMES_DASHBOARD_EMBEDDED_CHAT__=${embeddedJs};`,
          },
        ];
      } catch (err) {
        console.warn(
          `[hermes] Dashboard at ${BACKEND} unreachable — ` +
            `start with \`hermes dashboard\` or set HERMES_DASHBOARD_URL. ` +
            `(${(err as Error).message})`,
        );
      }
    },
  };
}


export default defineConfig({
    plugins: [
    react(),
    tailwindcss(),
    hermesDevToken(),
    // Inject crypto.randomUUID polyfill before any app code runs.
    // Handles @xterm/xterm@6 which calls it unconditionally.
    {
      name: "hermes:crypto-polyfill",
      transformIndexHtml(_html: string) {
        return [{
          tag: "script",
          injectTo: "head",
          children: `/* Hermes crypto.randomUUID polyfill */
(function(){
  try{
    if(!crypto.randomUUID||typeof crypto.randomUUID!=="function"){
      crypto.randomUUID=function(){return"rnd-"+Math.random().toString(36).slice(2)+"-"+Date.now().toString(36)};
    }
  }catch(e){}
})();`,
        }];
      },
    },
    // Patch ALL chunks (including lazy-loaded ones like xterm WebGL addon)
    // that call crypto.randomUUID without checking if it's callable.
    // The pattern comes from @xterm/xterm@6's ESBuild output.
    {
      name: "hermes:crypto-randomuuid-safe-call",
      apply: "build",
      transform(code: string, _id: string) {
        // Universal pattern: match ANY module with this exact minified form
        const OLD = 'typeof crypto<"u"&&"randomUUID"in crypto?crypto.randomUUID():';
        const NEW = 'typeof crypto<"u"&&"randomUUID"in crypto&&"function"===typeof crypto.randomUUID?crypto.randomUUID():';
        if (!code.includes(OLD)) return undefined;
        const patched = code.split(OLD).join(NEW);
        return { code: patched, map: null };
      },
    },
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
    dedupe: [
      "react",
      "react-dom",
      "@react-three/fiber",
      "@observablehq/plot",
      "three",
      "leva",
      "gsap",
    ],
  },
  build: {
    outDir: "../hermes_cli/web_dist",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/api": {
        target: BACKEND,
        ws: true,
      },
      "/dashboard-plugins": BACKEND,
    },
  },
});
